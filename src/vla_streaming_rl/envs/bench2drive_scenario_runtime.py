# SPDX-License-Identifier: MIT
"""Drive a Bench2Drive ``RouteScenario`` from inside our gym env.

The leaderboard evaluator wraps ``RouteScenario`` (ego spawn, scripted
scenarios, ``BackgroundBehavior`` traffic, parked vehicles) inside a
``ScenarioManager`` thread loop. The training env doesn't run that
manager — this runtime exposes a small ``reset``/``tick``/``cleanup``
surface that the env can drive directly so NPC behavior matches eval.
"""
import os
import threading

import carla
import numpy as np
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_parser import RouteParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


class Bench2DriveRuntime:
    """Owns the per-episode ``RouteScenario`` lifecycle for one env."""

    # build_scenarios runs in a background thread (mirrors eval's
    # build_scenarios_loop) so that a single scenario class init blocking on
    # the CARLA RPC client_timeout (300 s on Town12 in practice) does not
    # freeze env.step.
    BUILD_LOOP_INTERVAL_S = 1.0

    def __init__(
        self,
        client: carla.Client,
        traffic_manager_port: int,
        route_xml: str,
        route_id: str | int | None,
    ):
        # RouteScenario glob's srunner/scenarios/*.py via this env var to
        # discover scenario classes named in the XML — without it, no
        # scripted scenarios fire.
        if not os.getenv("SCENARIO_RUNNER_ROOT"):
            raise RuntimeError(
                "SCENARIO_RUNNER_ROOT must be set to .../Bench2Drive/scenario_runner "
                "so RouteScenario can discover scenario classes"
            )

        self.client = client
        self.traffic_manager_port = traffic_manager_port

        subset = "" if route_id is None else str(route_id)
        self.configs = RouteParser.parse_routes_file(route_xml, subset)
        if not self.configs:
            raise ValueError(f"no routes parsed from {route_xml} (subset={subset!r})")

        towns = {c.town for c in self.configs}
        if len(towns) > 1:
            raise ValueError(
                f"route XML mixes towns {sorted(towns)} — training env requires one town"
            )
        self.town: str = next(iter(towns))

        self.route_scenario: RouteScenario | None = None
        self._build_thread: threading.Thread | None = None
        self._build_stop = threading.Event()

    def reset(
        self, world: carla.World
    ) -> tuple[carla.Vehicle, list[carla.Location]]:
        """Build a fresh ``RouteScenario`` for ``world`` and return ``(ego, route_locations)``."""
        config = self.configs[np.random.randint(len(self.configs))]

        # set_world rebuilds the GlobalRoutePlanner from OpenDRIVE which
        # takes ~30 s on Town12. cleanup() keeps _world/_map/_grp around so
        # repeat resets on the same world skip this. Identity check (not
        # "is None") so a different world (e.g. another env in the same
        # process, or a town swap) still re-initializes correctly.
        if CarlaDataProvider.get_world() is not world:
            CarlaDataProvider.set_client(self.client)
            CarlaDataProvider.set_world(world)
            CarlaDataProvider.set_traffic_manager_port(self.traffic_manager_port)
        GameTime.restart()

        # RouteScenario __init__ spawns ego, sets weather, builds initial
        # in-range scenarios, and prepares parking slots. Parked vehicles
        # are spawned lazily by spawn_parked_vehicles below.
        self.route_scenario = RouteScenario(world=world, config=config, debug_mode=0)

        ego = self.route_scenario.ego_vehicles[0]
        self.route_scenario.spawn_parked_vehicles(ego)

        # Background thread that periodically tries to start any scenario
        # the ego is now close to. Mirrors ScenarioManager's
        # build_scenarios_loop. A scenario class init that hits the
        # 300 s CARLA RPC timeout (e.g. AccidentTwoWays on Town12) would
        # otherwise stall env.step for the full timeout.
        self._build_stop.clear()
        self._build_thread = threading.Thread(
            target=self._build_loop, args=(ego,), daemon=True
        )
        self._build_thread.start()

        route_locations = [t.location for (t, _) in self.route_scenario.route]
        return ego, route_locations

    def _build_loop(self, ego: carla.Vehicle) -> None:
        """Background loop: try to start nearby scenarios + spawn parked cars."""
        while not self._build_stop.is_set():
            scenario = self.route_scenario  # snapshot in case cleanup nulls it
            if scenario is None:
                return
            try:
                scenario.build_scenarios(ego, debug=False)
                scenario.spawn_parked_vehicles(ego)
            except Exception:
                # Per-scenario setup errors are already logged inside
                # build_scenarios; swallow so the loop keeps running.
                pass
            self._build_stop.wait(self.BUILD_LOOP_INTERVAL_S)

    def tick(self, world: carla.World) -> None:
        """Advance scenario state one tick. Call after ``world.tick()``."""
        if self.route_scenario is None:
            return

        # Mirror ScenarioManager._tick_scenario's bookkeeping so behaviors
        # that key off GameTime / CarlaDataProvider see the current state.
        timestamp = world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()

        # build_scenarios + spawn_parked_vehicles are done off-thread; here
        # we only advance the behavior tree.
        self.route_scenario.scenario_tree.tick_once()
        self.route_scenario.prune_completed_scenarios()

    def cleanup(self) -> None:
        """Destroy scenario-owned actors and clear per-episode CDP state.

        Deliberately *not* using ``CarlaDataProvider.cleanup()`` because that
        nulls ``_grp``/``_map``/``_world``, forcing a multi-second
        ``GlobalRoutePlanner`` rebuild on the next ``set_world`` (very
        expensive on Town12/13). Traffic lights and the road graph are tied
        to the persistent map, so keeping them across episodes is safe.
        """
        # Stop the build loop. We only wait briefly: if it is mid-RPC
        # (e.g. a scenario __init__ blocking on actor spawn that will hit
        # the 300 s client_timeout) Python cannot interrupt it, so waiting
        # longer just delays the next episode for no benefit. The old thread
        # will keep running until its CARLA RPC returns, then exit on the
        # stop_event check; while it lives, calls into destroyed actors emit
        # harmless `get_location: ... not found!` warnings (build_scenarios
        # sees None and skips that iteration).
        if self._build_thread is not None:
            self._build_stop.set()
            self._build_thread.join(timeout=1.0)
            self._build_thread = None

        if self.route_scenario is not None:
            # terminate() propagates INVALID status to every leaf in
            # scenario_tree, which is what triggers CollisionTest /
            # OutsideRouteLanesTest / ... terminate() and therefore
            # sensor.stop() + sensor.destroy() on the per-criterion sensors.
            # Without this, those sensors keep their port-2001 streaming
            # sockets and listener threads open forever and every reset leaks.
            try:
                self.route_scenario.terminate()
            except Exception:
                pass

            # Destroys other_actors (NPCs, scripted scenario actors) and the
            # parked vehicles via __del__.
            self.route_scenario.remove_all_actors()
            self.route_scenario = None

        cdp = CarlaDataProvider
        # Destroy every actor CDP knows about (ego, scenario actors). Mirrors
        # what cleanup() does, minus the global-state nuke.
        destroy = carla.command.DestroyActor
        batch = [
            destroy(actor)
            for actor in cdp._carla_actor_pool.values()
            if actor is not None and actor.is_alive
        ]
        if cdp._client and batch:
            try:
                cdp._client.apply_batch_sync(batch)
            except RuntimeError as e:
                if "time-out" not in str(e):
                    raise

        # Per-episode bookkeeping reset (kept: _world/_map/_grp/_blueprint_library
        # /_spawn_points/_client/_traffic_manager_port/_traffic_light_map).
        cdp._carla_actor_pool = {}
        cdp._actor_velocity_map.clear()
        cdp._actor_location_map.clear()
        cdp._actor_transform_map.clear()
        cdp._spawn_index = 0
        cdp._all_actors = None
        cdp._ego_vehicle_route = None
        cdp._runtime_init_flag = False
