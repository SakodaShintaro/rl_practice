# SPDX-License-Identifier: MIT
"""Drive a Bench2Drive ``RouteScenario`` from inside our gym env.

The leaderboard evaluator wraps ``RouteScenario`` (ego spawn, scripted
scenarios, ``BackgroundBehavior`` traffic, parked vehicles) inside a
``ScenarioManager`` thread loop. The training env doesn't run that
manager — this runtime exposes a small ``reset``/``tick``/``cleanup``
surface that the env can drive directly so NPC behavior matches eval.
"""
import os

import carla
import numpy as np
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.utils.route_parser import RouteParser
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


class Bench2DriveRuntime:
    """Owns the per-episode ``RouteScenario`` lifecycle for one env."""

    # Re-trigger build_scenarios + spawn_parked_vehicles every N ticks. Eval
    # runs that loop every 1 s; matching it so newly-reachable scenarios
    # actually fire while the ego drives.
    BUILD_INTERVAL_TICKS = 20

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

    def reset(
        self, world: carla.World
    ) -> tuple[carla.Vehicle, list[carla.Location]]:
        """Build a fresh ``RouteScenario`` for ``world`` and return ``(ego, route_locations)``."""
        config = self.configs[np.random.randint(len(self.configs))]

        # CarlaDataProvider is a singleton reset by cleanup(); re-seed every episode.
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

        route_locations = [t.location for (t, _) in self.route_scenario.route]
        return ego, route_locations

    def tick(self, world: carla.World, episode_step: int) -> None:
        """Advance scenario state one tick. Call after ``world.tick()``."""
        if self.route_scenario is None:
            return

        # Mirror ScenarioManager._tick_scenario's bookkeeping so behaviors
        # that key off GameTime / CarlaDataProvider see the current state.
        timestamp = world.get_snapshot().timestamp
        GameTime.on_carla_tick(timestamp)
        CarlaDataProvider.on_carla_tick()

        if episode_step % self.BUILD_INTERVAL_TICKS == 0:
            ego = self.route_scenario.ego_vehicles[0]
            self.route_scenario.build_scenarios(ego, debug=False)
            self.route_scenario.spawn_parked_vehicles(ego)

        self.route_scenario.scenario_tree.tick_once()

    def cleanup(self) -> None:
        """Destroy all scenario-owned actors and reset CarlaDataProvider."""
        if self.route_scenario is not None:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
        CarlaDataProvider.cleanup()
