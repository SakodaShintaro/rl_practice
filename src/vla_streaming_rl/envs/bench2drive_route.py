# SPDX-License-Identifier: MIT
"""Parser for Bench2Drive route XML files.

Consumes ``<waypoints>`` and the first ``<weathers>`` entry; ``<scenarios>``
(cut-ins, pedestrian crossings, etc.) is deliberately ignored because the
training environment does not run the Bench2Drive scenario engine.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import carla

# Fields accepted by ``carla.WeatherParameters`` that may appear as
# attributes of ``<weather>`` nodes in Bench2Drive route XMLs.
_WEATHER_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "wetness",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
)


@dataclass(frozen=True)
class RouteInfo:
    id: str
    town: str
    waypoints: list[carla.Location]
    start_pose: carla.Transform  # trigger_point of the first ``<scenario>``
    weather: carla.WeatherParameters | None


def _parse_start_pose(route_node: ET.Element) -> carla.Transform:
    """Return the ego start pose from the first scenario's ``trigger_point``."""
    tp = route_node.find("scenarios").find("scenario").find("trigger_point")
    return carla.Transform(
        carla.Location(x=float(tp.get("x")), y=float(tp.get("y")), z=float(tp.get("z"))),
        carla.Rotation(yaw=float(tp.get("yaw"))),
    )


def _parse_weather(route_node: ET.Element) -> carla.WeatherParameters | None:
    """Return the ``route_percentage=0`` weather, or None if absent."""
    weathers_node = route_node.find("weathers")
    if weathers_node is None:
        return None
    entries = weathers_node.findall("weather")
    if not entries:
        return None
    # Pick the entry that applies at the route start.
    start_entry = min(
        entries, key=lambda w: float(w.get("route_percentage", "0"))
    )
    kwargs = {}
    for field in _WEATHER_FIELDS:
        val = start_entry.get(field)
        if val is not None:
            kwargs[field] = float(val)
    return carla.WeatherParameters(**kwargs)


def parse_bench2drive_routes(
    xml_path: str | Path, route_id: str | None
) -> list[RouteInfo]:
    """Return routes from a Bench2Drive-format XML.

    Pass ``route_id=None`` to return every route, or a specific id to
    filter. Raises ``ValueError`` if no matching route is found.
    """
    tree = ET.parse(str(xml_path))
    routes: list[RouteInfo] = []
    for route in tree.getroot().findall("route"):
        rid = route.get("id")
        if route_id is not None and rid != route_id:
            continue
        town = route.get("town")
        waypoints_node = route.find("waypoints")
        if town is None or waypoints_node is None:
            continue
        waypoints = [
            carla.Location(
                float(p.get("x")), float(p.get("y")), float(p.get("z"))
            )
            for p in waypoints_node.findall("position")
        ]
        if waypoints:
            routes.append(
                RouteInfo(
                    id=rid,
                    town=town,
                    waypoints=waypoints,
                    start_pose=_parse_start_pose(route),
                    weather=_parse_weather(route),
                )
            )
    if not routes:
        raise ValueError(
            f"no usable routes in {xml_path}"
            + (f" (route_id={route_id})" if route_id else "")
        )
    return routes
