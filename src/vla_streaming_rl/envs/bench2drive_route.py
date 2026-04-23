# SPDX-License-Identifier: MIT
"""Parser for Bench2Drive route XML files.

Only the ``<waypoints>`` section is consumed; ``<scenarios>`` (cut-ins,
pedestrian crossings, etc.) is deliberately ignored because the training
environment does not run the Bench2Drive scenario engine.
"""
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import carla


@dataclass(frozen=True)
class RouteInfo:
    id: str
    town: str
    waypoints: list[carla.Location]


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
            routes.append(RouteInfo(id=rid, town=town, waypoints=waypoints))
    if not routes:
        raise ValueError(
            f"no usable routes in {xml_path}"
            + (f" (route_id={route_id})" if route_id else "")
        )
    return routes
