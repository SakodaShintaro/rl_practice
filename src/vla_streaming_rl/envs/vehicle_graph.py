# SPDX-License-Identifier: MIT
import io
from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Comfort thresholds (from Alpamayo comfort_reward.py)
COMFORT_MAX_ABS_MAG_JERK = 8.37  # [m/s^3]
COMFORT_MAX_ABS_LAT_ACCEL = 4.89  # [m/s^2]
COMFORT_MAX_LON_ACCEL = 2.40  # [m/s^2]
COMFORT_MIN_LON_ACCEL = -4.05  # [m/s^2]


@dataclass
class GraphConfig:
    label: str
    color: str
    ymin: float
    ymax: float
    thresholds: list[float]


GRAPH_CONFIGS = [
    GraphConfig(label="throttle", color="#00ff00", ymin=0.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="brake", color="#0080ff", ymin=0.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="steer", color="#ffff00", ymin=-1.0, ymax=1.0, thresholds=[]),
    GraphConfig(label="vel", color="#ff8000", ymin=0.0, ymax=80.0, thresholds=[]),
    GraphConfig(
        label="lat G",
        color="#ff00ff",
        ymin=-1.0,
        ymax=1.0,
        thresholds=[
            -COMFORT_MAX_ABS_LAT_ACCEL / 9.81,
            COMFORT_MAX_ABS_LAT_ACCEL / 9.81,
        ],
    ),
    GraphConfig(
        label="lon G",
        color="#00ffff",
        ymin=-1.0,
        ymax=1.0,
        thresholds=[
            COMFORT_MIN_LON_ACCEL / 9.81,
            COMFORT_MAX_LON_ACCEL / 9.81,
        ],
    ),
    GraphConfig(
        label="jerk",
        color="#ff6464",
        ymin=0.0,
        ymax=20.0,
        thresholds=[COMFORT_MAX_ABS_MAG_JERK],
    ),
]


def _extract_series(history, attr: str) -> list[float]:
    return [getattr(record, attr) for record in history]


def render_vehicle_graphs(history, panel_w: int, panel_h: int) -> np.ndarray:
    """Render time-series graphs of vehicle metrics using matplotlib.

    Returns an (panel_h, panel_w, 3) uint8 BGR image.
    """
    num = len(GRAPH_CONFIGS)
    fig, axes = plt.subplots(num, 1, figsize=(panel_w / 100, panel_h / 100), dpi=100)
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.98, bottom=0.02, hspace=0.3)

    series_map = {
        "throttle": _extract_series(history, "throttle"),
        "brake": _extract_series(history, "brake"),
        "steer": _extract_series(history, "steering"),
        "vel": _extract_series(history, "velocity_kph"),
        "lat G": [p.lat_acceleration / 9.81 for p in history],
        "lon G": [p.lon_acceleration / 9.81 for p in history],
        "jerk": _extract_series(history, "jerk"),
    }

    for ax, cfg in zip(axes, GRAPH_CONFIGS):
        data = series_map[cfg.label]
        ax.set_facecolor("black")
        ax.set_ylim(cfg.ymin, cfg.ymax)
        ax.set_ylabel(cfg.label, color=cfg.color, fontsize=6, rotation=0, labelpad=25)
        ax.tick_params(axis="y", labelsize=5, colors="gray")
        ax.tick_params(axis="x", labelbottom=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("gray")
        ax.spines["left"].set_color("gray")

        for thresh in cfg.thresholds:
            ax.axhline(thresh, color="gray", linewidth=0.5, linestyle="--")

        if len(data) >= 2:
            ax.plot(data, color=cfg.color, linewidth=0.8)
            ax.set_xlim(0, len(data) - 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="raw", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    w, h = int(fig.get_size_inches()[0] * fig.dpi), int(fig.get_size_inches()[1] * fig.dpi)
    arr = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    # RGB -> BGR for cv2
    arr = arr[:, :, ::-1].copy()
    return cv2.resize(arr, (panel_w, panel_h))


def overlay_vehicle_graphs(img: np.ndarray, history) -> None:
    """Overlay time-series graphs on the bottom-right of img (mutates img in-place)."""
    if len(history) < 2:
        return
    img_h, img_w = img.shape[:2]
    panel_w = int(img_w * 0.40)
    panel_h = int(img_h * 0.45)
    margin = 4

    graph_img = render_vehicle_graphs(history, panel_w, panel_h)
    x0 = img_w - panel_w - margin
    y0 = img_h - panel_h - margin

    # Alpha blend with background
    roi = img[y0 : y0 + panel_h, x0 : x0 + panel_w]
    blended = cv2.addWeighted(roi, 0.3, graph_img, 0.7, 0)
    img[y0 : y0 + panel_h, x0 : x0 + panel_w] = blended
