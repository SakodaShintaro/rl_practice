"""Plot moving average score across trials.

Reads log_episode_trial00{1,2,3}.tsv from each method directory, computes the mean and
std of recent_average_score across the 3 trials on a common global_step
grid (via linear interpolation), and plots them on a single figure.

Legend is rendered as inline text at the right end of each curve,
instead of a separate legend box.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    return parser.parse_args()


_TAB10 = plt.cm.tab10.colors

# Order here determines the display order (top-to-bottom in the bar chart).
# Colors are fixed per method so that the line plot and bar plot agree
# regardless of ordering.
METHODS = {
    "OFF_POLICY_actor_critic_with_action_value_off_policy_bs16": (
        "Off-policy bs16 (No VLM)",
        _TAB10[0],
    ),
    "OFF_POLICY_vlm_actor_critic_with_action_value_off_policy_bs16": (
        "Off-policy bs16 (VLM)",
        _TAB10[2],
    ),
    "STREAMING_vlm_actor_critic_with_action_value_streaming_": (
        "Streaming (VLM)",
        _TAB10[3],
    ),
    "OFF_POLICY_vlm_actor_critic_with_action_value_off_policy_bs1": (
        "Off-policy bs1 (VLM)",
        _TAB10[1],
    ),
}


def load_trials(method_dir: Path):
    """Return list of (global_step, recent_average_score) arrays for each trial."""
    trials = []
    for trial_file in sorted(method_dir.glob("log_episode_trial*.tsv")):
        df = pd.read_csv(trial_file, sep="\t")
        steps = df["global_step"].to_numpy()
        scores = df["recent_average_score"].to_numpy()
        order = np.argsort(steps)
        trials.append((steps[order], scores[order]))
    return trials


def aggregate(trials, num_points: int = 500):
    """Interpolate each trial onto a common step grid, return (grid, mean, std)."""
    lo = max(t[0].min() for t in trials)
    hi = min(t[0].max() for t in trials)
    grid = np.linspace(lo, hi, num_points)
    interp = np.stack([np.interp(grid, s, v) for s, v in trials], axis=0)
    return grid, interp.mean(axis=0), interp.std(axis=0)


def main():
    args = parse_args()

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7.5))

    label_positions = []  # (x_end, y_end, label, color)

    for dirname, (label, color) in METHODS.items():
        method_dir = args.data_dir / dirname
        if not method_dir.is_dir():
            print(f"Skipping missing method dir: {method_dir}")
            continue
        trials = load_trials(method_dir)
        if len(trials) == 0:
            print(f"No trial files in: {method_dir}")
            continue
        grid, mean, std = aggregate(trials)
        ax.plot(grid, mean, color=color, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        label_positions.append((grid[-1], mean[-1], label, color))

    ax.set_xlabel("Global Step")
    ax.set_ylabel("Moving Average Score")
    ax.set_xticks([0, 50_000, 100_000, 150_000, 200_000])
    ax.set_xticklabels(["0", "50k", "100k", "150k", "200k"])
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inline text labels at the right end of each curve, instead of a legend box.
    # Nudge overlapping labels vertically so they remain readable.
    x_max = max(max(p[0] for p in label_positions), 200_000)
    ax.set_xlim(right=x_max)
    label_positions_sorted = sorted(label_positions, key=lambda p: p[1])
    y_lo, y_hi = ax.get_ylim()
    min_gap = (y_hi - y_lo) * 0.05
    adjusted_ys = []
    prev_y = -np.inf
    for _, y_end, _, _ in label_positions_sorted:
        y = max(y_end, prev_y + min_gap)
        adjusted_ys.append(y)
        prev_y = y
    for (x_end, _, label, color), y in zip(label_positions_sorted, adjusted_ys):
        ax.text(
            x_end,
            y,
            "  " + label,
            color=color,
            va="center",
            ha="left",
            fontsize=20,
            fontweight="bold",
            clip_on=False,
        )

    # Make room on the right side for the inline labels.
    fig.subplots_adjust(right=0.72)

    output = args.data_dir / "car_racing_moving_average.pdf"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {output}")

    plot_sps_bar(args.data_dir)


def plot_sps_bar(data_dir: Path):
    """Bar chart of the final SPS value from trial003 for each method."""
    labels = []
    values = []
    bar_colors = []
    for dirname, (label, color) in METHODS.items():
        trial_file = data_dir / dirname / "log_episode_trial003.tsv"
        if not trial_file.is_file():
            print(f"Skipping missing trial003 file: {trial_file}")
            continue
        df = pd.read_csv(trial_file, sep="\t")
        final_sps = df["SPS"].to_numpy()[-1]
        labels.append(label)
        values.append(final_sps)
        bar_colors.append(color)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    y = np.arange(len(labels))
    ax.barh(y, values, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=20)
    ax.invert_yaxis()  # top-to-bottom matches METHODS order
    ax.set_xlabel("Step per Second (SPS)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:.1f}", ha="left", va="center", fontsize=17)

    output = data_dir / "car_racing_sps.pdf"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {output}")


if __name__ == "__main__":
    main()
