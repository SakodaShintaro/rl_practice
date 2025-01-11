import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir

    tsv_path_list = sorted(target_dir.glob("*/result.tsv"))

    y_key = "return"

    for tsv_path in tsv_path_list:
        df = pd.read_csv(tsv_path, delimiter="\t")

        df["steps"] *= 100
        df["cumulative_steps"] = df["steps"].cumsum()

        plt.plot(df["cumulative_steps"], df[y_key], label=str(tsv_path.parent.name))

    plt.xlabel("step")
    plt.ylabel(y_key)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    save_path = target_dir / "result.png"
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {save_path}")
