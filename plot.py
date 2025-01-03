import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir_list", type=Path, nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    target_dir_list = args.target_dir_list

    for target_dir in target_dir_list:
        tsv_path = target_dir / "result.tsv"
        df = pd.read_csv(tsv_path, delimiter="\t")

        sum_steps = df["steps"].cumsum()

        plt.plot(sum_steps, df["steps"], label=str(target_dir.name))

    plt.xlabel("Episode")
    plt.ylabel("steps")
    plt.legend()
    plt.grid()
    plt.savefig("result.png", bbox_inches="tight", pad_inches=0.05)
    print("Saved result.png")
