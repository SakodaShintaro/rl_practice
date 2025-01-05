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

    y_key = "return"

    for target_dir in target_dir_list:
        tsv_path = target_dir / "result.tsv"
        df = pd.read_csv(tsv_path, delimiter="\t")

        y_value = df[y_key].cumsum()

        plt.plot(y_value, df[y_key], label=str(target_dir.name))

    plt.xlabel("Episode")
    plt.ylabel(y_key)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid()
    plt.savefig("result.png", bbox_inches="tight", pad_inches=0.05)
    print("Saved result.png")
