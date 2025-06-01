from crosscoder import CrossCoder
import argparse
from typing import Dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Cross-coder metric")
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--crosscoder_path",
        type=str,
    )
    return parser.parse_args()

def compute_metric(
    crosscoder: CrossCoder,
):
    norms = crosscoder.get_norms()


if __name__ == "__main__":
    pass
