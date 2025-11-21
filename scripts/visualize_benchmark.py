# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import argparse
import sys
from pathlib import Path

from visualizer import visualize_benchmark_runs


def _build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for the visualize script."""
    parser = argparse.ArgumentParser(
        prog="python -m judo.benchmark.visualize", description="Visualize benchmark runs from an HDF5 file."
    )
    parser.add_argument("h5_path", metavar="H5_PATH", help="Path to the benchmark results .h5 file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the visualize script."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    path = Path(args.h5_path)
    if not path.exists():
        parser.error(f"File not found: {path}")

    try:
        visualize_benchmark_runs(path)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
