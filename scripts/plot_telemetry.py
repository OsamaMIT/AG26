#!/usr/bin/env python3
"""Plot telemetry CSV outputs."""

from __future__ import annotations

import argparse

from _bootstrap import bootstrap

bootstrap()

from gohan.analysis.plots import plot_telemetry_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    outputs = plot_telemetry_csv(args.input, args.output)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
