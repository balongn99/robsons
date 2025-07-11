#!/usr/bin/env python3
"""
orphan_check.py  ── Identify Robsons tasks that never wrote a .ok or .fail sentinel.

Usage
=====

    # simplest: writes report to the terminal
    python orphan_check.py

    # custom run directory / task list and save to a file
    python orphan_check.py --run-dir /path/to/robson_runs \
                           --taskfile tasklist.tsv.gz \
                           --json orphan_report.json

Output (JSON)
=============

{
  "run_dir": "/…/robson_runs",
  "taskfile": "/…/tasklist.tsv.gz",
  "count": 42,
  "orphans": [
      "RMC-0001", "RMC-0002", …
  ]
}

Drop it anywhere in the Robsons repo; no external packages required.
"""
import argparse
import csv
import gzip
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect orphan Robsons tasks")
    p.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / "robson_runs",
        help="Directory containing *.ok / *.fail sentinels (default: ./robson_runs)",
    )
    p.add_argument(
        "--taskfile",
        type=pathlib.Path,
        default=pathlib.Path.cwd() / "tasklist.tsv.gz",
        help="Compressed TSV with the master task list (default: ./tasklist.tsv.gz)",
    )
    p.add_argument(
        "--json",
        type=pathlib.Path,
        help="Write JSON report to this file instead of stdout",
    )
    return p.parse_args()


def load_tags(taskfile: pathlib.Path):
    """Yield task tags from the first column of tasklist.tsv.gz."""
    if not taskfile.is_file():
        sys.exit(f"[orphan_check] Task file not found: {taskfile}")
    with gzip.open(taskfile, "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if row:  # skip blank lines
                yield row[0].strip()


def sentinel_exists(tag: str, run_dir: pathlib.Path) -> bool:
    """Return True iff <tag>.ok or <tag>.fail exists in run_dir."""
    return any((run_dir / f"{tag}.{ext}").exists() for ext in ("ok", "fail"))


def main() -> None:
    args = parse_args()
    tags = list(load_tags(args.taskfile))
    orphans = [tag for tag in tags if not sentinel_exists(tag, args.run_dir)]

    report = {
        "run_dir": str(args.run_dir.resolve()),
        "taskfile": str(args.taskfile.resolve()),
        "count": len(orphans),
        "orphans": orphans,
    }

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"[orphan_check] Written report to {args.json}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

