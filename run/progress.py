#!/usr/bin/env python3
"""
Real-time Robson run progress.

Counts
-------
TOTAL      – lines in tasklist.tsv.gz        (# molecules requested)
OK         – *.ok  sentinels                 (# converged)
FAIL       – *.fail sentinels                (# all multiplicities failed)
RUNNING    – *.lock files                    (# currently in flight)
PENDING    = TOTAL − (OK + FAIL + RUNNING)

Usage
-----
$ python progress.py                         # default paths
$ python progress.py --run-dir /scratch/robson_runs --task tasklist.tsv.gz
$ watch -n 60 python progress.py             # live dashboard
"""

from __future__ import annotations
import argparse, gzip
from pathlib import Path
from time import strftime

def count_lines(fname: Path) -> int:
    with gzip.open(fname, "rt") as fh:
        return sum(1 for _ in fh)

def main(run_dir: Path, taskfile: Path):
    total = count_lines(taskfile) if taskfile.is_file() else 0
    ok    = len(list(run_dir.glob("*.ok")))
    fail  = len(list(run_dir.glob("*.fail")))
    run   = len(list(run_dir.glob("*.lock")))
    pend  = max(total - ok - fail - run, 0)

    print(f"[{strftime('%Y-%m-%d %H:%M:%S')}]  "
          f"TOTAL {total:,}  |  RUNNING {run:,}  |  OK {ok:,}  |  "
          f"FAIL {fail:,}  |  PENDING {pend:,}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="robson_runs", type=Path,
                    help="directory containing *.ok / *.fail / *.lock")
    ap.add_argument("--task", default="tasklist.tsv.gz", type=Path,
                    help="compressed task list with all requested jobs")
    args = ap.parse_args()
    main(args.run_dir, args.task)

