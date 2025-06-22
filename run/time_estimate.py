#!/usr/bin/env python3

from __future__ import annotations
import argparse, gzip, os, random, time, yaml
from collections import Counter
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Dict, List

# project scripts must be on PYTHONPATH
import run_one_task as r1

# ----------------------------------------------------------------------
def count_tasks(taskfile: Path) -> int:
    with gzip.open(taskfile, "rt") as fh:
        return sum(1 for _ in fh)

def quick_benchmark(taskfile: Path, omp: int,
                    need_conv: int = 1, need_fail: int = 9) -> float:
    """
    Collect exactly `need_conv` converged and `need_fail`
    non-converged samples.  Skip tasks that return None.
    """
    import run_one_task as r1
    from random import shuffle
    import gzip, time, sys

    with gzip.open(taskfile, "rt") as fh:
        lines = fh.readlines()
    shuffle(lines)

    conv_times, fail_times = [], []
    os.environ["OMP_NUM_THREADS_PER_WORKER"] = str(omp)

    for line in lines:
        # stop when both quotas are satisfied
        if len(conv_times) == need_conv and len(fail_times) == need_fail:
            break

        t0 = time.perf_counter()
        res = r1.run_task_from_line(line)
        dt = time.perf_counter() - t0

        if res is None:
            continue                        # lock clash or fatal crash → ignore

        tag, *_ , conv = res               # we only need `conv`
        if conv and len(conv_times) < need_conv:
            conv_times.append(dt)
        elif (not conv) and len(fail_times) < need_fail:
            fail_times.append(dt)

        # tidy up all scratch files from this probe
        for p in r1.RUN_DIR.glob(f"{tag}*"):
            p.unlink(missing_ok=True)

    if len(conv_times) < need_conv or len(fail_times) < need_fail:
        sys.exit(
            f"‼  Could not gather the required {need_conv}+{need_fail} samples "
            f"(got {len(conv_times)} OK, {len(fail_times)} FAIL).  "
            "Try lowering the quotas or enlarging the task list."
        )

    return (sum(conv_times) + sum(fail_times)) / (need_conv + need_fail)

# ----------------------------------------------------------------------
def main(taskfile: Path, cores: int, omp: int,
         bench: bool, tps_file: Path | None):
    total_tasks = count_tasks(taskfile)

    if bench:
        print("Running benchmark … this may take a few minutes.")
        sec_per_task = quick_benchmark(taskfile, omp)
        print(f"   average seconds per molecule  =  {sec_per_task:.1f} s")
    else:
        if not tps_file:
            raise SystemExit("--no-bench requires --tps-file")
        sec_per_task = yaml.safe_load(tps_file.read_text())["seconds_per_task"]
        print(f"Using seconds_per_task from {tps_file}: {sec_per_task}")

    conc = max(cores // omp, 1)
    eta  = timedelta(seconds = total_tasks * sec_per_task / conc)

    print("\n── Projection ─────────────────────────────────────────────")
    print(f"tasks        : {total_tasks:,}")
    print(f"cores        : {cores}")
    print(f"OMP / worker : {omp}")
    print(f"concurrency  : {conc} optimisations in parallel")
    print(f"ETA (wall)   : {eta}")
    print("Add ~10-20 % safety margin to your Slurm --time.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  default="tasklist.tsv.gz", type=Path)
    ap.add_argument("--cores", type=int, default=32,
                    help="total CPU cores you will request")
    ap.add_argument("--omp",   type=int, default=1,
                    help="OMP_NUM_THREADS_PER_WORKER you will set")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip probe and use --tps-file")
    ap.add_argument("--tps-file", type=Path,
                    help="YAML with seconds_per_task")
    args = ap.parse_args()

    main(args.task, args.cores, args.omp,
         not args.no_bench, args.tps_file)

