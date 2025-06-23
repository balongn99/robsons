#!/usr/bin/env python3
"""
Predict wall-time for a Robson campaign.

Changes vs. v1
--------------
1. Counts *calculations*  (= optimisations per multiplicity) from spins.
2. Benchmarks seconds per calculation, separately for converged and
   non-converged probes.
3. Forecast assumes 10 % converged  /  90 % non-converged.
"""

from __future__ import annotations
import argparse, gzip, os, random, time, sys
from pathlib import Path
from statistics import median
from collections import Counter, defaultdict

# ---------- helper to count multiplicities -------------------------------
def mult_count(s1: float, s2: float) -> int:
    lo, hi = abs(s1 - s2), s1 + s2
    return int(round(hi - lo)) + 1            # inclusive

# ---------- read task list ------------------------------------------------
def read_tasklist(taskfile: Path):
    tasks = []
    with gzip.open(taskfile, "rt") as fh:
        for line in fh:
            f = line.split()
            s1, s2 = map(float, f[8:10])
            tasks.append(dict(
                line=line.strip(),
                mults=mult_count(s1, s2)
            ))
    return tasks

# ---------- micro-benchmark ----------------------------------------------
def bench_seconds_per_calc(sample_lines: list[str],
                           need_conv=3, need_fail=3,
                           omp=1) -> tuple[float, float]:
    """return (median_sec_per_calc_conv, median_sec_per_calc_fail)"""
    import run_one_task as r1
    random.shuffle(sample_lines)

    per_calc_conv, per_calc_fail = [], []
    os.environ["OMP_NUM_THREADS_PER_WORKER"] = str(omp)

    for line in sample_lines:
        if len(per_calc_conv) >= need_conv and len(per_calc_fail) >= need_fail:
            break
        t0 = time.perf_counter()
        res = r1.run_task_from_line(line)
        dt = time.perf_counter() - t0
        # clean artefacts
        if res is None:
            continue
        tag, *_ , conv = res
        for p in r1.RUN_DIR.glob(f"{tag}*"):
            p.unlink(missing_ok=True)

        s1, s2 = map(float, line.split()[8:10])
        ncalc  = mult_count(s1, s2)
        per_calc = dt / ncalc
        if conv and len(per_calc_conv) < need_conv:
            per_calc_conv.append(per_calc)
        elif (not conv) and len(per_calc_fail) < need_fail:
            per_calc_fail.append(per_calc)

    if not per_calc_conv or not per_calc_fail:
        sys.exit("Could not collect both conv & fail benchmarks.")
    return median(per_calc_conv), median(per_calc_fail)

# ---------- main ----------------------------------------------------------
def main(taskfile: Path, cores: int, omp: int, sample: int):
    tasks = read_tasklist(taskfile)
    total_tasks = len(tasks)
    calc_total  = sum(t["mults"] for t in tasks)

    # --- benchmark on a subset -----------------------------------------
    sample_lines = random.sample([t["line"] for t in tasks],
                                 min(sample, total_tasks))
    sec_conv, sec_fail = bench_seconds_per_calc(sample_lines,
                                                omp=omp)

    # --- forecast (10 % ok, 90 % fail) ----------------------------------
    calc_conv = int(0.10 * calc_total)
    calc_fail = calc_total - calc_conv

    conc = max(cores // omp, 1)
    eta_sec = (calc_conv * sec_conv + calc_fail * sec_fail) / conc

    from datetime import timedelta
    print("──────── forecast ─────────────────────────────────────────")
    print(f"tasks               : {total_tasks:,}")
    print(f"calculations total  : {calc_total:,}")
    print(f"  – expected OK     : {calc_conv:,}")
    print(f"  – expected FAIL   : {calc_fail:,}")
    print(f"sec / calc (OK)     : {sec_conv:8.2f}")
    print(f"sec / calc (FAIL)   : {sec_fail:8.2f}")
    print(f"concurrency         : {conc} (cores={cores}, OMP={omp})")
    print(f"ETA (wall)          : {timedelta(seconds=int(eta_sec))}")

# ---------- CLI -----------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="tasklist.tsv.gz", type=Path)
    ap.add_argument("--cores", type=int, default=32,
                    help="total CPU cores you will request")
    ap.add_argument("--omp", type=int, default=1,
                    help="OMP_NUM_THREADS_PER_WORKER")
    ap.add_argument("--sample", type=int, default=20,
                    help="# tasks to probe for benchmarking")
    args = ap.parse_args()
    main(args.task, args.cores, args.omp, args.sample)

