#!/usr/bin/env python3

from __future__ import annotations
import argparse, gzip, itertools, os
from multiprocessing import Pool, cpu_count
import run_one_task

def slice_file(fname: str, start: int, end: int):
    with gzip.open(fname, "rt") as fh:
        yield from itertools.islice(fh, start, end)

def main(taskfile: str, start: int, end: int, omp_per_worker: int):
    os.environ["OMP_NUM_THREADS_PER_WORKER"] = str(omp_per_worker)
    os.environ["OMP_NUM_THREADS"]            = "1"        # for xTB
    n_workers = max(1, cpu_count() // omp_per_worker)

    with Pool(n_workers) as pool:
        pool.map(run_one_task.run_task_from_line,
                 slice_file(taskfile, start, end))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("taskfile")
    ap.add_argument("start", type=int)
    ap.add_argument("end",   type=int)
    ap.add_argument("--omp", type=int, default=int(os.getenv("OMP_PER_WORKER", "1")))
    args = ap.parse_args()
    main(args.taskfile, args.start, args.end, args.omp)

