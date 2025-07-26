#!/usr/bin/env python3
"""
convergence_check.py  –  classify failed Kongi cages by closest F-max

• Works with BFGS logs whose lines look like
      BFGS:  50 15:17:30   -2241.791524   0.144626
• Thresholds
      near : Fmax < 0.05  eV/Å
      mid  : 0.05 ≤ Fmax < 0.10
      far  : Fmax ≥ 0.10
• Writes conv_summary.tsv  and prints counts.
"""

from pathlib import Path, PurePath
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, sys, argparse, pandas as pd
from tqdm import tqdm
from collections import Counter

RUN_DIR = Path(os.getenv("KONGI_LOGDIR", "kongi_runs"))
CHUNK   = 8192
F_NEAR  = 0.02
F_FAR   = 0.10

# ── CLI ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("-j", "--threads", type=int, default=8,
                help="I/O threads (default 8)")
args = ap.parse_args()

# ── gather fail tags ─────────────────────────────────────────────────────
fail_tags = [p.stem for p in RUN_DIR.rglob("*.fail")]
if not fail_tags:
    sys.exit(f"No *.fail sentinels under {RUN_DIR.resolve()}")

# ── build map tag → list[logs] one time ───────────────────────────────────
log_map = {}
for log in RUN_DIR.rglob("*_m*.log"):
    tag = log.stem.rsplit("_m", 1)[0]
    log_map.setdefault(tag, []).append(log)

def best_mult(tag: str):
    """Return dict(Tag,Mult,Fmax,Steps,Status) for lowest-Fmax multiplicity."""
    best = None  # (mult, fmax, steps)
    for log in log_map.get(tag, []):
        try:
            with log.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - CHUNK))
                tail = fh.read().decode(errors="ignore")
        except OSError:
            continue

        # find last BFGS line
        for ln in reversed(tail.splitlines()):
            if ln.lstrip().startswith("BFGS"):
                parts = ln.split()
                if len(parts) < 5:
                    break
                steps = int(parts[1])
                fmax  = float(parts[-1])
                mult  = int(PurePath(log).stem.split("_m")[-1])
                if best is None or fmax < best[1]:
                    best = (mult, fmax, steps)
                break

    if best:
        mult, fmax, steps = best
        status = ("near" if fmax < F_NEAR else
                  "mid"  if fmax < F_FAR  else "far")
        return dict(Tag=tag, Mult=mult, Fmax=fmax,
                    Steps=steps, Status=status)

    return dict(Tag=tag, Mult=None, Fmax=None,
                Steps=None, Status="missing")

# ── threaded processing ───────────────────────────────────────────────────
rows = []
with ThreadPoolExecutor(max_workers=args.threads) as pool:
    futs = {pool.submit(best_mult, t): t for t in fail_tags}
    for fut in tqdm(as_completed(futs), total=len(futs), unit="mol"):
        rows.append(fut.result())

# ── output ────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv("conv_summary.tsv", sep="\t", index=False)

cnt = Counter(df["Status"])
print("\nNearest-multiplicity summary")
for k in ("near", "mid", "far", "missing"):
    print(f"{k:7s}: {cnt.get(k,0):6d}")
print(f"Total  : {len(df):6d}")
print("→ conv_summary.tsv written")

