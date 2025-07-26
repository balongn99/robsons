#!/usr/bin/env python3

from __future__ import annotations

import argparse, gzip, os
from itertools import combinations_with_replacement, product
from pathlib import Path
from typing import List

import run_one_task as r1         # all helpers live here ─ no other imports


# ────────────────────────────────────────────────────────────────
# sentinels: done  = .ok ∪ .fail
# ────────────────────────────────────────────────────────────────
def sentinel_tags(run_dir: Path) -> set[str]:
    """Set of tags already finished (converged OR failed)."""
    return {p.stem for p in run_dir.glob("*.ok")} | \
           {p.stem for p in run_dir.glob("*.fail")}


# ────────────────────────────────────────────────────────────────
# enumeration
# ────────────────────────────────────────────────────────────────
def main(outfile: Path, run_dir: Path):
    spin   = r1.SPIN                      # loaded once in run_one_task.py
    metals = list(spin.keys())            # [(sym, ox), …]

    done = sentinel_tags(run_dir)

    lines: List[str] = []
    for (m1, m2), (acid, base) in product(
            combinations_with_replacement(metals, 2),
            product(r1.ACID_KEYS, r1.BASE_KEYS)):

        for br1, br2 in r1.bridges_for(acid):
            tag = r1.name_tag(m1[0], m1[1], m2[0], m2[1],
                              acid, base, br1, br2)
            if tag in done:
                continue

            s1, s2 = float(spin[m1][0]), float(spin[m2][0])
            br1s, br2s = ('.' if br1 is None else br1,
                          '.' if br2 is None else br2)
            lines.append(f"{m1[0]}\t{m1[1]}\t{m2[0]}\t{m2[1]}\t"
                         f"{acid}\t{base}\t{br1s}\t{br2s}\t{s1}\t{s2}\n")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(outfile, "wt") as fh:
        fh.writelines(lines)

    print(f"✓  {len(lines):,} new tasks written → {outfile}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outfile", default="tasklist.tsv.gz", type=Path,
                    help="compressed TSV task list")
    ap.add_argument("--run-dir", default=os.getenv("KONGI_LOGDIR", "kongi_runs"),
                    type=Path, help="directory containing *.ok / *.fail files")
    args = ap.parse_args()
    main(args.outfile, args.run_dir)

