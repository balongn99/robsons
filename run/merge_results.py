#!/usr/bin/env python3
"""
Merge converged results/<tag>.json → ASE database (robson.db)

For every JSON file the script:

1. Finds the relaxed geometry:
      <RUN_DIR>/<tag>.traj            (preferred, if present)
   or <RUN_DIR>/<tag>_m<M>.traj       (fallback, M = multiplicity in JSON)

2. Writes a row to the database that includes the Atoms object.

Rows whose `name` already exists in the DB are skipped.
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Any

from ase import Atoms
from ase.db import connect
from ase.io import read

# ───────────────────────────────────────────────────────────────
def load_meta(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)

def find_traj(run_dir: Path, tag: str, multiplicity: int) -> Path | None:
    """Return the path of a trajectory file that exists, else None."""
    preferred = run_dir / f"{tag}.traj"
    if preferred.is_file():
        return preferred

    fallback = run_dir / f"{tag}_m{multiplicity}.traj"
    return fallback if fallback.is_file() else None

# ───────────────────────────────────────────────────────────────
def main(run_dir: Path, db_path: Path):
    result_dir = run_dir / "results"
    if not result_dir.is_dir():
        print(f"‼  No results directory: {result_dir}")
        return

    db = connect(db_path)
    existing = {row.name for row in db.select()}
    added = 0

    for jf in sorted(result_dir.glob("*.json")):
        meta = load_meta(jf)
        tag  = meta["tag"]
        if tag in existing:
            continue

        traj_path = find_traj(run_dir, tag, meta["multiplicity"])
        if traj_path is None:
            print(f"⚠  No trajectory found for {tag} – skipped.")
            continue

        atoms: Atoms = read(traj_path, -1)   # last frame = relaxed
        db.write(
            atoms,
            name=tag,
            Energy=meta["energy_eV"],
            Multiplicity=meta["multiplicity"],
            Charge=meta["charge"],
            Spin1=meta["S1_ref"],
            Spin2=meta["S2_ref"],
            Acid=meta["acid"],
            Base=meta["base"],
            Bridge=".".join(x or "" for x in meta["bridge"]),
            username="balongn99",
        )
        added += 1
        existing.add(tag)
    print(f"✓  Added {added} new rows with geometries to {db_path}")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default=os.getenv("ROBSON_LOGDIR", "robson_runs"),
        type=Path,
        help="root directory with *.traj and results/*.json",
    )
    ap.add_argument(
        "--db",
        default="robson.db",
        type=Path,
        help="ASE SQLite database to update / create",
    )
    args = ap.parse_args()
    main(args.run_dir, args.db)

