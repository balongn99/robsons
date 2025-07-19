#!/usr/bin/env python3


from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from typing import Dict, Any

import ase.io
from ase import Atoms
from ase.db import connect

# ───────────────────────────────────────────────────────────────
def _load_meta(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)

def _traj_path(run_dir: Path, tag: str, multiplicity: int) -> Path | None:
    """Return the trajectory that exists for *tag*, else None."""
    p = run_dir / f"{tag}.traj"
    if p.is_file():
        return p
    p = run_dir / f"{tag}_m{multiplicity}.traj"
    return p if p.is_file() else None

def _finite(x):
    """Convert x → float if finite, else None (so JSON NaN/Inf→SQL NULL)."""
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────
def main(run_dir: Path, db_path: Path):
    result_dir = run_dir / "results"
    if not result_dir.is_dir():
        print(f"‼  No results directory: {result_dir}", file=sys.stderr)
        return

    db = connect(db_path)
    existing = {row.name for row in db.select()}
    added = 0

    for jf in sorted(result_dir.glob("*.json")):
        meta = _load_meta(jf)
        tag  = meta.get("tag") or jf.stem
        if tag in existing:
            continue

        traj_file = _traj_path(run_dir, tag, meta.get("multiplicity", 0))
        if traj_file is None:
            print(f"⚠  No trajectory found for {tag} – skipped.", file=sys.stderr)
            continue

        # read the last frame = relaxed geometry
        try:
            atoms: Atoms = ase.io.read(traj_file, -1)
        except Exception as err:
            print(f"⚠  Cannot read {traj_file.name}: {err}", file=sys.stderr)
            continue

        # put Mulliken charges back on the Atoms object if present
        if "mulliken_q_e" in meta and meta["mulliken_q_e"]:
            atoms.set_initial_charges(meta["mulliken_q_e"])

        # dipole vector may be missing or null
        dvec = meta.get("dipole_vec_eA", [None, None, None]) or [None, None, None]
        dx, dy, dz = (_finite(dvec[i]) for i in range(3))

        db.write(
            atoms,
            # ── mandatory identifiers ─────────────────────────────
            name=tag,
            Energy=_finite(meta.get("energy_eV")),
            Multiplicity=int(meta.get("multiplicity", 0)),
            Charge=int(meta.get("charge_total", meta.get("charge", 0))),
            # ── legacy ID fields (keep if present) ───────────────
            Metal1=meta.get("sym1"),
            Oxidation1=meta.get("ox1"),
            Spin1=meta.get("S1_ref"),
            Metal2=meta.get("sym2"),
            Oxidation2=meta.get("ox2"),
            Spin2=meta.get("S2_ref"),
            Acid=meta.get("acid"),
            Base=meta.get("base"),
            Bridge=".".join((meta.get("bridge") or ["", ""])),
            # ── new EDL-relevant columns ─────────────────────────
            DipoleX_eA=dx,
            DipoleY_eA=dy,
            DipoleZ_eA=dz,
            Dipole_D=_finite(meta.get("dipole_mag_D")),
            MullikenJSON=json.dumps(meta.get("mulliken_q_e", [])),
            HOMO_eV=_finite(meta.get("HOMO_eV")) or "-",
            LUMO_eV=_finite(meta.get("LUMO_eV")) or "-",
            Gap_eV=_finite(meta.get("gap_eV")) or "-",
            # user tag for provenance
            username="balongn99",
        )
        added += 1
        existing.add(tag)

    print(f"✓  Added {added} new row(s) to {db_path}")

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge JSON results into ASE DB")
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
    main(args.run_dir.resolve(), args.db.resolve())

