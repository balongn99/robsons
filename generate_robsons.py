#!/usr/bin/env python3

from __future__ import annotations
import importlib
import os
import time
from typing import Dict, List, Tuple, Sequence, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from traceback import format_exc
from itertools import combinations_with_replacement, product

import pandas as pd
from ase import Atoms
from ase.db import connect
from ase.optimize import BFGS
from tblite.ase import TBLite
from ase.constraints import FixedPlane

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 0.  Fragment library (auto‑discovered)
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

rob = importlib.import_module("Robsons")

# Helper: return every attribute whose name starts with the given prefix
# and whose value is an ASE Atoms object.  Order alphanumerically so that
# F1 < F2 < …, S1 < S2 < …, making results reproducible.

def _fragment_keys(prefix: str) -> List[str]:
    keys = [k for k, v in vars(rob).items() if k.startswith(prefix) and isinstance(v, Atoms)]
    return sorted(keys, key=lambda s: (len(s), s))  # S1 < S10 < S2 without tricks

FRAME_KEYS   = _fragment_keys("F")  # e.g. ["F1", "F2", …]
SPACER_KEYS = _fragment_keys("S")  # e.g. ["S1", "S2", …]

if not FRAME_KEYS:
    raise RuntimeError("No frame fragments (variables \"F*\") found in Robsons.py!")
if not SPACER_KEYS:
    raise RuntimeError("No spacer fragments (variables \"S*\") found in Robsons.py!")

print(f"Discovered frames:   {', '.join(FRAME_KEYS)}")
print(f"Discovered spacers: {', '.join(SPACER_KEYS)}")

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 1.  Constrain helper
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

def constrain_xy(self):
    constraints = []
    for i in range(len(self)):
        if i not in (0, 1):
            constraints.append(FixedPlane(i, (0, 0, 1)))
    self.set_constraint(constraints)

Atoms.constrain_xy = constrain_xy

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 2.  Spin helpers
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

def load_spin_data(module: str = "Spin") -> Dict[Tuple[str, int], List[float]]:
    """Import spin data from either `spin_dict` or `df` in Spin.py."""

    spin = importlib.import_module(module)
    if hasattr(spin, "spin_dict"):
        raw: Dict[Tuple[str, int], Sequence[float]] = spin.spin_dict  # type: ignore
        df = pd.DataFrame({
            "Element": [k[0] for k in raw],
            "Oxidation": [k[1] for k in raw],
            "Spins": list(raw.values())
        })
    elif hasattr(spin, "df"):
        df = spin.df  # type: ignore
    else:
        raise AttributeError("Spin.py must define either `spin_dict` or `df`")

    return {
        (row.Element, int(row.Oxidation)): list(map(float, row.Spins))
        for _, row in df.iterrows()
    }

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 3.  Utilities
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

def total_S_set(S1: float, S2: float) -> set[float]:
    """All allowed total‑spin quantum numbers S (including half‑integers)."""
    low, high = abs(S1 - S2), S1 + S2
    return {round(low + i, 1) for i in range(int(high - low) + 1)}

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 4.  Geometry optimisation helper
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

def optimise(
    atoms: Atoms,
    *,
    charge: int,
    multiplicity: int,
    fmax: float = 0.01,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None,
) -> float:
    """Run a BFGS geometry optimisation until the largest force < fmax."""

    atoms.constrain_xy()

    atoms.calc = TBLite(
        method="GFN2-xTB",
        accuracy=1,
        charge=charge,
        multiplicity=multiplicity,
        solvation=("alpb", "water"),
        verbosity=0,
    )

    if trajectory or logfile:
        opt = BFGS(atoms, trajectory=trajectory, logfile=logfile)
    else:
        opt = BFGS(atoms, logfile=None)

    opt.run(fmax=fmax, steps=200)
    return atoms.get_potential_energy()

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 5.  Worker process
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

_CORNER_CHARGE = {
    "O": -1,
    "N": -2,
    "S": -1,
    "P": -2,
    "H": 0,
}

def worker(args) -> Optional[Tuple[str, float, int, Atoms, str]]:
    """Build, optimise and return one Robson framework."""

    (
        m1,
        m2,
        corner,
        frame_key,
        spacer_key,
        S1_ref,
        S2_ref,
        tag,
    ) = args

    sym1, ch1 = m1
    sym2, ch2 = m2
    c1, c2 = corner

    # Limit OpenMP threads inside each worker
    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")

    logdir = os.getenv("ROBSON_LOGDIR", "robson_runs")
    os.makedirs(logdir, exist_ok=True)

    frame_fragment = getattr(rob, frame_key)
    spacer_fragment = getattr(rob, spacer_key)

    # File‑name stem shared by trajectory, log and DB entry
    base_name = f"RMC_{sym1}{ch1}{sym2}{ch2}_{frame_key}{spacer_key}_{c1}{c2}{tag}"

    # ─── Build initial structure ────────────────────────────────
    create   = rob.create_site
    assemble = rob.build_Robson

    site = create(atom_0=sym1, atom_1=sym2, atom_2=c1, atom_3=c2, corner_atom="N")
    mol  = assemble(site, spacer_fragment.copy(), frame_fragment.copy(), x=6.0, y=5.0)

    multiplicities = sorted({int(2 * S + 1) for S in total_S_set(S1_ref, S2_ref)})

    try:
        total_charge = int(ch1 + ch2 + _CORNER_CHARGE[c1] + _CORNER_CHARGE[c2])
    except KeyError as bad:
        raise ValueError(f"Unknown corner atom “{bad.args[0]}”") from None

    best_energy: Optional[float] = None
    best_mult:   Optional[int] = None
    best_mol:    Optional[Atoms] = None

    for mult in multiplicities:
        trial = mol.copy()
        try:
            traj_file = os.path.join(logdir, f"{base_name}_mult{mult}.traj")
            log_file  = os.path.join(logdir, f"{base_name}_mult{mult}.log")

            E = optimise(
                trial,
                charge=total_charge,
                multiplicity=mult,
                fmax=0.01,
                trajectory=traj_file,
                logfile=log_file,
            )
        except Exception:
            continue

        if best_energy is None or E < best_energy:
            best_energy, best_mult, best_mol = E, mult, trial.copy()

    if best_mult is None:
        return None

    name  = base_name  # Already includes tag
    descr = (
        f"frame={frame_key} spacer={spacer_key} "
        f"S1={S1_ref} S2={S2_ref} multiplicity={best_mult} charge={total_charge}"
    )
    return name, best_energy, best_mult, best_mol, descr

# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
# 6.  Main driver
# ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑

def main(db_path: str = "balongtest.db") -> None:
    spin_dict = load_spin_data()
    metals    = list(spin_dict.keys())

    CORNER_ATOMS = ["N", "O", "S", "P", "H"]
    CORNERS      = list(combinations_with_replacement(CORNER_ATOMS, 2))

    # ── 1) CONNECT + CACHE EXISTING NAMES ───────────────────────
    db = connect(db_path)

    # (optional WAL + async writes)
    conn = db._connect()
    existing = {row.name for row in db.select()}
    # ────────────────────────────────────────────────────────────

    # ── 2) BUILD raw task list (unfiltered) ─────────────────────
    raw_tasks: List[Tuple] = []
    for (m1, m2), corner, (frame_key, spacer_key) in product(
        combinations_with_replacement(metals, 2),
        CORNERS,
        product(FRAME_KEYS, SPACER_KEYS),
    ):
        s1_low, s1_high = min(spin_dict[m1]), max(spin_dict[m1])
        s2_low, s2_high = min(spin_dict[m2]), max(spin_dict[m2])

        tag_low = "_low" if (s1_low != s1_high or s2_low != s2_high) else ""
        raw_tasks.append((m1, m2, corner, frame_key, spacer_key, s1_low, s2_low, tag_low))

        if (s1_low != s1_high) or (s2_low != s2_high):
            raw_tasks.append((m1, m2, corner, frame_key, spacer_key, s1_high, s2_high, "_high"))
    # ────────────────────────────────────────────────────────────

    # ── 3) FILTER tasks by “name” before submitting ──────────────
    def make_name(args: Tuple) -> str:
        (m1, m2, corner, frame_key, spacer_key, S1_ref, S2_ref, tag) = args
        sym1, ch1 = m1
        sym2, ch2 = m2
        c1, c2 = corner
        return f"RMC_{sym1}{ch1}{sym2}{ch2}_{frame_key}{spacer_key}_{c1}{c2}{tag}"

    tasks = []
    for t in raw_tasks:
        name = make_name(t)
        if name not in existing:
            tasks.append(t)
    # Now “tasks” contains only those molecules NOT yet in the DB
    # ────────────────────────────────────────────────────────────

    max_workers = int(os.getenv("ROBSON_WORKERS", "1"))
    print(f"Running with {max_workers} worker process(es)…", flush=True)

    t0   = time.time()
    done = 0

    # ── 4) SUBMIT ONLY filtered tasks ────────────────────────────
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                print("‼ Worker crashed for", futures[fut][:6], format_exc(), flush=True)
                continue

            if result is None:
                continue

            name, E, mult, mol, descr = result

            # (You can safely skip “if name in existing” here,
            #  because we already filtered above—but to be safe:)
            if name in existing:
                continue

            db.write(mol, name=name, description=descr, Energy=E)
            existing.add(name)   # keep cache up-to-date

            done += 1
            minutes = (time.time() - t0) / 60.0
            print(f"✔ {done}/{len(tasks)}  {name}  mult={mult}  "
                  f"E={E:.2f} eV  ({minutes:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

