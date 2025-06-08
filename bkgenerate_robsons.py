#!/usr/bin/env python3
"""
generate_robsons.py – build and optimise dinuclear Robson frameworks
===================================================================
**New coupling-strategy logic (June 2025)**

For each metal ion we now consider **only two reference spin states**:
lowest-S (LS) and highest-S (HS) in the `Spin.py` list.  For every
metal-metal pair we test two possible couplings:

* **low-spin reference coupling**  (`LS₁‖LS₂`)  → suffix `_low`
* **high-spin reference coupling** (`HS₁‖HS₂`) → suffix `_high`

If LS and HS are identical for *both* metals, only a single configuration
is generated (no suffix).  For each configuration we enumerate total
spin S from |S₁ − S₂| … S₁ + S₂ in unit steps, optimise every multiplicity,
and keep the lowest-energy state.

During optimisation every atom except the two metal centres is constrained
to the XY plane to preserve the square-planar geometry.

Written molecules:
    name  = `MC_<M1><q1><M2><q2>_<c1><c2>[_low|_high]`
    descr = `S1=<S₁> S2=<S₂> multiplicity=<best_mult> charge=<Q>`

Multiprocessing is controlled by `ROBSON_WORKERS` and
`OMP_NUM_THREADS_PER_WORKER`.
"""

from __future__ import annotations
import importlib
import os
import time
from typing import Dict, List, Tuple, Sequence, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from traceback import format_exc
from itertools import combinations_with_replacement   # ← NEW

import pandas as pd
from ase import Atoms
from ase.db import connect
from ase.optimize import BFGS
from tblite.ase import TBLite
from ase.constraints import FixedPlane

# ───────────────────────────────────────────────────────────────
# 0.  Constrain helper
# ───────────────────────────────────────────────────────────────

def constrain_xy(self):
    constraints = []
    for i in range(len(self)):
        if i not in (0, 1):
            constraints.append(FixedPlane(i, (0, 0, 1)))
    self.set_constraint(constraints)

Atoms.constrain_xy = constrain_xy

# ───────────────────────────────────────────────────────────────
# 1.  Spin helpers
# ───────────────────────────────────────────────────────────────

def load_spin_data(module: str = "Spin") -> Dict[Tuple[str, int], List[float]]:
    """
    Import spin data from either `spin_dict` or `df` in Spin.py. Keep only
    2+ cations and Fe³⁺ entries.

    Returns
    -------
    {(element_symbol, oxidation_state): [allowed_spin_values]}
    """
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

    # Keep only 2+ cations or Fe³⁺
    mask = (df.Oxidation == 2) | ((df.Element == "Fe") & (df.Oxidation == 3))
    df = df.loc[mask]

    return {
        (row.Element, int(row.Oxidation)): list(map(float, row.Spins))
        for _, row in df.iterrows()
    }

# ───────────────────────────────────────────────────────────────
# 2.  Utilities
# ───────────────────────────────────────────────────────────────

def total_S_set(S1: float, S2: float) -> set[float]:
    """
    All allowed total-spin quantum numbers S (including half-integers),
    stepping by 1 from |S1 − S2| to S1 + S2 inclusive.
    """
    low, high = abs(S1 - S2), S1 + S2
    return {round(low + i, 1) for i in range(int(high - low) + 1)}

# ───────────────────────────────────────────────────────────────
# 3.  Geometry optimisation helper
# ───────────────────────────────────────────────────────────────

def optimise(
    atoms: Atoms,
    *,
    charge: int,
    multiplicity: int,
    fmax: float = 0.01,
    trajectory: Optional[str] = None,
    logfile: Optional[str] = None
) -> float:
    """
    Run a BFGS geometry optimisation until the largest force < fmax.
    Return the final potential energy in eV.
    """
    atoms.constrain_xy()

    atoms.calc = TBLite(
        method="GFN2-xTB",
        accuracy=1,
        charge=charge,
        multiplicity=multiplicity,
        solvation=("alpb", "water"),
        verbosity=0
    )

    if trajectory or logfile:
        opt = BFGS(atoms, trajectory=trajectory, logfile=logfile)
    else:
        opt = BFGS(atoms, logfile=None)

    opt.run(fmax=fmax)
    return atoms.get_potential_energy()

# ───────────────────────────────────────────────────────────────
# 4.  Worker process
# ───────────────────────────────────────────────────────────────
_CORNER_CHARGE = {          # formal charge each corner contributes
    "O":  -1,
    "N":  -2,
    "S":  -1,
    "P":  -2,               # NEW
    "H": 0,                # NEW – “empty corner”
}

def worker(args) -> Optional[Tuple[str, float, int, Atoms, str]]:
    """
    Build, optimise and return one Robson framework.

    Args tuple = (m1, m2, corner_pair, S1, S2, tag)
      m1/m2          = (element_symbol, oxidation_state)
      corner_pair    = (corner1, corner2)
      S1, S2         = reference spin values
      tag            = '', '_low' or '_high'

    Returns
    -------
      (name, best_energy, best_multiplicity, Atoms object, descr)
      or None if *all* multiplicities failed.
    """
    (m1, m2, corner, S1, S2, tag) = args
    sym1, ch1 = m1
    sym2, ch2 = m2
    c1,  c2   = corner

    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")

    logdir = os.getenv("ROBSON_LOGDIR", "robson_runs")
    os.makedirs(logdir, exist_ok=True)

    base_name = f"MC_{sym1}{ch1}{sym2}{ch2}_{c1}{c2}{tag}"

    # Local import keeps the worker picklable
    rob      = importlib.import_module("Robsons")
    create   = rob.create_site
    assemble = rob.build_Robson
    BASE, SPACER = rob.A1, rob.S1

    site = create(atom_0=sym1, atom_1=sym2,
                  atom_2=c1,    atom_3=c2,
                  corner_atom="N")
    mol  = assemble(site, SPACER.copy(), BASE.copy(), x=6.0, y=5.0)

    multiplicities = sorted({int(2*S + 1) for S in total_S_set(S1, S2)})

    try:
        total_charge = int(ch1 + ch2 + _CORNER_CHARGE[c1] + _CORNER_CHARGE[c2])
    except KeyError as bad:
        raise ValueError(f"Unknown corner atom “{bad.args[0]}”") from None

    best_energy: Optional[float] = None
    best_mult:   Optional[int]   = None
    best_mol:    Optional[Atoms] = None

    for mult in multiplicities:
        trial = mol.copy()
        try:
            traj_file = os.path.join(logdir, f"{base_name}_mult{mult}.traj")
            log_file  = os.path.join(logdir, f"{base_name}_mult{mult}.log")

            E = optimise(
                trial, charge=total_charge, multiplicity=mult,
                fmax=0.01, trajectory=traj_file, logfile=log_file,
            )
        except Exception:
            continue                    # xTB/BFGS crashed → skip

        if best_energy is None or E < best_energy:
            best_energy, best_mult, best_mol = E, mult, trial.copy()

    if best_mult is None:
        return None

    name  = f"MC_{sym1}{ch1}{sym2}{ch2}_{c1}{c2}{tag}"
    descr = f"S1={S1} S2={S2} multiplicity={best_mult} charge={total_charge}"
    return name, best_energy, best_mult, best_mol, descr

# ───────────────────────────────────────────────────────────────
# 5.  Main driver
# ───────────────────────────────────────────────────────────────

def main(db_path: str = "balongtest.db") -> None:
    spin_dict = load_spin_data()
    metals    = list(spin_dict.keys())

    # Corner-atom combinations  (unordered, with repetition)
    CORNER_ATOMS = ["N", "O", "S", "P", "H"]          # NEW members included
    CORNERS      = list(combinations_with_replacement(CORNER_ATOMS, 2))

    # Build tasks: (m1, m2, corner, S1, S2, tag)
    tasks: List[Tuple] = []
    for m1, m2 in combinations_with_replacement(metals, 2):   # NEW – symmetry handled
        s1_low, s1_high = min(spin_dict[m1]), max(spin_dict[m1])
        s2_low, s2_high = min(spin_dict[m2]), max(spin_dict[m2])

        for corner in CORNERS:
            tag_low = "_low" if (s1_low != s1_high or s2_low != s2_high) else ""
            tasks.append((m1, m2, corner, s1_low, s2_low, tag_low))

            if (s1_low != s1_high) or (s2_low != s2_high):
                tasks.append((m1, m2, corner, s1_high, s2_high, "_high"))

    max_workers = int(os.getenv("ROBSON_WORKERS", "1"))
    print(f"Running with {max_workers} worker process(es)…", flush=True)

    db = connect(db_path)
    t0 = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                print("‼ Worker crashed for", futures[fut][:3], format_exc(), flush=True)
                continue

            if result is None:
                continue                        # optimisation failed

            name, E, mult, mol, descr = result
            if list(db.select(name=name)):
                print(f"↺ {name} already in DB – skipped", flush=True)
                continue

            db.write(mol, name=name, description=descr, Energy=E)

            done += 1
            minutes = (time.time() - t0) / 60.0
            print(f"✔ {done}/{len(tasks)}  {name}  mult={mult}  "
                  f"E={E:.2f} eV  ({minutes:.1f} min)", flush=True)

if __name__ == "__main__":
    main()

