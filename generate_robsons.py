#!/usr/bin/env python3

from __future__ import annotations

import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement, product
from traceback import format_exc
from typing import Dict, List, Optional, Sequence, Tuple

from pathlib import Path
import pandas as pd
from ase import Atoms
from ase.constraints import FixedPlane
from ase.db import connect
from ase.optimize import BFGS
from tblite.ase import TBLite

# ──────────────────────────────────────────────────────────────────────
# 0.  Globals & fragment registry
# ──────────────────────────────────────────────────────────────────────
RUN_DIR = Path(os.getenv("ROBSON_LOGDIR", "robson_runs"))
RUN_DIR.mkdir(exist_ok=True)

rob = importlib.import_module("Robsons")
if not (hasattr(rob, "_ACIDS") and hasattr(rob, "_BASES")):
    raise RuntimeError("Robsons.py must expose _ACIDS and _BASES lists.")

ACID_KEYS: List[str] = [k for k in rob._ACIDS if isinstance(k, str)
                        and hasattr(rob, k) and isinstance(getattr(rob, k), Atoms)]
BASE_KEYS: List[str] = [k for k in rob._BASES if isinstance(k, str)
                        and hasattr(rob, k) and isinstance(getattr(rob, k), Atoms)]
if not ACID_KEYS or not BASE_KEYS:
    raise RuntimeError("No valid acid/base fragments found in Robsons._ACIDS/_BASES!")
print("Fragments →  acids:  {}\nbases:   {}".format(
      ", ".join(ACID_KEYS), ", ".join(BASE_KEYS)), flush=True)

# ──────────────────────────────────────────────────────────────────────
# 1.  Spin helpers (robust to dict or DataFrame formats)
# ──────────────────────────────────────────────────────────────────────

def load_spin_data(module: str = "Spin") -> Dict[Tuple[str, int], List[float]]:
    spin = importlib.import_module(module)
    if hasattr(spin, "spin_dict"):
        raw: Dict[Tuple[str, int], Sequence[float]] = spin.spin_dict  # type: ignore[attr-defined]
        df = pd.DataFrame({"Element": [k[0] for k in raw],
                           "Oxidation": [k[1] for k in raw],
                           "Spins": list(raw.values())})
    elif hasattr(spin, "df"):
        df = spin.df  # type: ignore[attr-defined]
    else:
        raise AttributeError("Spin.py must expose either 'spin_dict' or 'df'.")

    ox_col = "Oxidation" if "Oxidation" in df.columns else "Ox"
    if ox_col not in df.columns:
        raise ValueError("Spin DataFrame needs an 'Oxidation' or 'Ox' column")

    def _as_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [float(x)]

    return {(row.Element, int(row[ox_col])): _as_list(row.Spins)
            for _, row in df.iterrows()}

# ──────────────────────────────────────────────────────────────────────
# 2.  Bridges per acid
# ──────────────────────────────────────────────────────────────────────
NO_BRIDGE_ACIDS: set[str] = {"A3"}
BRIDGE_ATOMS = ["N", "O", "S"]
_BRIDGE_CHARGE = {"O": -1, "N": -2, "S": -1}

def bridges_for(acid_key: str):
    return [(None, None)] if acid_key in NO_BRIDGE_ACIDS else \
           list(combinations_with_replacement(BRIDGE_ATOMS, 2))

# ──────────────────────────────────────────────────────────────────────
# 3.  Geometry optimiser helper
# ──────────────────────────────────────────────────────────────────────

def _constrain_xy(self: Atoms):
    self.set_constraint(
        [
            FixedPlane(i, (0, 0, 1))
            for i, a in enumerate(self)
            if i not in (0, 1)
               and a.symbol != "H"
               and -0.01 <= a.position[2] <= 0.01
        ]
    )

Atoms.constrain_xy = _constrain_xy

def run_opt(atoms: Atoms, *, charge: int, mult: int,
            traj: str, log: str, fmax: float = 0.01, steps: int = 50):
    atoms.constrain_xy()
    atoms.calc = TBLite(method="GFN2-xTB", accuracy=1,
                        charge=charge, multiplicity=mult,
                        solvation=("alpb", "water"), verbosity=0)
    opt = BFGS(atoms, trajectory=traj, logfile=log)
    opt.run(fmax=fmax, steps=steps)
    return opt

# ──────────────────────────────────────────────────────────────────────
# 4.  Misc helpers
# ──────────────────────────────────────────────────────────────────────

def total_S_set(S1: float, S2: float) -> set[float]:
    lo, hi = abs(S1 - S2), S1 + S2
    return {round(lo + i, 1) for i in range(int(hi - lo) + 1)}


def name_tag(sym1: str, ch1: int, sym2: str, ch2: int,
             acid: str, base: str, c1: Optional[str], c2: Optional[str]) -> str:
    suffix = "-NNNN" if c1 is None else f"-NNNN{c1}{c2}"
    return f"RMC-{sym1}_{ch1}-{sym2}_{ch2}-{acid}-{base}{suffix}"

# ──────────────────────────────────────────────────────────────────────
# 5.  Worker – returns best structure + convergence flag
# ──────────────────────────────────────────────────────────────────────

def worker(task):
    (m1, m2, bridge, acid_key, base_key, S1_ref, S2_ref) = task
    sym1, ch1 = m1; sym2, ch2 = m2; c1, c2 = bridge

    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")
    logdir = RUN_DIR

    acid_frag = getattr(rob, acid_key)
    base_frag = getattr(rob, base_key)
    tag = name_tag(sym1, ch1, sym2, ch2, acid_key, base_key, c1, c2)

    if c1 is None:
        site = rob.create_site_no_bridge(atom_0=sym1, atom_1=sym2)  # type: ignore[attr-defined]
        charge = ch1 + ch2
    else:
        site = rob.create_site(atom_0=sym1, atom_1=sym2,
                               atom_2=c1, atom_3=c2)  # type: ignore[attr-defined]
        charge = ch1 + ch2 + _BRIDGE_CHARGE[c1] + _BRIDGE_CHARGE[c2]

    mol = rob.build_Robson(site, base_frag.copy(), acid_frag.copy(), x=5.6, y=5)

    # Track absolute best and best converged separately
    best_E = best_mult = None
    best_mol = None

    best_conv_E = best_conv_mult = None
    best_conv_mult = None
    best_conv_mol = None

    for mult in sorted({int(2*S + 1) for S in total_S_set(S1_ref, S2_ref)}):
        trial = mol.copy()
        try:
            opt = run_opt(trial,
                          charge=int(charge), mult=mult,
                          traj=str(logdir / f"{tag}_m{mult}.traj"),
                          log=str(logdir / f"{tag}_m{mult}.log"))
        except Exception:
            continue

        E = trial.get_potential_energy()

        # Track lowest converged
        if opt.converged():
            if best_conv_E is None or E < best_conv_E:
                best_conv_E, best_conv_mult, best_conv_mol = E, mult, trial.copy()

        # Track absolute minimum (for .fail case)
        if best_E is None or E < best_E:
            best_E, best_mult, best_mol = E, mult, trial.copy()

    # Nothing optimised successfully
    if best_mol is None:
        return None

    if best_conv_E is not None:
        return (tag, best_conv_E, best_conv_mult, best_conv_mol,
                True, float(S1_ref), float(S2_ref), int(charge))
    else:
        return (tag, best_E, best_mult, best_mol,
                False, float(S1_ref), float(S2_ref), int(charge))

# ──────────────────────────────────────────────────────────────────────
# 6.  Main driver
# ──────────────────────────────────────────────────────────────────────

def main(db_file: str = "robson.db") -> None:
    spin = load_spin_data()
    metals = list(spin.keys())

    db = connect(db_file)
    print("DB →", os.path.abspath(db_file), flush=True)

    in_db     = {row.name for row in db.select()}
    ok_tags   = {p.stem for p in RUN_DIR.glob("*.ok")}
    fail_tags = {p.stem for p in RUN_DIR.glob("*.fail")}
    skip_set  = in_db | ok_tags | fail_tags

    tasks = []
    for (m1, m2), (acid, base) in product(
            combinations_with_replacement(metals, 2),
            product(ACID_KEYS, BASE_KEYS)):
        for br in bridges_for(acid):
            S1, S2 = map(float, (spin[m1][0], spin[m2][0]))
            tag = name_tag(m1[0], m1[1], m2[0], m2[1], acid, base, br[0], br[1])
            if tag not in skip_set:
                tasks.append((m1, m2, br, acid, base, S1, S2))

    n_workers = int(os.getenv("ROBSON_WORKERS", "1"))
    print(f"Launching {len(tasks)} tasks on {n_workers} worker(s)…", flush=True)

    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed({pool.submit(worker, t): t for t in tasks}):
            try:
                res = fut.result()
            except Exception:
                print("‼ Worker crashed\n" + format_exc(), flush=True)
                continue
            if res is None:
                continue

            tag, E, mult, mol, conv, S1, S2, ch = res
            if conv:
                db.write(mol, name=tag, Energy=E, Multiplicity=mult, Spin1=S1, Spin2=S2, Charge=ch, user="balongn99")

            (RUN_DIR / f"{tag}{'.ok' if conv else '.fail'}").touch()
            done += 1
            print(f"✔ {done}/{len(tasks)} {tag} mult={mult} E={E:.2f} eV "
                  f"({'converged' if conv else 'non-conv'})", flush=True)

    print(f"✓ Finished {done} new structures in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
