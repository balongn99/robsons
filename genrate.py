#!/usr/bin/env python3
"""
Generate Robsons cages – **consistent with `generate_robsons.py`** naming
while honouring the user‑specified workflow:

1.  Remove the two metal atoms and optimise the organic frame/spacer *in the
   z = 0 plane* using charge = total_charge − metal_charge.
2.  Copy the relaxed frame coordinates back into the full molecule.
3.  Freeze the frame/spacer atoms (they stay in‑plane by construction), leave
   the metals completely unconstrained, and run the final optimisation at the
   full system charge/multiplicity.
4.  Store the resulting structure and metadata in the database.

Key choices
-----------
* Only Stage 1 uses a z‑plane (`FixedPlane`) constraint; Stage 2 does **not** –
  metals are free to move in 3‑D, frame atoms are frozen via `FixAtoms`.
* Unused return value of Stage 1 optimisation is now discarded with `_`.
"""
from __future__ import annotations

import importlib
import os
import time
from itertools import combinations_with_replacement, product
from typing import Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from traceback import format_exc

import pandas as pd
from ase import Atoms
from ase.db import connect
from ase.optimize import BFGS
from tblite.ase import TBLite
from ase.constraints import FixedPlane, FixAtoms

# ──────────────────────────────────────────────────────────────────────
# 0.  Fragment library (auto‑discovered)
# ──────────────────────────────────────────────────────────────────────
rob = importlib.import_module("Robsons")


def _fragment_keys(prefix: str) -> List[str]:
    keys = [k for k, v in vars(rob).items() if k.startswith(prefix) and isinstance(v, Atoms)]
    return sorted(keys, key=lambda s: (len(s), s))


FRAME_KEYS = _fragment_keys("A")  # frames
SPACER_KEYS = _fragment_keys("B")  # spacers

if not FRAME_KEYS or not SPACER_KEYS:
    raise RuntimeError("Missing A* or B* fragments in Robsons.py")

print("Frames:", ", ".join(FRAME_KEYS))
print("Spacers:", ", ".join(SPACER_KEYS))

# ──────────────────────────────────────────────────────────────────────
# 1.  Spin data helper (unchanged)
# ──────────────────────────────────────────────────────────────────────

def load_spin_data(module: str = "Spin") -> Dict[Tuple[str, int], List[float]]:
    spin = importlib.import_module(module)
    if hasattr(spin, "spin_dict"):
        raw: Dict[Tuple[str, int], Sequence[float]] = spin.spin_dict  # type: ignore
        df = pd.DataFrame({"Element": [k[0] for k in raw], "Oxidation": [k[1] for k in raw], "Spins": list(raw.values())})
    elif hasattr(spin, "df"):
        df = spin.df.copy()
    else:
        raise AttributeError("Spin.py must define spin_dict or df")
    return {(r.Element, int(r.Oxidation)): list(map(float, r.Spins)) for _, r in df.iterrows()}


def total_S_set(S1: float, S2: float) -> set[float]:
    low, high = abs(S1 - S2), S1 + S2
    return {round(low + i, 1) for i in range(int(high - low) + 1)}

# ──────────────────────────────────────────────────────────────────────
# 2.  Optimisation helpers
# ──────────────────────────────────────────────────────────────────────

def _single_opt(
    atoms: Atoms,
    *,
    charge: int,
    multiplicity: int,
    fmax: float,
    plane_constraint: bool,
    traj: Optional[str] = None,
    log: Optional[str] = None,
) -> float:
    """Run one optimisation; optionally keep all atoms in the z = 0 plane."""
    if plane_constraint:
        atoms.set_constraint([FixedPlane(i, (0, 0, 1)) for i in range(len(atoms))])
    atoms.calc = TBLite(method="GFN2-xTB", accuracy=1, charge=charge, multiplicity=multiplicity, solvation=("alpb", "water"), verbosity=0)
    opt = BFGS(atoms, trajectory=traj, logfile=log)
    opt.run(fmax=fmax, steps=200)
    return atoms.get_potential_energy()


def optimise(
    atoms: Atoms,
    *,
    total_charge: int,
    metal_charge: int,
    multiplicity: int,
    fmax: float = 0.01,
    traj_base: Optional[str] = None,
    log_base: Optional[str] = None,
) -> float:
    """Two‑stage optimisation per workflow description."""
    # ── Stage 1 – organic fragment only, constrained to plane ────────
    frame_only = atoms.copy()
    del frame_only[[1, 0]]  # remove metals (index 1 first!)

    frame_E = _single_opt(
        frame_only,
        charge=total_charge - metal_charge,
        multiplicity=1,
        fmax=fmax / 2,
        plane_constraint=True,
    )
    # copy relaxed coordinates back
    atoms.positions[2:] = frame_only.positions

    # ── Stage 2 – freeze frame, metals free ──────────────────────────
    atoms.set_constraint([FixAtoms(indices=range(2, len(atoms)))])

    E_tot = _single_opt(
        atoms,
        charge=total_charge,
        multiplicity=multiplicity,
        fmax=fmax,
        plane_constraint=False,
        traj=traj_base,
        log=log_base,
    )
    return E_tot

# ──────────────────────────────────────────────────────────────────────
# 3.  Naming helper (reference‑compatible)
# ──────────────────────────────────────────────────────────────────────

def _build_name(sym1, ch1, sym2, ch2, frame_key, spacer_key, c1, c2, tag):
    tag_part = f"-{tag.lstrip('_')}" if tag else ""
    return f"RMC-{sym1}_{ch1}-{sym2}_{ch2}-{frame_key}-{spacer_key}-NNNN{c1}{c2}{tag_part}"

_CORNER_CHARGE = {"O": -1, "N": -2, "S": -1, "P": -2, "H": 0}

# ──────────────────────────────────────────────────────────────────────
# 4.  Worker
# ──────────────────────────────────────────────────────────────────────

def worker(args):
    (m1, m2, corner, frame_key, spacer_key, S1, S2, tag) = args
    sym1, ch1 = m1
    sym2, ch2 = m2
    c1, c2 = corner

    os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS_PER_WORKER", "1"))

    base_name = _build_name(sym1, ch1, sym2, ch2, frame_key, spacer_key, c1, c2, tag)
    logdir = os.getenv("ROBSON_LOGDIR", "robson_runs"); os.makedirs(logdir, exist_ok=True)

    # Build initial structure
    site = rob.create_site(atom_0=sym1, atom_1=sym2, atom_2=c1, atom_3=c2, corner_atom="N")
    mol = rob.build_Robson(site, getattr(rob, spacer_key).copy(), getattr(rob, frame_key).copy(), x=6.0, y=5.0)

    metal_charge = ch1 + ch2
    total_charge = metal_charge + _CORNER_CHARGE[c1] + _CORNER_CHARGE[c2]
    best_E = best_mult = best_mol = None

    for mult in sorted({int(2 * S + 1) for S in total_S_set(S1, S2)}):
        trial = mol.copy()
        try:
            E = optimise(
                trial,
                total_charge=total_charge,
                metal_charge=metal_charge,
                multiplicity=mult,
                fmax=0.01,
                traj_base=os.path.join(logdir, f"{base_name}_mult{mult}.traj"),
                log_base=os.path.join(logdir, f"{base_name}_mult{mult}.log"),
            )
        except Exception:
            continue
        if best_E is None or E < best_E:
            best_E, best_mult, best_mol = E, mult, trial.copy()

    if best_mult is None:
        return None
    return base_name, best_E, best_mult, best_mol, S1, S2, total_charge

# ──────────────────────────────────────────────────────────────────────
# 5.  Main driver – keeps extra DB columns
# ──────────────────────────────────────────────────────────────────────

def main(db_path: str = "testtest.db") -> None:
    spin_dict = load_spin_data(); metals = list(spin_dict.keys())
    corners = list(combinations_with_replacement(["N", "O", "S", "P", "H"], 2))

    db = connect(db_path); existing = {r.name for r in db.select()}

    raw_tasks = []
    for (m1, m2), corner, (frame_key, spacer_key) in product(combinations_with_replacement(metals, 2), corners, product(FRAME_KEYS, SPACER_KEYS)):
        s1_low, s1_high = min(spin_dict[m1]), max(spin_dict[m1])
        s2_low, s2_high = min(spin_dict[m2]), max(spin_dict[m2])
        raw_tasks.append((m1, m2, corner, frame_key, spacer_key, s1_low, s2_low, "_low" if (s1_low!=s1_high or s2_low!=s2_high) else ""))
        if s1_low!=s1_high or s2_low!=s2_high:
            raw_tasks.append((m1, m2, corner, frame_key, spacer_key, s1_high, s2_high, "_high"))

    def task_name(t):
        m1,m2,corner,fk,sk,_,_,tag = t
        return _build_name(m1[0],m1[1],m2[0],m2[1],fk,sk,corner[0],corner[1],tag)

    tasks = [t for t in raw_tasks if task_name(t) not in existing]
    if not tasks:
        print("Nothing to do – all combinations present in DB."); return

    workers = int(os.getenv("ROBSON_WORKERS", "1"))
    print(f"Launching {workers} worker(s) for {len(tasks)} tasks…", flush=True)

    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed(pool.submit(worker, t) for t in tasks):
            try:
                res = fut.result()
            except Exception:
                print("Worker crashed:", format_exc()); continue
            if res is None: continue
            name,E,mult,mol,S1,S2,Q = res
            if name in existing: continue
            db.write(mol, name=name, Energy=E, N_atoms=len(mol), Spin1=S1, Spin2=S2, Charge=Q, Multiplicity=mult, M_M_distance=mol.get_distance(0,1), username="balongn99")
            existing.add(name); done += 1
            print(f"✔ {done}/{len(tasks)} {name} mult={mult} E={E:.2f} eV ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()

