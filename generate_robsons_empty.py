#!/usr/bin/env python3

from __future__ import annotations
import importlib, os, time
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

# ──────────────────────────────────────────────────────────────────────
# 0.  Fragment library (auto-discovered)
# ──────────────────────────────────────────────────────────────────────
rob = importlib.import_module("Robsons_empty")

def _fragment_keys(prefix: str) -> List[str]:
    """Return A* / B* keys in alpha-numeric order (A1 < A10 < A2 …)."""
    keys = [k for k, v in vars(rob).items()
            if k.startswith(prefix) and isinstance(v, Atoms)]
    return sorted(keys, key=lambda s: (len(s), s))

FRAME_KEYS  = _fragment_keys("A")   # frames  A*
SPACER_KEYS = _fragment_keys("B")   # spacers B*

if not FRAME_KEYS:
    raise RuntimeError('No frame fragments (variables "A*") found in Robsons_empty.py!')
if not SPACER_KEYS:
    raise RuntimeError('No spacer fragments (variables "B*") found in Robsons_empty.py!')

print(f"Discovered frames:  {', '.join(FRAME_KEYS)}")
print(f"Discovered spacers: {', '.join(SPACER_KEYS)}")

# ──────────────────────────────────────────────────────────────────────
# 1.  Constrain helper
# ──────────────────────────────────────────────────────────────────────
def constrain_xy(self):
    constraints = [FixedPlane(i, (0, 0, 1))
                   for i in range(len(self)) if i not in (0, 1)]
    self.set_constraint(constraints)

Atoms.constrain_xy = constrain_xy  # monkey-patch

# ──────────────────────────────────────────────────────────────────────
# 2.  Spin helpers  (exactly ONE spin per oxidation state)
# ──────────────────────────────────────────────────────────────────────
def load_spin_data(module: str = "Spin") -> Dict[Tuple[str, int], List[float]]:
    """Load {(element, ox_state): [S]} either from spin_dict or df."""
    spin = importlib.import_module(module)
    if hasattr(spin, "spin_dict"):
        raw: Dict[Tuple[str, int], Sequence[float]] = spin.spin_dict
        df = pd.DataFrame({"Element": [k[0] for k in raw],
                           "Oxidation": [k[1] for k in raw],
                           "Spins": list(raw.values())})
    elif hasattr(spin, "df"):
        df = spin.df
    else:
        raise AttributeError("Spin.py must define either `spin_dict` or `df`")
    # Ensure each value is a list (len == 1)
    return {(r.Element, int(r.Oxidation)): list(map(float, r.Spins))
            for _, r in df.iterrows()}

# ──────────────────────────────────────────────────────────────────────
# 3.  Utilities
# ──────────────────────────────────────────────────────────────────────
def total_S_set(S1: float, S2: float) -> set[float]:
    """All allowed total-spin quantum numbers S = |S1−S2| … S1+S2."""
    low, high = abs(S1 - S2), S1 + S2
    return {round(low + i, 1) for i in range(int(high - low) + 1)}

# ──────────────────────────────────────────────────────────────────────
# 4.  Geometry optimisation helper
# ──────────────────────────────────────────────────────────────────────
def optimise(atoms: Atoms, *, charge: int, multiplicity: int,
             fmax: float = 0.01,
             trajectory: Optional[str] = None,
             logfile: Optional[str] = None) -> float:
    """BFGS geometry optimisation (GFN2-xTB)."""
    atoms.constrain_xy()
    atoms.calc = TBLite(method="GFN2-xTB", accuracy=1,
                        charge=charge, multiplicity=multiplicity,
                        solvation=("alpb", "water"), verbosity=0)
    opt = BFGS(atoms, trajectory=trajectory, logfile=logfile) \
          if (trajectory or logfile) else BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=200)
    return atoms.get_potential_energy()

# ──────────────────────────────────────────────────────────────────────
# 5.  Worker process
# ──────────────────────────────────────────────────────────────────────
def _build_name(sym1: str, ch1: int, sym2: str, ch2: int,
                frame_key: str, spacer_key: str) -> str:
    """Return string with constant NNNN suffix (no tag)."""
    return (f"RMC-{sym1}_{ch1}-{sym2}_{ch2}-"
            f"{frame_key}-{spacer_key}-NNNN")

def worker(args) -> Optional[Tuple[str, float, int, Atoms,
                                   float, float, int]]:
    (m1, m2, frame_key, spacer_key, S1_ref, S2_ref) = args

    sym1, ch1 = m1
    sym2, ch2 = m2

    # Limit OpenMP threads inside each worker
    os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")

    logdir = os.getenv("ROBSON_LOGDIR", "robson_runs")
    os.makedirs(logdir, exist_ok=True)

    frame_fragment  = getattr(rob, frame_key)
    spacer_fragment = getattr(rob, spacer_key)

    base_name = _build_name(sym1, ch1, sym2, ch2, frame_key, spacer_key)

    # ── Build initial structure (no shared corners) ───────────────────
    site = rob.create_site(atom_0=sym1, atom_1=sym2)  # only two metals
    mol  = rob.build_Robson(site,
                            spacer_fragment.copy(),
                            frame_fragment.copy(),
                            x=6.0, y=5.0)

    multiplicities = sorted({int(2*S + 1) for S in total_S_set(S1_ref, S2_ref)})
    total_charge   = int(ch1 + ch2)      # no corner charge

    best_energy = best_mult = best_mol = None
    for mult in multiplicities:
        trial = mol.copy()
        try:
            E = optimise(trial,
                         charge=total_charge,
                         multiplicity=mult,
                         fmax=0.01,
                         trajectory=os.path.join(logdir, f"{base_name}_mult{mult}.traj"),
                         logfile=os.path.join(logdir,  f"{base_name}_mult{mult}.log"))
        except Exception:
            continue
        if best_energy is None or E < best_energy:
            best_energy, best_mult, best_mol = E, mult, trial.copy()

    if best_mult is None:
        return None

    return (base_name, best_energy, best_mult, best_mol,
            S1_ref, S2_ref, total_charge)

# ──────────────────────────────────────────────────────────────────────
# 6.  Main driver
# ──────────────────────────────────────────────────────────────────────
def main(db_path: str = "robsonscatalyst.db") -> None:
    spin_dict = load_spin_data()
    metals    = list(spin_dict.keys())

    # 1) Connect & cache existing names
    db       = connect(db_path)
    existing = {row.name for row in db.select()}

    # 2) Build task list
    tasks: List[Tuple] = []
    for (m1, m2), (frame_key, spacer_key) in product(
            combinations_with_replacement(metals, 2),
            product(FRAME_KEYS, SPACER_KEYS)):
        S1 = float(spin_dict[m1][0])  # exactly one spin per state
        S2 = float(spin_dict[m2][0])
        tasks.append((m1, m2, frame_key, spacer_key, S1, S2))

    # Remove tasks already in DB
    def _name_from_tuple(t: Tuple) -> str:
        (m1, m2, frame_key, spacer_key, _, _) = t
        return _build_name(m1[0], m1[1], m2[0], m2[1],
                           frame_key, spacer_key)

    tasks = [t for t in tasks if _name_from_tuple(t) not in existing]

    max_workers = int(os.getenv("ROBSON_WORKERS", "1"))
    print(f"Running with {max_workers} worker process(es)… ({len(tasks)} to do)", flush=True)

    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:
                print("‼ Worker crashed for", _name_from_tuple(futures[fut]),
                      format_exc(), flush=True)
                continue
            if result is None:
                continue

            name, E, mult, mol, S1, S2, charge = result
            if name in existing:     # race-condition guard
                continue

            # ── Custom columns ─────────────────────────────────────────
            n_atoms = len(mol)
            mm_dist = mol.get_distance(0, 1)
            db.write(
                mol,
                name=name,
                Energy=E,
                N_atoms=n_atoms,
                Spin1=S1,
                Spin2=S2,
                Charge=charge,
                Multiplicity=mult,
                M_M_distance=mm_dist,
                username='balongn99'
            )
            # ───────────────────────────────────────────────────────────
            existing.add(name)
            done += 1
            print(f"✔ {done}/{len(tasks)}  {name}  mult={mult}  "
                  f"E={E:.2f} eV  ({(time.time()-t0)/60:.1f} min)", flush=True)

    print(f"Finished: {done} new structures written "
          f"({(time.time()-t0)/60:.1f} min total)")

if __name__ == "__main__":
    main()

