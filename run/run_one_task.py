#!/usr/bin/env python3

from __future__ import annotations
import os, sys, json, importlib
from itertools import combinations_with_replacement
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Sequence

import pandas as pd
from ase import Atoms
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.io import write
from tblite.ase import TBLite

# ───── 0. paths & fragment registry ───────────────────────────────────────
RUN_DIR = Path(os.getenv("ROBSON_LOGDIR", "robson_runs"))
RUN_DIR.mkdir(exist_ok=True)
RESULT_DIR = RUN_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

rob = importlib.import_module("Robsons")
ACID_KEYS: List[str] = [k for k in rob._ACIDS if isinstance(k, str)]
BASE_KEYS: List[str] = [k for k in rob._BASES if isinstance(k, str)]

# ───── 1. spin data ───────────────────────────────────────────────────────
def _as_list(x: Any) -> List[float]:
    return list(x) if isinstance(x, (list, tuple)) else [float(x)]

def load_spin_data(mod="Spin") -> Dict[Tuple[str, int], List[float]]:
    m = importlib.import_module(mod)
    if hasattr(m, "spin_dict"):
        raw = m.spin_dict  # type: ignore[attr-defined]
        return {k: _as_list(v) for k, v in raw.items()}
    if hasattr(m, "df"):
        df = m.df                                            # type: ignore[attr-defined]
        col = "Oxidation" if "Oxidation" in df.columns else "Ox"
        return {(r.Element, int(r[col])): _as_list(r.Spins)
                for _, r in df.iterrows()}
    raise ValueError("Spin.py must expose spin_dict or df")

SPIN = load_spin_data()

# ───── 2. bridge helpers ──────────────────────────────────────────────────
BRIDGE_ATOMS = ["N", "O", "S"]
_BRIDGE_CHARGE = {"N": -2, "O": -1, "S": -1}
NO_BRIDGE_ACIDS = {"A3"}

def bridges_for(acid: str):
    return [(None, None)] if acid in NO_BRIDGE_ACIDS else \
           list(combinations_with_replacement(BRIDGE_ATOMS, 2))

# ───── 3. optimiser helper ────────────────────────────────────────────────
def _constrain_xy(self: Atoms):
    self.set_constraint([FixedPlane(i, (0, 0, 1))
                         for i, a in enumerate(self)
                         if i not in (0, 1)
                         and a.symbol != "H"
                         and -0.01 <= a.position[2] <= 0.01])
Atoms.constrain_xy = _constrain_xy

def run_opt(atoms: Atoms, *, charge: int, mult: int,
            traj: Path, log: Path, fmax=0.01, steps=50):
    atoms.constrain_xy()
    atoms.calc = TBLite(method="GFN2-xTB", accuracy=1,
                        charge=charge, multiplicity=mult,
                        solvation=("alpb", "water"), verbosity=0)
    opt = BFGS(atoms, trajectory=str(traj), logfile=str(log))
    opt.run(fmax=fmax, steps=steps)

    # ── ★ continuation if Fmax < 0.02 eV/Å but not yet converged ─────────
    if not opt.converged() and opt.fmax < 0.02:
    # Continue *with the same optimizer*
        opt.run(fmax=fmax, steps=50)
    return opt


# ───── 4. misc helpers ────────────────────────────────────────────────────
def total_S_set(S1: float, S2: float):
    lo, hi = abs(S1 - S2), S1 + S2
    return {round(lo + i, 1) for i in range(int(hi - lo) + 1)}

def name_tag(sym1: str, ch1: int, sym2: str, ch2: int,
             acid: str, base: str, br1: Optional[str], br2: Optional[str]) -> str:
    suf = "-NNNN" if br1 is None else f"-NNNN{br1}{br2}"
    return f"RMC-{sym1}_{ch1}-{sym2}_{ch2}-{acid}-{base}{suf}"

# ───── 5. core worker ─────────────────────────────────────────────────────
def worker(task):
    """task = ((sym1,ch1),(sym2,ch2),(br1,br2),acid,base,S1,S2)"""
    (m1, m2, bridge, acid, base, S1_ref, S2_ref) = task
    sym1, ch1 = m1; sym2, ch2 = m2
    br1, br2  = bridge
    tag = name_tag(sym1, ch1, sym2, ch2, acid, base, br1, br2)

    # lock file prevents duplicate work
    lock = RUN_DIR / f"{tag}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None

    try:
        os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")

        acid_frag = getattr(rob, acid)
        base_frag = getattr(rob, base)

        if br1 is None:
            site = rob.create_site_no_bridge(atom_0=sym1, atom_1=sym2)  # type: ignore[attr-defined]
            charge = ch1 + ch2
        else:
            site = rob.create_site(atom_0=sym1, atom_1=sym2,
                                   atom_2=br1, atom_3=br2)              # type: ignore[attr-defined]
            charge = ch1 + ch2 + _BRIDGE_CHARGE[br1] + _BRIDGE_CHARGE[br2]

        mol = rob.build_Robson(site, base_frag.copy(), acid_frag.copy(),
                               x=5.3, y=5.0)

        best_E = best_mult = None
        best_mol: Optional[Atoms] = None
        best_conv = False

        # ─── scan multiplicities ────────────────────────────────────────
        for mult in sorted({int(2*S + 1) for S in total_S_set(S1_ref, S2_ref)}):
            trial = mol.copy()
            try:
                opt = run_opt(trial,
                              charge=int(charge), mult=mult,
                              traj=RUN_DIR / f"{tag}_m{mult}.traj",
                              log =RUN_DIR / f"{tag}_m{mult}.log")
            except Exception:
                continue

            E, conv, fmax = trial.get_potential_energy(), opt.converged(), opt.fmax

            # extra 50 steps if Fmax < 0.02 eV/Å and not yet converged
            if (not conv) and fmax < 0.02:
                try:
                    opt2 = BFGS(
                        trial,
                        trajectory=str(RUN_DIR / f"{tag}_m{mult}.traj"),
                        logfile   =str(RUN_DIR / f"{tag}_m{mult}.log"),
                        restart=True,
                        append_trajectory=True,
                    )
                    opt2.run(fmax=0.01, steps=50)          # add 50 steps
                    E    = trial.get_potential_energy()
                    conv = opt2.converged()
                except Exception:
                    pass

            better = (
                best_E is None or
                (conv and not best_conv) or
                (conv == best_conv and E < best_E)
            )
            if better:
                best_E, best_mult, best_mol = E, mult, trial.copy()
                best_conv = conv

        # ─── write output & sentinel ────────────────────────────────────
        sentinel = ".ok" if best_conv else ".fail"

        if best_mol is None:
            (RUN_DIR / f"{tag}{sentinel}").touch()
            return tag, None, None, False

        write(RUN_DIR / f"{tag}.traj", best_mol)
        (RUN_DIR / f"{tag}{sentinel}").touch()

        if best_conv:
            (RESULT_DIR / f"{tag}.json").write_text(json.dumps({
                "tag": tag, "energy_eV": best_E, "multiplicity": best_mult,
                "charge": charge,
                "sym1": sym1, "ox1": ch1, "sym2": sym2, "ox2": ch2,
                "acid": acid, "base": base, "bridge": [br1, br2],
                "S1_ref": S1_ref, "S2_ref": S2_ref
            }, indent=2))

        return tag, best_E, best_mult, best_conv
    finally:
        lock.unlink(missing_ok=True)

# ───── 6. TSV front-end ───────────────────────────────────────────────────
def parse_line(line: str):
    f = line.strip().split()
    if len(f) != 10:
        raise ValueError("Need 10 whitespace-separated fields.")
    sym1, ch1, sym2, ch2, acid, base, br1, br2, s1, s2 = f
    bridge = (None if br1 == "." else br1,
              None if br2 == "." else br2)
    return ((sym1, int(ch1)), (sym2, int(ch2)),
            bridge, acid, base, float(s1), float(s2))

def run_task_from_line(line: str):
    return worker(parse_line(line))

if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.readline()
    run_task_from_line(raw)

