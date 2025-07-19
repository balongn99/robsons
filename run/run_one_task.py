#!/usr/bin/env python3

from __future__ import annotations
import os, sys, json, importlib
from itertools import combinations_with_replacement
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict, Sequence

import pandas as pd
import numpy as np
from numpy.linalg import norm
from ase.units import _e
from ase import Atoms
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from ase.io import write
from tblite.ase import TBLite
import tblite.interface as tb
from ase.units import Bohr, Hartree   # 1 Bohr = 0.529177 Å ; 1 Ha = 27.211 386 eV


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
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
def _finite(x):
    """Return float(x) if finite, else None (keeps JSON valid)."""
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None

def frontier_orbitals(atoms, charge, multiplicity):
    """
    HOMO/SOMO = highest MO with occupation > 0.5 e⁻
    LUMO      = next MO (index = homo_idx + 1)

    Returns (HOMO_eV, LUMO_eV) or (None, None) on failure.
    """
    try:
        # --- run a one-shot xTB single-point in native tblite ----------
        numbers   = atoms.get_atomic_numbers()
        positions = atoms.get_positions() / Bohr
        uhf       = max(0, multiplicity - 1)

        calc   = tb.Calculator("GFN2-xTB",
                               numbers, positions,
                               charge=float(charge),
                               uhf=uhf)
        result = calc.singlepoint()

        eig = result["orbital-energies"]        # Hartree (np.ndarray)
        occ = result["orbital-occupations"]     # fractional occupations

        occ_arr = np.asarray(occ)
        occ_idx = np.nonzero(occ_arr > 0.5)[0]  # “occupied” (>0.5 e⁻)

        if occ_idx.size == 0:
            return None, None                   # nothing satisfies >0.5

        homo_idx = int(occ_idx[-1])             # highest such index
        lumo_idx = homo_idx + 1                 # definition per request
        if lumo_idx >= len(eig):                # no higher orbital exists
            return float(eig[homo_idx] * Hartree), None

        homo_e = float(eig[homo_idx] * Hartree)   # → eV
        lumo_e = float(eig[lumo_idx] * Hartree)
        return homo_e, lumo_e

    except Exception as err:
        print(f"⚠ frontier-orbitals failed – {err}")
        return None, None
        
def worker(task):
    """
    task = ((sym1, ox1), (sym2, ox2), (br1, br2), acid, base, S1_ref, S2_ref)
    """
    (m1, m2, bridge, acid, base, S1_ref, S2_ref) = task
    sym1, ch1 = m1
    sym2, ch2 = m2
    br1, br2  = bridge

    tag = name_tag(sym1, ch1, sym2, ch2, acid, base, br1, br2)

    # ── per–task lock ───────────────────────────────────────────────────────
    lock = RUN_DIR / f"{tag}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return None          # another process picked it up

    try:
        # limit BLAS threads inside the worker
        os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS_PER_WORKER", "1")

        acid_frag = getattr(rob, acid)
        base_frag = getattr(rob, base)

        if br1 is None:
            site   = rob.create_site_no_bridge(atom_0=sym1, atom_1=sym2)  # type: ignore
            charge = ch1 + ch2
        else:
            site   = rob.create_site(atom_0=sym1, atom_1=sym2,
                                     atom_2=br1, atom_3=br2)              # type: ignore
            charge = ch1 + ch2 + _BRIDGE_CHARGE[br1] + _BRIDGE_CHARGE[br2]

        mol = rob.build_Robson(site, base_frag.copy(), acid_frag.copy(), x=5.3, y=5.0)

        best_E = best_mult = None
        best_mol: Optional[Atoms] = None
        best_conv = False
        best_props = {
            "mulliken_q_e": [],
            "dipole_vec_eA": [None, None, None],
            "dipole_mag_D": None,
            # ── NEW frontier-orbital slots ────────────────────────
            "HOMO_eV": None,
            "LUMO_eV": None,
            "gap_eV":  None,
        }

        # ── scan total-spin multiplicities ───────────────────────
        for mult in sorted({int(2*S + 1) for S in total_S_set(S1_ref, S2_ref)}):
            trial = mol.copy()
            try:
                opt = run_opt(trial,
                              charge=int(charge),
                              mult=mult,
                              traj=RUN_DIR / f"{tag}_m{mult}.traj",
                              log =RUN_DIR / f"{tag}_m{mult}.log")
            except Exception:
                continue

            E    = _finite(trial.get_potential_energy())
            conv = opt.converged()          # ← FIX: call the method

            better = (
                best_mol is None or
                (conv and not best_conv) or
                (conv == best_conv and (best_E is None or E < best_E))
            )
            if better:
                best_E, best_mult, best_mol, best_conv = E, mult, trial.copy(), conv

                # ── grab physical properties for the *new* best ──
                try:
                    # Mulliken charges
                    q = (trial.calc.get_atomic_charges()
                         if hasattr(trial.calc, "get_atomic_charges")
                         else trial.calc.get_charges())
                    best_props["mulliken_q_e"] = [_finite(x) for x in q]

                    # Dipole
                    dvec = (trial.calc.get_dipole_moment()
                            if hasattr(trial.calc, "get_dipole_moment")
                            else trial.calc.get_dipole())
                    dvec = [_finite(v) for v in dvec]
                    best_props["dipole_vec_eA"] = dvec
                    if all(v is not None for v in dvec):
                        best_props["dipole_mag_D"] = _finite(norm(dvec) * 4.80320427)

                    # ── NEW: frontier orbitals via tblite-python ──────
                    homo_e, lumo_e = frontier_orbitals(trial, charge, mult)
                    if homo_e is not None and lumo_e is not None:
                        best_props["HOMO_eV"] = _finite(homo_e)
                        best_props["LUMO_eV"] = _finite(lumo_e)
                        best_props["gap_eV"]  = _finite(lumo_e - homo_e)

                except Exception as err:
                    print(f"⚠ [{tag}] property grab failed – {err}")

        # ─────────── deal with failure / success ────────────────
        if best_mol is None:           # optimisation totally failed
            (RUN_DIR / f"{tag}.fail").touch()
            return None

        write(RUN_DIR / f"{tag}.traj", best_mol)
        sentinel = ".ok" if best_conv else ".fail"
        (RUN_DIR / f"{tag}{sentinel}").touch()

        # ─── JSON metadata (converged only) ─────────────────────
        if best_conv:
            meta = {
                "tag": tag,
                "energy_eV":    _finite(best_E),
                "multiplicity": int(best_mult),
                "charge":       int(charge),
                "sym1": sym1, "ox1": ch1,
                "sym2": sym2, "ox2": ch2,
                "acid": acid, "base": base,
                "bridge": [br1, br2],
                "S1_ref": S1_ref,
                "S2_ref": S2_ref,
                **best_props,
            }
            (RESULT_DIR / f"{tag}.json").write_text(json.dumps(meta, indent=2))

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

