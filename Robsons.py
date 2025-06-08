#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
from typing import List
from ase.visualize import view

from ase import Atom, Atoms
from ase.io import read

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from ase.data import covalent_radii
from ase.neighborlist import NeighborList

def replace_H_with_CH3(atoms: Atoms,
                       h_index: int,
                       bond_length_CC: float = 1.54,
                       bond_length_CH: float = 1.09) -> None:
    # 1) sanity check & store old H position
    if atoms[h_index].symbol != 'H':
        raise ValueError(f"Atom {h_index} is {atoms[h_index].symbol}, not H.")
    r_H = atoms.positions[h_index].copy()

    # 2) build per-atom cutoffs & find parent
    cutoffs = [covalent_radii[a.number] * 1.2 for a in atoms]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    neighs, _ = nl.get_neighbors(h_index)
    if len(neighs) != 1:
        raise RuntimeError(f"H at index {h_index} has {len(neighs)} neighbors; expected 1.")
    parent = neighs[0]
    r_parent = atoms.positions[parent].copy()

    # 3) remove the H, adjust index
    atoms.pop(h_index)
    if parent > h_index:
        parent -= 1

    # 4) unit vector parent → old H
    n_hat = (r_H - r_parent) / np.linalg.norm(r_H - r_parent)

    # 5) place the new C in the H’s spot
    r_C = r_parent + n_hat * bond_length_CC
    atoms.append(Atom('C', position=r_C))

    # 6) compute the axis from the new C back to the parent
    axis = (r_parent - r_C)
    axis /= np.linalg.norm(axis)

    # 7) build an orthonormal frame around that axis
    arb = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(axis, arb)) > 0.9:
        arb = np.array([0.0, 1.0, 0.0])
    x_hat = np.cross(arb, axis)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(axis, x_hat)

    # 8) tetrahedral angles
    cos_a = -1.0/3.0
    sin_a = np.sqrt(1 - cos_a**2)

    # 9) append the three H’s around the C
    for i in range(3):
        phi = 2 * np.pi * i / 3
        direction = cos_a*axis + sin_a*(np.cos(phi)*x_hat + np.sin(phi)*y_hat)
        atoms.append(Atom('H', position=r_C + bond_length_CH * direction))

def change_symbol_and_adjust_bond(atoms, i, new_symbol, j, bond_length=None):

    atoms[i].symbol = new_symbol
    vec = atoms[i].position - atoms[j].position
    direction = vec / np.linalg.norm(vec)

    if bond_length is None:
        bond_length = covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number]

    atoms[i].position = atoms[j].position + direction * bond_length
    return atoms

# ----------------------------------------------------------------------
# Local data helpers
# ----------------------------------------------------------------------

try:
    _FRAME_SPACER_DIR = os.path.join(os.path.dirname(__file__), "frame_spacer")
except NameError:
    _FRAME_SPACER_DIR = os.path.join(os.getcwd(), "frame_spacer")


def _mol_path(fname: str) -> str:
    """Return absolute path of *fname* inside ``frame_spacer``."""
    return os.path.join(_FRAME_SPACER_DIR, fname)


# ----------------------------------------------------------------------
# Frame fragments  (F1 – F9)
# ----------------------------------------------------------------------

F1 = read(_mol_path('F1.mol'))
del F1[12], F1[3], F1[2], F1[1], F1[0]
replace_H_with_CH3(F1, h_index=10)
F1.translate(-F1.get_center_of_mass())

F2 = read(_mol_path('F2.mol'))
del F2[12], F2[3], F2[2], F2[1], F2[0]
F2.translate(-F2.get_center_of_mass())

F3 = read(_mol_path('F3.mol'))
F3.rotate(180,'z')
del F3[9], F3[2], F3[1]
F3.translate(-F3.get_center_of_mass())

F4 = read(_mol_path('F4.mol'))
F4.rotate(180,'z')
del F4[13], F4[4], F4[3], F4[2], F4[1]
F4.translate(-F4.get_center_of_mass())

F5 = read(_mol_path('F5.mol'))
F5.rotate(180,'z')
del F5[13], F5[4], F5[3], F5[2], F5[1]
F5.translate(-F5.get_center_of_mass())

F6 = read(_mol_path('F6.mol'))
F6.rotate(180,'z')
del F6[13], F6[4], F6[3], F6[2], F6[1]
change_symbol_and_adjust_bond(F6, i=0, new_symbol='I', j=6)
F6.translate(-F6.get_center_of_mass())

_FRAMES: List[Atoms] = [F1, F2, F3, F4, F5, F6]

# ----------------------------------------------------------------------
# Spacer fragments  (S1 – S13)
# ----------------------------------------------------------------------

S1 = read(_mol_path('S1.mol'))
S1.rotate(180,'z')
del S1[7], S1[6]
S1.translate(-S1.get_center_of_mass())

S2 = read(_mol_path('S2.mol'))
S2.rotate(90,'z')
del S2[13], S2[11]
S2.translate(-S2.get_center_of_mass())

S3 = read(_mol_path('S3.mol'))
S3.rotate(180,'z')
del S3[11], S3[10]
S3.translate(-S3.get_center_of_mass())

# S4: ethane, eclipsed
_eth=Chem.MolFromSmiles('CC');_eth=Chem.AddHs(_eth)
AllChem.EmbedMolecule(_eth,randomSeed=42)
conf=_eth.GetConformer()
for h_idx,partner in zip([2,3,4],[5,6,7]): rdMolTransforms.SetDihedralDeg(conf,h_idx,0,1,partner,0)
S4=Atoms(symbols=[a.GetSymbol() for a in _eth.GetAtoms()],positions=conf.GetPositions())
S4.rotate(-90,'z')
del S4[6], S4[3]
S4.translate(-S4.get_center_of_mass())

S5=read(_mol_path('S5.mol'))
S5.rotate(-90,'z')
del S5[4], S5[3]
S5.translate(-S5.get_center_of_mass())

S6=read(_mol_path('S6.mol'))
S6.rotate(-90,'z')
S6.rotate(-90,'y')
del S6[10], S6[6]
S6.translate(-S6.get_center_of_mass())

S7=read(_mol_path('S7.mol'))
S7.rotate(-90,'z')
S7.rotate(180,'y')
del S7[8], S7[4]
S7.translate(-S7.get_center_of_mass())

S8=read(_mol_path('S8.mol'))
S8.rotate(-90,'z')
del S8[16], S8[15]
S8.translate(-S8.get_center_of_mass())

S9=read(_mol_path('S9.mol'))
S9.rotate(-90,'z')
del S9[16], S9[15]
S9.translate(-S9.get_center_of_mass())

S10=read(_mol_path('S10.mol'))
S10.rotate(-90,'z')
del S10[15], S10[13]
S10.translate(-S10.get_center_of_mass())

S11=read(_mol_path('S11.mol'))
S11.rotate(-45,'z')
del S11[17], S11[15]
S11.translate(-S11.get_center_of_mass())

S12=read(_mol_path('S12.mol'))
S12.rotate(-90,'z')
del S12[30], S12[29]
S12.translate(-S12.get_center_of_mass())

S13=read(_mol_path('S13.mol'))
S13.rotate(-90,'z')
del S13[13], S13[12]
S13.translate(-S13.get_center_of_mass())

S99 = read(_mol_path('bor.mol'))
S99.rotate(180, "z")
del S99[9], S99[8]
S99.translate(-S99.get_center_of_mass())

_SPACERS: List[Atoms] = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S99]

# ----------------------------------------------------------------------
# Site & cage builders  (unchanged)
# ----------------------------------------------------------------------

def create_site(*, length: float = 2.8,
                atom_0: str = "Fe", atom_1: str = "Fe",
                atom_2: str = "N",  atom_3: str = "N",
                corner_atom: str = "N") -> Atoms:
    """Return a 4‑metal / 4‑corner‑N building site template."""
    L = length
    syms = [atom_0, atom_1, atom_2, atom_3] + [corner_atom] * 4
    pos  = [
        [-L/2,   0,   0],
        [ L/2,   0,   0],
        [  0 ,  L/2, 0],
        [  0 , -L/2, 0],
        [-L  ,  L/2, 0],
        [-L  , -L/2, 0],
        [ L  ,  L/2, 0],
        [ L  , -L/2, 0],
    ]
    return Atoms(symbols=syms, positions=pos)


def build_Robson(site: Atoms, spacer: Atoms, frame: Atoms, *,
                 x: float = 5.3, y: float = 4.2) -> Atoms:
    """Assemble a complete Robson cage from its parts."""
    frame1, frame2   = frame.copy(), frame.copy()
    spacer1, spacer2 = spacer.copy(), spacer.copy()

    frame1.translate([0, -y, 0])
    frame2.positions *= [1, -1, 1]
    frame2.translate([0,  y, 0])

    spacer1.translate([-x, 0, 0])
    spacer2.positions *= [-1, 1, 1]
    spacer2.translate([ x, 0, 0])

    return site + frame1 + frame2 + spacer1 + spacer2


# ----------------------------------------------------------------------
# Convenience exports
# ----------------------------------------------------------------------

__all__ = (
    # fragments
    *[f"F{i}" for i in range(1, 10)],
    *[f"S{i}" for i in range(1, 14)],
    # utilities
    "create_site", "build_Robson",
)

