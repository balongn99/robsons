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
# Frame fragments  (A1 – A6)
# ----------------------------------------------------------------------

A1 = read(_mol_path('A1.mol'))
del A1[12], A1[3], A1[2], A1[1], A1[0]
replace_H_with_CH3(A1, h_index=10)
A1.translate(-A1.get_center_of_mass())

A2 = read(_mol_path('A2.mol'))
del A2[12], A2[3], A2[2], A2[1], A2[0]
A2.translate(-A2.get_center_of_mass())

A4 = read(_mol_path('A4.mol'))
A4.rotate(180,'z')
del A4[13], A4[4], A4[3], A4[2], A4[1]
A4.translate(-A4.get_center_of_mass())

A5 = read(_mol_path('A5.mol'))
A5.rotate(180,'z')
del A5[13], A5[4], A5[3], A5[2], A5[1]
A5.translate(-A5.get_center_of_mass())

A6 = read(_mol_path('A6.mol'))
A6.rotate(180,'z')
del A6[13], A6[4], A6[3], A6[2], A6[1]
change_symbol_and_adjust_bond(A6, i=0, new_symbol='I', j=6)
A6.translate(-A6.get_center_of_mass())

_FRAMES: List[Atoms] = [A1, A2, A4, A5, A6]

# ----------------------------------------------------------------------
# Spacer fragments  (B1 – B13)
# ----------------------------------------------------------------------

B1 = read(_mol_path('B1.mol'))
B1.rotate(180,'z')
del B1[7], B1[6]
B1.translate(-B1.get_center_of_mass())

B2 = read(_mol_path('B2.mol'))
B2.rotate(90,'z')
del B2[13], B2[11]
B2.translate(-B2.get_center_of_mass())

B3 = read(_mol_path('B3.mol'))
B3.rotate(180,'z')
del B3[11], B3[10]
B3.translate(-B3.get_center_of_mass())

# B4: ethane, eclipsed
_eth=Chem.MolFromSmiles('CC');_eth=Chem.AddHs(_eth)
AllChem.EmbedMolecule(_eth,randomSeed=42)
conf=_eth.GetConformer()
for h_idx,partner in zip([2,3,4],[5,6,7]): rdMolTransforms.SetDihedralDeg(conf,h_idx,0,1,partner,0)
B4=Atoms(symbols=[a.GetSymbol() for a in _eth.GetAtoms()],positions=conf.GetPositions())
B4.rotate(-90,'z')
del B4[6], B4[3]
B4.translate(-B4.get_center_of_mass())

B5=read(_mol_path('B5.mol'))
B5.rotate(-90,'z')
del B5[4], B5[3]
B5.translate(-B5.get_center_of_mass())

B6=read(_mol_path('B6.mol'))
B6.rotate(-90,'z')
B6.rotate(-90,'y')
del B6[10], B6[6]
B6.translate(-B6.get_center_of_mass())

B7=read(_mol_path('B7.mol'))
B7.rotate(-90,'z')
B7.rotate(180,'y')
del B7[8], B7[4]
B7.translate(-B7.get_center_of_mass())

B8=read(_mol_path('B8.mol'))
B8.rotate(-90,'z')
del B8[16], B8[15]
B8.translate(-B8.get_center_of_mass())

B9=read(_mol_path('B9.mol'))
B9.rotate(-90,'z')
del B9[16], B9[15]
B9.translate(-B9.get_center_of_mass())

B10=read(_mol_path('B10.mol'))
B10.rotate(-90,'z')
del B10[15], B10[13]
B10.translate(-B10.get_center_of_mass())

B99 = read(_mol_path('bor.mol'))
B99.rotate(180, "z")
del B99[9], B99[8]
B99.translate(-B99.get_center_of_mass())

_SPACERS: List[Atoms] = [B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B99]

# ----------------------------------------------------------------------
# Site & cage builders  (unchanged)
# ----------------------------------------------------------------------

def create_site(*, length: float = 2.8,
                atom_0: str = "Fe", atom_1: str = "Fe",
                atom_2: str = "N",  atom_3: str = "N",
                corner_atom: str = "N") -> Atoms:
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
    *[f"A{i}" for i in range(1, 10)],
    *[f"B{i}" for i in range(1, 14)],
    # utilities
    "create_site", "build_Robson",
)

