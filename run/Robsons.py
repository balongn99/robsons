#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import List
import numpy as np

from ase import Atom, Atoms
from ase.build import molecule

# ----------------------------------------------------------------------
# Function for adding atoms
# ----------------------------------------------------------------------

def add_atom(self, index, symbol, degree, distance=1.4, z=0.0):
    """
    Adds a new atom relative to a reference atom.
    Parameters:
    - atoms: ASE Atoms object
    - ref_index: index of the reference atom
    - symbol: chemical symbol of new atom
    - distance: distance from reference atom (e.g. C_H, C_C)
    - degree: angle in degrees, 0 means along +y axis
    - z: shift along z-axis
    Returns:
    - new Atoms object with added atom
    """
    ref_pos = self.positions[index]
    theta = np.radians(degree)
    dx = distance * np.sin(theta)
    dy = distance * np.cos(theta)
    new_pos = ref_pos + np.array([dx, dy, z])
    self.append(Atom(symbol, position=new_pos))

Atoms.add_atom = add_atom

# ----------------------------------------------------------------------
# Acid (aldehyde, frame) fragments
# ----------------------------------------------------------------------

A1 = molecule('C6H6')
del A1[11], A1[9], A1[7], A1[6]
A1.add_atom( 1,'C',   60)
A1.add_atom( 3,'C',  180)
A1.add_atom( 5,'C',  -60)
A1.add_atom( 8,'H',  120, 1.0)
A1.add_atom( 9,'H',  120, 1.0, -0.4)
A1.add_atom( 9,'H', -120, 1.0, -0.4)
A1.add_atom( 9,'H',  180, 0.7,  0.7)
A1.add_atom( 10,'H', -120, 1.0)

A2 = molecule('C6H6')
del A2[11], A2[7], A2[6]
A2.add_atom( 1,'C',   60)
A2.add_atom( 5,'C',  -60)
A2.add_atom( 9,'H',  120, 1.0)
A2.add_atom(10,'H', -120, 1.0)

A3 = molecule('C6H6')
A3[0].symbol = 'N'
del A3[6:12],  A3[2:5]
A3.add_atom( 1, 'C',  200)
A3.add_atom( 1, 'C',   60)
A3.add_atom( 2, 'C',- 200)
A3.add_atom( 2, 'C',  -60)
A3.add_atom( 3, 'H',  144, 1.0)
A3.add_atom( 4, 'H',  120, 1.0)
A3.add_atom( 5, 'H', -144, 1.0)
A3.add_atom( 6, 'H', -120, 1.0)

A4 = molecule('C6H6')
del A4[11], A4[9], A4[7], A4[6]
A4.add_atom( 1,'C',   60)
A4.add_atom( 3,'Cl', 180, 1.6)
A4.add_atom( 5,'C',  -60)
A4.add_atom( 8,'H',  120, 1.0)
A4.add_atom( 10,'H',-120, 1.0)

A5 = molecule('C6H6')
del A5[11], A5[9], A5[7], A5[6]
A5.add_atom( 1,'C',   60)
A5.add_atom( 3,'Br', 180, 1.8)
A5.add_atom( 5,'C',  -60)
A5.add_atom( 8,'H',  120, 1.0)
A5.add_atom(10,'H', -120, 1.0)

A6 = molecule('C6H6')
del A6[11], A6[9], A6[7], A6[6]
A6.add_atom( 1,'C',   60)
A6.add_atom( 3,'I',  180, 2.0)
A6.add_atom( 5,'C',  -60)
A6.add_atom( 8,'H',  120, 1.0)
A6.add_atom(10,'H', -120, 1.0)

_ACIDS: List[Atoms] = ["A2"]

# ----------------------------------------------------------------------
# Base (amine, spacer) fragments
# ----------------------------------------------------------------------

B1 = molecule('C6H6')
del B1[8], B1[7]

B2a = molecule('C6H6')
del B2a[8], B2a[7]
B2a.rotate(90, 'z')
B2b = B2a.copy()
del B2b[2], B2b[1]
B2b.rotate(180, 'z')
B2b.translate([0, 2.5, 0])
B2 = B2a + B2b
B2.translate(-B2.get_center_of_mass())
del B2[14], B2[7]

B4 = molecule('C2H4')
B4.rotate(90, 'x')

B5 = molecule('C2H4')
B5.rotate( 90, 'y')
B5.rotate(-90, 'z')
del B5[4], B5[2]

B6 = molecule('C3H8')
B6.rotate(-36, 'y')
del B6[10], B6[7]
B6.translate([0, 0, 0.21])

B7 = molecule('C3H6_Cs')
B7.rotate(-105, 'z')
B7.rotate( 180, 'y')
del B7[6], B7[2]

B10b = molecule('C6H6')
B10b.rotate(90, 'z')
del B10b[8], B10b[7]
B10a = B10b.copy()
B10a.rotate(180, 'z')
B10a.translate([0, 3.9, 0])
B10 = B10a + B10b
B10.translate(-B10.get_center_of_mass())
del B10[17], B10[6]

_BASES: List[Atoms] = ["B1", "B2", "B4", "B5", "B6", "B7", "B10"]

# ----------------------------------------------------------------------
# Site & cage builders  (unchanged)
# ----------------------------------------------------------------------

def create_site(*, length: float = 2.8,
                atom_0: str = "Fe", atom_1: str = "Fe",
                atom_2: str = "N",  atom_3: str = "N",
                corner_atom: str = "N"):
    """Full site with two bridge atoms (positions 2 & 3)."""
    L = length
    syms = [atom_0, atom_1, atom_2, atom_3] + [corner_atom] * 4
    pos = [
        [-L / 2, 0, 0], [L / 2, 0, 0],
        [0,  L / 2, 0], [0, -L / 2, 0],
        [-L,  L / 2, 0], [-L, -L / 2, 0],
        [ L,  L / 2, 0], [ L, -L / 2, 0],
    ]
    return Atoms(symbols=syms, positions=pos)

def create_site_no_bridge(*, length: float = 2.8,
                          atom_0: str = "Fe", atom_1: str = "Fe",
                          corner_atom: str = "N"):
    """Variant without the two bridge atoms."""
    L = length
    syms = [atom_0, atom_1] + [corner_atom] * 4
    pos = [
        [-L / 2, 0, 0], [L / 2, 0, 0],
        [-L,  L / 2, 0], [-L, -L / 2, 0],
        [ L,  L / 2, 0], [ L, -L / 2, 0],
    ]
    return Atoms(symbols=syms, positions=pos)


def build_Robson(site: Atoms, base: Atoms, acid: Atoms, *,
                 x: float = 5.3, y: float = 4.2) -> Atoms:
    """Assemble a complete Robson cage from its parts."""
    acid1, acid2 = acid.copy(), acid.copy()
    base1, base2 = base.copy(), base.copy()

    acid1.translate([0, -y, 0])
    acid2.positions *= [1, -1, 1]
    acid2.translate([0,  y, 0])

    base1.translate([-x, 0, 0])
    base2.positions *= [-1, 1, 1]
    base2.translate([ x, 0, 0])

    return site + acid1 + acid2 + base1 + base2


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
