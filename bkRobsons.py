#!/usr/bin/env python3
"""
Robsons.py – original builder utilities.
Only changes:
  • ase.visualize.view is stubbed so headless runs don't raise GUI errors.
"""

from ase import Atom, Atoms
from ase.build import molecule
from ase.constraints import FixedPlane
from ase.optimize import BFGS
from tblite.ase import TBLite
import numpy as np

# ----------------------------------------------------------------------
def view(*_a, **_k):          # head-less stub
    pass

# ---- everything below is exactly your previous code ------------------
def add_atom(self, index, symbol, degree, distance=1.4, z=0.0):
    theta   = np.radians(degree)
    ref_pos = self.positions[index]
    dx, dy  = distance*np.sin(theta), distance*np.cos(theta)
    self.append(Atom(symbol, position=ref_pos + np.array([dx, dy, z])))
Atoms.add_atom = add_atom

A1 = molecule("C6H6")
A1.translate(-A1.get_center_of_mass())
del A1[11], A1[7], A1[6]
A1.add_atom(1, "C",  60)
A1.add_atom(5, "C", -60)
A1.add_atom(9, "H",  120, 1.0)
A1.add_atom(10,"H", -120, 1.0)

S1 = molecule("C6H6")
S1.translate(-S1.get_center_of_mass())
del S1[8], S1[7]

def create_site(length=2.8,
                atom_0="Fe", atom_1="Fe",
                atom_2="N",  atom_3="N",
                corner_atom="N"):
    L = length
    syms = [atom_0, atom_1, atom_2, atom_3] + [corner_atom]*4
    pos  = [
        [-L/2, 0, 0], [ L/2, 0, 0], [0,  L/2, 0], [0, -L/2, 0],
        [-L,   L/2,0],[-L,  -L/2,0],[ L,  L/2,0],[ L,  -L/2,0]
    ]
    return Atoms(symbols=syms, positions=pos)

def build_Robson(site, spacer, base, *, x=5.3, y=4.2):
    base1, base2 = base.copy(), base.copy()
    spacer1, spacer2 = spacer.copy(), spacer.copy()

    base1.translate([0, -y, 0])
    base2.set_positions(base2.get_positions()*[1,-1,1]); base2.translate([0, y, 0])
    spacer1.translate([-x,0,0])
    spacer2.set_positions(spacer2.get_positions()*[-1,1,1]); spacer2.translate([x,0,0])

    return site + base1 + base2 + spacer1 + spacer2


# Optional demo optimisation – unchanged (can comment out on cluster)
if __name__ == "__main__":
    DACS = create_site()
    rob  = build_Robson(DACS, S1, A1, 6, 5)
    rob.constrain_xy = lambda: None                     # keep demo minimal
    rob.constrain_xy()
    rob.calc = TBLite(method='GFN2-xTB', accuracy=1, charge=2,
                      multiplicity=9, solvation=("alpb","water"))
    BFGS(rob).run(fmax=0.1, steps=5)
    print("Demo run completed - energy:", rob.get_potential_energy())

