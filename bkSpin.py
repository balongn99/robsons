#!/usr/bin/env python3
"""
spin_table.py – Build a pandas DataFrame that lists every physically
allowed spin quantum number S for selected metal ions in an octahedral
field, using mendeleev for electron configurations.

Extend the IONS dictionary at the end to add more elements/charges.
"""

import re
import pandas as pd
from typing import List
from mendeleev import element

# ──────────────────────────────────────────────────────────────
# 1.  Helpers: get valence ns / (n–1)d electron counts
# ──────────────────────────────────────────────────────────────
_EC_ATTRS = ("ec", "electron_configuration",
             "electron_configuration_semantic")
_TOKEN = re.compile(r"(\d+)([spdf])(\d+)$")

def _conf_string(el) -> str:
    for att in _EC_ATTRS:
        if hasattr(el, att):
            return str(getattr(el, att))
    raise AttributeError("mendeleev element lacks an electron-configuration attribute")

def _valence_ns_nd(conf: str):
    s_e = d_e = 0
    s_n = d_n = -1
    for tok in conf.split():
        if tok.startswith("["):           # skip core ([Ar], [Kr], …)
            continue
        m = _TOKEN.match(tok)
        if not m:
            continue
        n, sub, e = int(m[1]), m[2], int(m[3])
        if sub == "s" and n > s_n:
            s_n, s_e = n, e
        elif sub == "d" and n > d_n:
            d_n, d_e = n, e
    return s_e, d_e

def d_electrons(sym: str, charge: int) -> int:
    """Return d-electron count after removing *charge* electrons (ns first)."""
    el = element(sym)
    s_e, d_e = _valence_ns_nd(_conf_string(el))
    lost_s = min(charge, s_e)
    lost_d = charge - lost_s
    return max(0, d_e - lost_d)

# ──────────────────────────────────────────────────────────────
# 2.  All total-spin quantum numbers S for a given d^n
# ──────────────────────────────────────────────────────────────
def all_S(dn: int):
    """
    Allowed total-spin quantum numbers S for a dⁿ ion (octahedral):
        • max unpaired = n  (if n ≤ 5)  else 10 − n
        • even dn  →  S = 0, 1, 2, …      (integers)
        • odd  dn  →  S = ½, 3⁄2, 5⁄2, …  (half-integers)
    """
    max_unpaired = dn if dn <= 5 else 10 - dn
    max_S = max_unpaired / 2

    if dn % 2 == 0:                      # even electrons → integer S
        return [s for s in range(int(max_S) + 1)]
    else:                                # odd electrons → half-integer S
        # -------------   FIX is the “+ 1”  --------------------------
        return [0.5 + i for i in range(int(max_S) + 1)]


# ──────────────────────────────────────────────────────────────
# 3.  Define ions of interest  ←  edit here to add more
# ──────────────────────────────────────────────────────────────
IONS = {
    "Sc": [3],
    "Ti": [2, 3, 4],
    "V":  [2, 3, 4, 5],
    "Cr": [2, 3, 6],
    "Mn": [2, 3, 4, 6, 7],
    "Fe": [2, 3],          # Fe²⁺ will yield S = [0, 1, 2]  (not 1.5)
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2],
    "Zn": [2],
    "Mg": [2],
}

# ──────────────────────────────────────────────────────────────
# 4.  Build DataFrame
# ──────────────────────────────────────────────────────────────
rows = []
for sym, charges in IONS.items():
    for q in charges:
        dn = d_electrons(sym, q)
        rows.append({
            "Ion": f"{sym}{q:+}",
            "Element": sym,
            "Oxidation": q,
            "d-electrons": dn,
            "Spins": all_S(dn)
        })

df = (
    pd.DataFrame(rows)
      .sort_values(["Element", "Oxidation"])
      .reset_index(drop=True)
)

# ──────────────────────────────────────────────────────────────
# 5.  Display (and keep df in namespace for export)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with pd.option_context("display.max_colwidth", None):
        print(df[["Ion", "d-electrons", "Spins"]].to_string(index=False))

