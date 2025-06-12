#!/usr/bin/env python3

import re
import pandas as pd
from mendeleev import element

_EC_ATTRS = ("ec", "electron_configuration", "electron_configuration_semantic")
_TOKEN = re.compile(r"(\d+)([spdf])(\d+)$")

def _conf_string(el) -> str:
    for att in _EC_ATTRS:
        if hasattr(el, att):
            return str(getattr(el, att))
    raise AttributeError("Element lacks an electron-configuration attribute")

def _valence_ns_nd(conf: str):
    s_e = d_e = 0
    s_n = d_n = -1
    for tok in conf.split():
        if tok.startswith("["):
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
    el = element(sym)
    s_e, d_e = _valence_ns_nd(_conf_string(el))
    lost_s = min(charge, s_e)
    lost_d = charge - lost_s
    return max(0, d_e - lost_d)

def all_S(dn: int):
    max_unpaired = dn if dn <= 5 else 10 - dn
    max_S = max_unpaired / 2
    if dn % 2 == 0:
        return [s for s in range(int(max_S) + 1)]
    else:
        return [0.5 + i for i in range(int(max_S) + 1)]

IONS = {
    # s-block
    "Mg": [2],

    # 3d-block
    "Sc": [3],
    "Ti": [2, 3],
    "V":  [2, 3],
    "Cr": [2, 3],
    "Mn": [2, 3],
    "Fe": [2, 3],
    "Co": [2, 3],
    "Ni": [2, 3],
    "Cu": [1, 2, 3],
    "Zn": [2],

    # 4d-block
    "Y":  [3],
    "Zr": [2, 3],
    "Nb": [3],
    "Mo": [2, 3],
    "Ru": [2, 3],
    "Rh": [2, 3],
    "Pd": [2, 3],
    "Ag": [1, 2],
    "Cd": [2],

    # 5d-block
    "Hf": [3],
    "Ta": [2, 3],
    "W":  [2, 3],
    "Re": [2, 3],
    "Os": [3],
    "Ir": [1, 3],
    "Pt": [2, 3],
    "Au": [1, 3],
    "Hg": [1, 2],

    # p-block
    "Ga": [1, 3],
    "In": [1, 3],
    "Sn": [2, 3],
    "Pb": [2, 3],
    "Bi": [3],
}

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

if __name__ == "__main__":
    with pd.option_context("display.max_colwidth", None):
        print(df[["Ion", "d-electrons", "Spins"]].to_string(index=False))

