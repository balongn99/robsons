#!/usr/bin/env python3
"""Filter Robson database for converged structures ('.ok' sentinels).

Usage
-----
$ ./filter_ok.py                 # robson.db → robson_ok.db
$ ./filter_ok.py -i my.db -o ok.db --delete-ok
"""

from pathlib import Path
from ase.db import connect
import argparse
import sys

def main(db_in: str, db_out: str, run_dir: str, delete_ok: bool) -> None:
    run_dir_path = Path(run_dir)
    ok_tags = {p.stem for p in run_dir_path.glob("*.ok")}

    if not ok_tags:
        sys.exit("No *.ok files found – nothing to do.")

    # create / truncate output DB
    out_db = connect(db_out, append=False)

    kept = 0
    with connect(db_in) as in_db:
        for row in in_db.select():            # iterate once over input DB
            if row.name in ok_tags:           # keep only converged tags
                kv = {k: v for k, v in row.key_value_pairs.items() if k != "name"}
                out_db.write(row.toatoms(),
                             name=row.name,
                             **kv)
                kept += 1

    print(f"✓ Copied {kept} converged structures to {db_out}")

    if delete_ok:
        for p in run_dir_path.glob("*.ok"):
            p.unlink(missing_ok=True)
        print("✓ Removed all '.ok' sentinel files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract converged (.ok) molecules from an ASE database")
    parser.add_argument("-i", "--input",  default="robson.db",
                        help="input ASE database")
    parser.add_argument("-o", "--output", default="robson_ok.db",
                        help="output ASE database containing only ok entries")
    parser.add_argument("-r", "--run-dir", default="robson_runs",
                        help="directory that contains *.ok / *.fail sentinels")
    parser.add_argument("--delete-ok", action="store_true",
                        help="delete sentinel files after successful copy")
    args = parser.parse_args()

    main(args.input, args.output, args.run_dir, args.delete_ok)
