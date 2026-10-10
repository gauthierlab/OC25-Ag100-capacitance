#!/usr/bin/env python3
"""
make_state_list.py -- write ROOT/s0/structures/state_list.txt (one state_id
per line, in manifest order) from structures/manifest.json, for use by the
SLURM job array in slurm/submit_md_array.sh (array index -> line number).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures-dir", default=str(SCRIPT_DIR / "structures"))
    p.add_argument("--out", default=None,
                    help="Default: <structures-dir>/state_list.txt")
    args = p.parse_args(argv)

    manifest_path = Path(args.structures_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return 1

    with open(manifest_path) as fh:
        entries = json.load(fh)

    state_ids = [e["state_id"] for e in entries]
    out_path = Path(args.out) if args.out else Path(args.structures_dir) / "state_list.txt"
    with open(out_path, "w") as fh:
        for sid in state_ids:
            fh.write(sid + "\n")

    print(f"Wrote {len(state_ids)} state_id(s) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
