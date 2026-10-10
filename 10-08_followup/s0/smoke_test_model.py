#!/usr/bin/env python3
"""
smoke_test_model.py -- cluster-side sanity check for the OC25 eSEN checkpoint.

Loads the model, evaluates energy/forces on one S0 structure, and prints
max|F| and the per-atom energy. Large initial forces are EXPECTED for the
packmol-built starting structures (random packing) -- that is what
--prerelax (here, or the same flag in run_md_oc25.py) is for.

Usage:
    huggingface-cli login   # once, needs access to the gated facebook/OC25 repo
    python smoke_test_model.py --state-id L4_n8_NaF_dN-2_far_r3
    python smoke_test_model.py --state-id L4_n8_NaF_dN-2_far_r3 --prerelax --prerelax-steps 200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md_common as mc  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--state-id", required=True)
    p.add_argument("--structures-dir", default=str(SCRIPT_DIR / "structures"))
    p.add_argument("--model", default="esen-sm-conserving-all-oc25")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--task-name", default=None)
    p.add_argument("--prerelax", action="store_true")
    p.add_argument("--prerelax-steps", type=int, default=200)
    p.add_argument("--prerelax-fmax", type=float, default=2.0)
    args = p.parse_args(argv)

    from ase.io import read
    import numpy as np

    init_path = Path(args.structures_dir) / args.state_id / "init.extxyz"
    print(f"Reading {init_path}")
    atoms = read(init_path)
    mc.attach_fixatoms(atoms)
    print(f"{len(atoms)} atoms, {int((~mc.free_mask(atoms)).sum())} fixed")

    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    print(f"Loading predict unit '{args.model}' on device={args.device} ...")
    t0 = time.time()
    try:
        predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    except TypeError:
        predictor = pretrained_mlip.get_predict_unit(
            args.model, device=args.device, task_name=args.task_name)
    try:
        calc = FAIRChemCalculator(predictor)
    except TypeError:
        calc = FAIRChemCalculator(predictor, task_name=args.task_name)
    print(f"Model loaded in {time.time() - t0:.1f} s")

    atoms.calc = calc

    if args.prerelax:
        from ase.optimize import FIRE
        print(f"Running FIRE pre-relaxation (<= {args.prerelax_steps} steps, "
              f"fmax={args.prerelax_fmax} eV/Ang) ...")
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=args.prerelax_fmax, steps=args.prerelax_steps)
        print(f"Pre-relax finished after {opt.nsteps} steps (converged={opt.converged()})")

    t0 = time.time()
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    dt = time.time() - t0

    fmax = float(np.abs(f).max())
    print(f"Single-point eval took {dt:.2f} s")
    print(f"Total energy:      {e:.6f} eV")
    print(f"Energy per atom:   {e / len(atoms):.6f} eV/atom")
    print(f"max|F| (all atoms): {fmax:.4f} eV/Angstrom")
    free = mc.free_mask(atoms)
    if free.any():
        print(f"max|F| (free atoms only): {float(np.abs(f[free]).max()):.4f} eV/Angstrom")


if __name__ == "__main__":
    main()
