#!/usr/bin/env python3
"""
run_md_oc25.py -- S0 MD driver for Ag(100)/electrolyte interfaces.

Runs Langevin NVT MD on a single state (one call = one state_id) with the
OC25 eSEN MLIP (or, for local plumbing testing, a cheap dependency-free
"dummy" calculator), and monitors reactive/interfacial chemistry every
written frame (see md_common.compute_topology / INTERFACE_SPEC.md).

Example (cluster, production):
    python run_md_oc25.py --state-id L4_n8_NaF_dN-2_far_r3 --calculator fairchem \\
        --device cuda --prerelax

Example (local plumbing test, no fairchem/GPU needed):
    python run_md_oc25.py --state-id my_test --calculator dummy \\
        --structures-dir /path/to/structures --out-dir /path/to/md \\
        --equil-ps 0.05 --prod-ps 0.1 --write-every-ps 0.01 --snapshot-every-ps 0.05

Assumptions about the fairchem API (fairchem-core, PyPI, checked ~2026-09):
  - `from fairchem.core import pretrained_mlip, FAIRChemCalculator`
  - `pretrained_mlip.get_predict_unit(model_name, device=...)` returns a
    predictor; `FAIRChemCalculator(predictor)` wraps it as an ASE calculator.
  - The OC25 checkpoints (`esen-sm-conserving-all-oc25`,
    `esen-md-direct-all-oc25`) do not require a `task_name` per the HF model
    card. `--task-name` is kept as an escape hatch: if given, or if
    construction fails in a way that suggests a task name is required, we
    retry passing `task_name=...` to get_predict_unit() and then to
    FAIRChemCalculator(). This has NOT been verified against a live install
    (fairchem-core / GPU are unavailable in the environment this was
    written in) -- treat as an assumption to confirm on first cluster run.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md_common as mc  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Langevin NVT MD driver for S0 Ag(100)/electrolyte states, "
                    "with per-frame reactive-chemistry monitoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--state-id", required=True)
    p.add_argument("--structures-dir", default=str(SCRIPT_DIR / "structures"))
    p.add_argument("--out-dir", default=str(SCRIPT_DIR / "md"))
    p.add_argument("--model", default="esen-sm-conserving-all-oc25")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--task-name", default=None,
                    help="Only needed if the installed fairchem version requires "
                         "a task_name for OC25 checkpoints; not expected to be needed.")
    p.add_argument("--temperature", type=float, default=298.0)
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--friction-per-fs", type=float, default=0.01)
    p.add_argument("--equil-ps", type=float, default=100.0)
    p.add_argument("--prod-ps", type=float, default=400.0)
    p.add_argument("--write-every-ps", type=float, default=1.0)
    p.add_argument("--snapshot-every-ps", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--calculator", choices=["fairchem", "dummy"], default="fairchem")
    p.add_argument("--restart", action="store_true",
                    help="Resume from restart.extxyz/restart_meta.json if present.")
    p.add_argument("--prerelax", action="store_true",
                    help="Run a short FIRE relaxation (FixAtoms respected) before MD.")
    p.add_argument("--prerelax-steps", type=int, default=200)
    p.add_argument("--prerelax-fmax", type=float, default=2.0,
                    help="eV/Ang force convergence target for prerelax (loose on purpose; "
                         "random-packed starts have large initial forces).")

    # topology / contact cutoffs (Angstrom), CLI-tunable per task spec
    p.add_argument("--cut-oh", type=float, default=mc.DEFAULT_CUTOFFS["oh"])
    p.add_argument("--cut-hf", type=float, default=mc.DEFAULT_CUTOFFS["hf"])
    p.add_argument("--cut-agh", type=float, default=mc.DEFAULT_CUTOFFS["agh"])
    p.add_argument("--cut-agf", type=float, default=mc.DEFAULT_CUTOFFS["agf"])
    p.add_argument("--cut-agcat", type=float, default=mc.DEFAULT_CUTOFFS["agcat"])
    p.add_argument("--cut-ago-search", type=float, default=mc.DEFAULT_CUTOFFS["ago_search"])
    p.add_argument("--top-layer-tol", type=float, default=mc.DEFAULT_TOP_LAYER_TOL)
    p.add_argument("--desorb-offset", type=float, default=mc.DEFAULT_DESORB_OFFSET)
    p.add_argument("--z-film-top", type=float, default=None,
                    help="Override manifest.json z_film_top_A (e.g. for states/tests "
                         "with no manifest entry). Desorption count is skipped (-1) "
                         "if neither is available.")

    # reactive-event warning thresholds
    p.add_argument("--thresh-hf", type=int, default=0,
                    help="Warn immediately if n_HF exceeds this.")
    p.add_argument("--thresh-h-on-ag", type=int, default=2,
                    help="Warn immediately if n_H_on_Ag exceeds this.")
    p.add_argument("--thresh-ohminus", type=int, default=2,
                    help="Warn if n_OH- exceeds this and persists.")
    p.add_argument("--thresh-persist-frames", type=int, default=5,
                    help="Consecutive written frames n_OH- must exceed threshold "
                         "before a persistence warning fires.")

    # dummy-calculator tuning (irrelevant for --calculator fairchem)
    p.add_argument("--dummy-oh-cut", type=float, default=1.25)
    p.add_argument("--dummy-agag-cut", type=float, default=3.2)
    p.add_argument("--dummy-k-bond", type=float, default=8.0)
    p.add_argument("--dummy-rep-sigma", type=float, default=1.6)
    p.add_argument("--dummy-rep-eps", type=float, default=0.02)
    p.add_argument("--dummy-rep-cut", type=float, default=3.2)
    return p


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def setup_logging(log_path):
    logger = logging.getLogger("run_md_oc25")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def get_calculator(args, atoms, log):
    if args.calculator == "dummy":
        from dummy_calculator import ToyCalculator
        log.info("Using dummy ToyCalculator (plumbing-test only, not physical).")
        return ToyCalculator(
            atoms, oh_cut=args.dummy_oh_cut, ag_cut=args.dummy_agag_cut,
            k_bond=args.dummy_k_bond, rep_sigma=args.dummy_rep_sigma,
            rep_eps=args.dummy_rep_eps, rep_cut=args.dummy_rep_cut,
        )

    # --- fairchem / OC25 ---
    try:
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
    except ImportError as e:
        log.error("fairchem-core is not importable: %s", e)
        raise

    log.info("Loading OC25 predict unit '%s' on device=%s", args.model, args.device)
    try:
        predictor = pretrained_mlip.get_predict_unit(args.model, device=args.device)
    except TypeError:
        log.warning("get_predict_unit() rejected call signature; retrying with task_name=%s",
                    args.task_name)
        predictor = pretrained_mlip.get_predict_unit(
            args.model, device=args.device, task_name=args.task_name)

    try:
        calc = FAIRChemCalculator(predictor)
    except TypeError:
        log.warning("FAIRChemCalculator() rejected call signature; retrying with task_name=%s",
                    args.task_name)
        calc = FAIRChemCalculator(predictor, task_name=args.task_name)
    return calc


def init_velocities(atoms, temperature_K, seed):
    rng = np.random.default_rng(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    fmask = mc.free_mask(atoms)
    vel = atoms.get_velocities()
    vel[~fmask] = 0.0
    if fmask.any():
        masses = atoms.get_masses()[fmask]
        p = (vel[fmask] * masses[:, None]).sum(axis=0)
        vcom = p / masses.sum()
        vel[fmask] -= vcom
    atoms.set_velocities(vel)


def run_prerelax(atoms, steps, fmax, log):
    from ase.optimize import FIRE
    log.info("Running FIRE pre-relaxation for up to %d steps (fmax=%.2f eV/Ang)",
              steps, fmax)
    opt = FIRE(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    log.info("Pre-relaxation finished after %d steps (converged=%s)",
              opt.nsteps, opt.converged())


def write_csv_row(path, fields, row):
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a") as fh:
        if new_file:
            fh.write(",".join(fields) + "\n")
        fh.write(",".join(str(row[k]) for k in fields) + "\n")


def atomic_write_atoms(path, atoms):
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp, atoms, format="extxyz")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    args = build_parser().parse_args(argv)

    struct_dir = Path(args.structures_dir) / args.state_id
    init_path = struct_dir / "init.extxyz"
    out_dir = Path(args.out_dir) / args.state_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(exist_ok=True)

    log = setup_logging(out_dir / "run.log")
    log.info("=== run_md_oc25.py state=%s calculator=%s ===", args.state_id, args.calculator)
    log.info("args: %s", vars(args))

    manifest_entry = mc.load_manifest_entry(args.structures_dir, args.state_id)
    z_film_top = None
    if manifest_entry is not None:
        z_film_top = manifest_entry.get("z_film_top_A")
    if args.z_film_top is not None:
        z_film_top = args.z_film_top
    if z_film_top is None:
        log.warning("No z_film_top available (manifest missing entry/key and "
                    "--z-film-top not given); n_desorbed will be reported as -1.")

    cutoffs = {
        "oh": args.cut_oh, "hf": args.cut_hf, "agh": args.cut_agh,
        "agf": args.cut_agf, "agcat": args.cut_agcat,
        "ago_search": args.cut_ago_search,
    }

    thermo_path = out_dir / "thermo.csv"
    topo_path = out_dir / "topology.csv"
    traj_path = out_dir / "traj.extxyz"
    restart_path = out_dir / "restart.extxyz"
    meta_path = out_dir / "restart_meta.json"

    resumed = False
    if args.restart and restart_path.exists() and meta_path.exists():
        atoms = read(restart_path)
        mc.attach_fixatoms(atoms)
        with open(meta_path) as fh:
            meta = json.load(fh)
        start_step = int(meta["step"])
        resumed = True
        log.info("Resuming from restart.extxyz at absolute step %d (t=%.3f ps)",
                 start_step, meta.get("time_ps", float("nan")))
    else:
        if not init_path.exists():
            log.error("init.extxyz not found: %s", init_path)
            raise FileNotFoundError(init_path)
        atoms = read(init_path)
        mc.attach_fixatoms(atoms)
        start_step = 0
        for p in (thermo_path, topo_path, traj_path):
            if p.exists():
                p.unlink()
        log.info("Fresh start from %s (%d atoms, %d free, %d fixed)",
                 init_path, len(atoms), int(mc.free_mask(atoms).sum()),
                 int((~mc.free_mask(atoms)).sum()))

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    if args.seed is None:
        log.info("No --seed given; using random seed=%d (recorded for reproducibility)", seed)

    calc = get_calculator(args, atoms, log)
    atoms.calc = calc

    if args.prerelax and not resumed:
        run_prerelax(atoms, args.prerelax_steps, args.prerelax_fmax, log)

    if not resumed:
        init_velocities(atoms, args.temperature, seed)

    dt_fs = args.timestep_fs
    dt = dt_fs * units.fs
    friction = args.friction_per_fs / units.fs
    dyn = Langevin(atoms, dt, temperature_K=args.temperature, friction=friction,
                    rng=np.random.default_rng(seed))

    equil_steps_total = max(0, round(args.equil_ps * 1000.0 / dt_fs))
    prod_steps_total = max(0, round(args.prod_ps * 1000.0 / dt_fs))
    total_steps = equil_steps_total + prod_steps_total
    write_every = max(1, round(args.write_every_ps * 1000.0 / dt_fs))
    snap_every = max(1, round(args.snapshot_every_ps * 1000.0 / dt_fs))
    restart_every = max(1, round(10_000.0 / dt_fs))  # 10 ps

    log.info("equil_steps=%d prod_steps=%d total_steps=%d write_every=%d "
             "snap_every=%d restart_every=%d dt_fs=%.4f",
             equil_steps_total, prod_steps_total, total_steps, write_every,
             snap_every, restart_every, dt_fs)

    if start_step >= total_steps:
        log.info("start_step (%d) >= total_steps (%d); nothing to do.",
                 start_step, total_steps)
        return

    state = {
        "wall_t0": time.time(),
        "last_thermo_t": time.time(),
        "last_thermo_step": start_step,
        "ohminus_history": [],
    }

    def abs_step():
        return start_step + dyn.nsteps

    def phase_of(s):
        return "equil" if s <= equil_steps_total else "prod"

    def do_thermo_and_topology():
        if resumed and dyn.nsteps == 0:
            # ASE calls attached observers once before the stepping loop
            # starts; on a resumed run that pre-loop call corresponds to the
            # already-checkpointed step and would otherwise duplicate it.
            return
        s = abs_step()
        t_ps = s * dt_fs / 1000.0
        phase = phase_of(s)
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        n_free = max(1, int(mc.free_mask(atoms).sum()))
        # 3 DOF per free atom minus COM removal (~3); good enough for monitoring
        dof = max(1, 3 * n_free - 3)
        T = 2.0 * ekin / (units.kB * dof)

        write_csv_row(thermo_path, mc.THERMO_FIELDS, {
            "step": s, "time_ps": f"{t_ps:.6f}", "T_K": f"{T:.3f}",
            "Epot_eV": f"{epot:.6f}", "Ekin_eV": f"{ekin:.6f}", "phase": phase,
        })

        topo = mc.compute_topology(
            atoms, cutoffs=cutoffs, z_film_top=z_film_top,
            top_layer_tol=args.top_layer_tol, desorb_offset=args.desorb_offset,
        )
        row = {"step": s, "time_ps": f"{t_ps:.6f}", "phase": phase}
        for k in mc.TOPOLOGY_FIELDS:
            if k in ("step", "time_ps", "phase"):
                continue
            v = topo[k]
            row[k] = f"{v:.4f}" if isinstance(v, float) else v
        write_csv_row(topo_path, mc.TOPOLOGY_FIELDS, row)

        # append frame to traj.extxyz (fixed/role arrays are in atoms.arrays
        # already and are untouched by the dynamics, so they round-trip)
        write(traj_path, atoms, format="extxyz", append=True)

        # --- reactive-event warnings ---
        if topo["n_HF"] > args.thresh_hf:
            log.warning("frame step=%d t=%.3fps: n_HF=%d exceeds threshold (%d)",
                        s, t_ps, topo["n_HF"], args.thresh_hf)
        if topo["n_H_on_Ag"] > args.thresh_h_on_ag:
            log.warning("frame step=%d t=%.3fps: n_H_on_Ag=%d exceeds threshold (%d)",
                        s, t_ps, topo["n_H_on_Ag"], args.thresh_h_on_ag)
        hist = state["ohminus_history"]
        hist.append(topo["n_OH-"] > args.thresh_ohminus)
        if len(hist) > args.thresh_persist_frames:
            hist.pop(0)
        if len(hist) == args.thresh_persist_frames and all(hist):
            log.warning(
                "frame step=%d t=%.3fps: n_OH- > %d has persisted for %d consecutive "
                "written frames (current n_OH-=%d)",
                s, t_ps, args.thresh_ohminus, args.thresh_persist_frames, topo["n_OH-"])

        # --- snapshot (production only) ---
        if phase == "prod" and s % snap_every == 0:
            snap_path = out_dir / "snapshots" / f"frame_{s:08d}.extxyz"
            write(snap_path, atoms, format="extxyz")
            log.info("Wrote DFT-candidate snapshot %s", snap_path.name)

        # --- throughput logging ---
        now = time.time()
        dsteps = s - state["last_thermo_step"]
        dt_wall = now - state["last_thermo_t"]
        if dsteps > 0 and dt_wall > 0:
            ns_per_day = (dsteps * dt_fs * 1e-6) / (dt_wall / 86400.0)
            log.info("step=%d t=%.3fps phase=%s T=%.1fK Epot=%.4feV ns/day=%.3f",
                     s, t_ps, phase, T, epot, ns_per_day)
        state["last_thermo_t"] = now
        state["last_thermo_step"] = s

    def do_restart():
        if resumed and dyn.nsteps == 0:
            return
        s = abs_step()
        t_ps = s * dt_fs / 1000.0
        atomic_write_atoms(restart_path, atoms)
        mc.atomic_write_json(meta_path, {
            "step": s, "time_ps": t_ps, "phase": phase_of(s), "seed": seed,
        })
        log.info("Checkpoint written at step=%d t=%.3fps", s, t_ps)

    dyn.attach(do_thermo_and_topology, interval=write_every)
    dyn.attach(do_restart, interval=restart_every)

    remaining = total_steps - start_step
    log.info("Starting dynamics.run(%d) (already completed %d/%d steps)",
             remaining, start_step, total_steps)
    dyn.run(remaining)

    # always leave a final checkpoint
    do_restart()
    log.info("Run complete: total wall time %.1f s", time.time() - state["wall_t0"])


if __name__ == "__main__":
    main()
