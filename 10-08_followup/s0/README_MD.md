# S0 MD stage: run_md_oc25.py / analyze_md.py

See `INTERFACE_SPEC.md` for the shared file contract. See `ENVIRONMENT.md`
for the cluster environment / fairchem install / HuggingFace login.

## What each script does

- **`md_common.py`** -- shared constants and helpers (fixed/role handling,
  the reactive-chemistry topology classifier, SPC/E charges for the dipole
  proxy). Imported by both driver scripts so their definitions can't drift
  apart.
- **`dummy_calculator.py`** -- `ToyCalculator`: a cheap, dependency-free ASE
  calculator (frozen-topology harmonic bonds + short-range repulsion) used
  only for local plumbing testing via `--calculator dummy`. Not physical;
  never use it for production MD.
- **`run_md_oc25.py`** -- Langevin NVT MD driver for one state_id. Optional
  FIRE pre-relaxation, then equilibration + production, writing
  `traj.extxyz`, `thermo.csv`, `topology.csv`, periodic `snapshots/`,
  atomic `restart.extxyz`/`restart_meta.json` checkpoints every 10 ps, and
  `run.log`. Supports `--restart` to resume after a requeue/timeout.
- **`analyze_md.py`** -- post-hoc analysis of one or more finished MD runs:
  density profiles, water dipole orientation, ion contact statistics, film
  density, bulk ion concentration, a classical dipole-proxy potential, a
  two-sided-start ("surface" vs "far" initial ion placement) comparison,
  bulk-box density/RDF/diffusion/freezing checks, and a cross-state
  `summary.csv`.
- **`make_state_list.py`** -- writes `structures/state_list.txt` from
  `structures/manifest.json`, consumed by the SLURM array (line number =
  `SLURM_ARRAY_TASK_ID`).
- **`smoke_test_model.py`** -- cluster-side check that the OC25 checkpoint
  loads and gives finite energy/forces on a real S0 structure.
- **`slurm/submit_md_array.sh`**, **`slurm/submit_one.sh`** -- SLURM
  scripts; they source `slurm/cluster.env` for modules, conda env and token.

## Commands

### (a) Local plumbing test (no fairchem/GPU needed)

```bash
python run_md_oc25.py --state-id <toy_state_id> \
    --structures-dir <dir with toy_state_id/init.extxyz> \
    --out-dir <scratch>/md \
    --calculator dummy --prerelax --prerelax-steps 300 \
    --equil-ps 0.05 --prod-ps 0.1 --write-every-ps 0.01 --snapshot-every-ps 0.05

python analyze_md.py --state-id <toy_state_id> \
    --md-dir <scratch>/md --structures-dir <dir with toy_state_id/init.extxyz>
```

`--prerelax` matters even for the toy potential: packing 20 waters onto a
small 3x3 Ag(100) footprint (or any packmol-built real state) starts with
large repulsive contacts, and a short FIRE relaxation before MD avoids a
numerically unstable first few timesteps.

### (b) Cluster run of the whole S0 array

```bash
python make_state_list.py                       # writes structures/state_list.txt
sbatch --partition=<p> --gres=gpu:1 --time=24:00:00 \
       --array=1-$(wc -l < structures/state_list.txt)%16 slurm/submit_md_array.sh
# or, for one state:
sbatch --partition=<p> --gres=gpu:1 --time=24:00:00 slurm/submit_one.sh <state_id>
```

Cluster-specific settings (modules, conda env, HuggingFace token) are read from
`slurm/cluster.env`; copy `slurm/cluster.env.example` and fill it in first.

### (c) Analysis (after MD has produced production-phase data)

```bash
python analyze_md.py --all --md-dir md --structures-dir structures
# or specific states:
python analyze_md.py --state-id L4_n8_NaF_dN-2_far_r3 L4_n8_NaF_dN-2_surface_r1
```

Outputs land in `md/<state_id>/analysis/` (CSVs + PNGs, PNGs skipped if
matplotlib is unavailable) and `md/summary.csv` (one row per state) plus
`md/_comparisons/` for the surface-vs-far ion-placement comparisons.

## Monitored reactive events and thresholds (per-frame, in topology.csv)

| Quantity | Cutoff (default, CLI-tunable) | Warning trigger |
|---|---|---|
| O-H bond | 1.25 A | classifies water/OH-/H3O+ |
| H-F bond (`n_HF`) | 1.10 A | any `n_HF > 0` -> immediate WARNING |
| H on Ag (`n_H_on_Ag`) | 2.00 A (Ag-H), excludes H already O-bonded | `n_H_on_Ag > 2` -> immediate WARNING |
| F contact / cation contact | 2.90 A (Ag-F) / 3.60 A (Ag-Na,Cs), top Ag layer only | reported, not a warning trigger |
| OH- persistence (`n_OH-`) | O with exactly 1 bonded H | `n_OH- > 2` for 5 consecutive written frames -> WARNING |
| Desorbed water (`n_desorbed`) | O with z > z_film_top (manifest) + 5 A | reported (-1 if z_film_top unavailable) |

All cutoffs/thresholds are CLI flags on `run_md_oc25.py`
(`--cut-oh`, `--cut-hf`, `--cut-agh`, `--cut-agf`, `--cut-agcat`,
`--thresh-hf`, `--thresh-h-on-ag`, `--thresh-ohminus`,
`--thresh-persist-frames`). Warnings are logged to `run.log`; the run is
never stopped automatically.

`topology.csv` columns: `n_HF, n_OH-, n_H3O+, n_H_on_Ag, n_F_contact, n_cation_contact,
n_water_dissociated` (as in `INTERFACE_SPEC.md`) followed by the diagnostics
`n_water`, `min_Ag_O_dist`, `z_top_Ag`, `n_desorbed`; `n_water_dissociated := n_OH- + n_H3O+`.

## Known assumptions / things to verify on first real cluster run

- **fairchem-core API**: `from fairchem.core import pretrained_mlip,
  FAIRChemCalculator`; `pretrained_mlip.get_predict_unit(model,
  device=...)` then `FAIRChemCalculator(predictor)`, no `task_name`
  required for OC25 checkpoints. `--task-name` exists as a fallback on both
  `run_md_oc25.py` and `smoke_test_model.py` (retried once, on `TypeError`,
  as `task_name=<value>`). **Not verified against a live install** --
  fairchem-core and a GPU were unavailable while writing this. See
  `ENVIRONMENT.md` and the module docstrings for details.
- **Package name/version**: `fairchem-core` on PyPI, confirmed via a web
  check (2026-09); pin an exact version before a production run.
- **Langevin friction units**: `--friction-per-fs` is converted to ASE's
  internal time units as `friction_per_fs / ase.units.fs`; verified this
  gives a stable, correctly-thermostatted toy run locally.
- **Dummy calculator** is a frozen-topology harmonic-bond + short-range
  repulsion potential; it is deliberately not physical and is only used
  with `--calculator dummy`.

## Offline test status

Run end-to-end with `--calculator dummy` (no fairchem, no GPU) on a small synthetic
interface, a synthetic bulk-water box, and one real 1843-atom S0 state: every output
file in `INTERFACE_SPEC.md` is produced and parses, `fixed`/`role` arrays survive the
trajectory round-trip, and `--restart` resumes without duplicating frames. The fairchem
code path itself has not been executed; the first cluster run (Step 0 of `README.md`)
is that test.
