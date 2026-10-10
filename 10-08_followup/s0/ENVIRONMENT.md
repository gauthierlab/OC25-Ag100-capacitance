# S0 MD environment (run_md_oc25.py / analyze_md.py / smoke_test_model.py)

## What's needed

- Python >= 3.10
- PyTorch with CUDA support (matched to the cluster's CUDA/driver version --
  check `nvidia-smi` and the cluster docs)
- `fairchem-core` (PyPI package name confirmed via a live web check,
  2026-09; the pinned version below was the latest at that time --
  re-check before installing, since this is a fast-moving package)
- `ase` (this repo's other S0 scripts were developed against ASE 3.30)
- `numpy`, `scipy`, `pandas`, `matplotlib`

None of `fairchem-core`, a CUDA-capable PyTorch, or a GPU were available in
the environment this code was written in, so the exact install command below
is a best effort from the package's PyPI/HF documentation, not something
that was actually exercised end-to-end here. `run_md_oc25.py --calculator
dummy` (no fairchem/GPU needed) was used instead for local plumbing testing
-- see README_MD.md.

## Install (adjust CUDA/torch build for the actual cluster)

```bash
conda create -n oc25 python=3.11 -y
conda activate oc25

# Install a CUDA-matched torch build FIRST (see https://pytorch.org for the
# correct index-url for the cluster's CUDA version), e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install fairchem-core   # PyPI, ~2.20.x as of 2026-09; pin a version once confirmed
pip install ase==3.30.* numpy scipy pandas matplotlib
```

## HuggingFace login (required once per user/cluster account)

The OC25 checkpoints live in the gated `facebook/OC25` HuggingFace repo.
Request access on the model page, then authenticate once (the token is
cached under `~/.cache/huggingface` and is picked up automatically by
subsequent batch jobs run as the same user):

```bash
huggingface-cli login
```

Alternatively set `HF_TOKEN` in `slurm/cluster.env`; the submission scripts
export it if non-empty.

## Smoke test (run on the cluster, with a GPU, after the steps above)

`smoke_test_model.py` loads the model, evaluates one S0 structure's
energy/forces, and prints max|F| and the per-atom energy. Large initial
forces are expected for the packmol-built starting structures -- pass
`--prerelax` to FIRE-relax first (same flag/semantics as in
`run_md_oc25.py`):

```bash
python smoke_test_model.py --state-id <some_state_id>
python smoke_test_model.py --state-id <some_state_id> --prerelax --prerelax-steps 200
```

Expected output: total energy, energy/atom, max|F| over all atoms, and
max|F| restricted to the free (non-FixAtoms) atoms.

## Assumptions about the fairchem API (see also run_md_oc25.py's docstring)

```python
from fairchem.core import pretrained_mlip, FAIRChemCalculator
predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-oc25", device="cuda")
calc = FAIRChemCalculator(predictor)
```

- No `task_name` is required for the OC25 checkpoints per the HuggingFace
  model card. `--task-name` is exposed as a CLI flag on `run_md_oc25.py` and
  `smoke_test_model.py` anyway (default `None`); if the installed fairchem
  version's `get_predict_unit()`/`FAIRChemCalculator()` raise `TypeError`
  without it, both scripts retry once with `task_name=<value>` passed
  through, and otherwise let the real exception propagate.
- Use `esen-sm-conserving-all-oc25` (energy-conserving, autograd forces) for
  all production MD. Do **not** use `esen-md-direct-all-oc25` for MD (direct-force
  models are not energy-conserving and bias canonical sampling) -- it exists in the repo but no script here defaults to it.
- None of the above has been verified against a live install. Confirm on
  the first real cluster run and update this file / the docstring if the
  actual API differs (e.g. if `device` is not a valid kwarg, or if the
  predictor/calculator split works differently in the installed version).
