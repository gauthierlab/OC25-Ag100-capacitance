# S0 smoke tests: Ag(100) / water / NaF with the OC25 MLIP

Purpose: before any production run, establish that composition charging of Ag(100)/NaF with the OC25 potential
(`esen-sm-conserving-all-oc25`) produces ionized Na⁺/F⁻ with the balancing charge on the metal, that the water film is liquid and bulk-like in its middle, and that Na⁺
(not an OC25 training ion) behaves like Cs⁺ (which is). Full context: `../Ag100_OC25_capacitance_plan.md` §6;
label legend `../TESTS.md`; file conventions every script obeys: `INTERFACE_SPEC.md`.

Contents: 29 ready-to-run initial structures (`structures/`), an MD driver with a per-frame reactive-chemistry
monitor, an analysis script, a VASP single-point generator and analyzer for the potential/charge read-out, and
SLURM templates. Everything cluster-specific lives in one file, `slurm/cluster.env`.

## Runbook (run every command from this directory, `s0/`)

Step 0. Environment, once. Create the conda env per `ENVIRONMENT.md`, get access to the gated HuggingFace repo
`facebook/OC25`, then:
```bash
cp slurm/cluster.env.example slurm/cluster.env   # and edit every value in it
python smoke_test_model.py --state-id L4_n8_NaF_dN0_random_r1 --prerelax   # on a GPU node
```
Success: prints total energy, energy/atom, and max |F| after pre-relaxation below ~1 eV/Å. If the fairchem import
or model load fails, see the API notes at the end of `ENVIRONMENT.md`.

Step 1. Structures are already built (29 states, 4-layer slabs only since 2026-09-17, manifest in `structures/manifest.json`). Rebuild only if you
change the lattice constant, film, or concentration:
```bash
python build_s0_structures.py            # ~8 min; writes structures/<state_id>/{init.extxyz,POSCAR,packmol.inp}
```
The lattice constant (a = 4.1617 Å) is read from `../ag_rpbe_d3zero_CONTCAR`, the RPBE-D3 zero-damping (IVDW = 11) value matching the OC25 recipe. The earlier 3.8436 Å build used the BJ-damped (IVDW = 12) constant by mistake; that CONTCAR is kept as `../ag_rpbe_d3bj_IVDW12_CONTCAR_superseded` and the old structures as `structures_a3.844_superseded/`.

Step 2. MD, one GPU per state, 100 ps equilibration + 400 ps production at 298 K:
```bash
python make_state_list.py                # writes structures/state_list.txt (29 lines)
sbatch --partition=<gpu partition> --gres=gpu:1 --time=24:00:00 \
       --array=1-$(wc -l < structures/state_list.txt)%16 slurm/submit_md_array.sh
```
Success per state: `md/<state_id>/run.log` ends with the production phase complete, `md/<state_id>/snapshots/`
holds 8 frames (every 50 ps of production), `topology.csv` shows no WARNING lines in `run.log`. `run.log` prints
ns/day after the first picosecond; set `--time` from that (500 ps per state). Re-submitting the same array is safe:
finished states exit immediately and unfinished ones resume from their last 10 ps checkpoint.

Step 3. MD analysis:
```bash
python analyze_md.py --all --md-dir md --structures-dir structures
```
Outputs: `md/summary.csv` (one row per state), `md/<state_id>/analysis/` (profiles, PNGs), `md/_comparisons/`
(surface-start vs far-start convergence per charged state). What to look at is listed in `README_MD.md`.

Step 4. DFT inputs on the last 5 production snapshots of every state (4-layer slab at native thickness; the
3-layer and 8-layer-extended variants were dropped 2026-09-17):
```bash
python make_vasp_inputs.py --all --frames last:5
```
This writes `dft/jobs.txt` (one job directory per line).

Step 5. DFT array (VASP 6.3.2, RPBE-D3, Γ-only, ~2190–2450 atoms per cell, then Bader):
```bash
sbatch --partition=<cpu partition> --nodes=1 --ntasks-per-node=<cores> --time=06:00:00 \
       --array=1-$(wc -l < dft/jobs.txt)%32 slurm/submit_dft_array.sh
```
Success per job: `OUTCAR` contains `E-fermi`, `LOCPOT`, `CHGCAR`, and `ACF.dat` exist. Re-submission skips
finished jobs.

Step 6. Potential and charge read-out, then the go/no-go table:
```bash
python analyze_potentiometer.py --all
python summarize_s0_dft.py               # writes dft/s0_dft_summary.md
```
Go/no-go criteria and the meaning of every column are in `README_DFT.md`.

## Directory map

| Path | What it is |
|---|---|
| `INTERFACE_SPEC.md` | file names, per-atom arrays, manifest schema, state_id pattern |
| `build_s0_structures.py`, `structures/` | structure generator, 29 states, `manifest.json`, packmol inputs, `README.md` with the state matrix |
| `run_md_oc25.py`, `md_common.py`, `dummy_calculator.py` | MD driver, shared helpers (FixAtoms handling, topology classifier), toy calculator for offline tests |
| `analyze_md.py`, `make_state_list.py`, `smoke_test_model.py` | MD analysis, state list for the array, model load test |
| `make_vasp_inputs.py`, `analyze_potentiometer.py`, `summarize_s0_dft.py` | DFT input generation, per-job read-out, S0 summary |
| `slurm/cluster.env.example`, `slurm/submit_*.sh` | the one file to edit, and the three sbatch scripts that source it |
| `ENVIRONMENT.md`, `README_MD.md`, `README_DFT.md` | install notes, MD stage details, DFT stage details |
