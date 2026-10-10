# S0 shared file conventions (all scripts must follow this)

Root: `ag_100/s0/`

## Structures (written by build_s0_structures.py)
- `s0/structures/<state_id>/init.extxyz` : ASE extended-XYZ, full cell + pbc=[T,T,T], with
  per-atom column `fixed` (1 = held at bulk position during MD, 0 = free) and `role`
  (one of: slab, water, cation, anion). Constraint also stored as ASE FixAtoms in info if possible.
- `s0/structures/<state_id>/POSCAR` : same structure, VASP5 format, selective dynamics (F F F for fixed).
- `s0/structures/<state_id>/packmol.inp` : the packmol input used (or that would be used) for this state.
- `s0/structures/manifest.json` : list of dicts, one per state:
  {state_id, slab_layers, n_lateral, electrolyte ("NaF"|"CsF"|"H2O"|"NaF_bulk"), delta_N (int),
   n_cation, n_anion, n_water, n_Ag, n_atoms, ion_start ("surface"|"far"|"random"|null),
   replica (int), lattice_constant_A, area_A2, sigma_nominal_uC_cm2, cell (3x3 list), z_slab_top_A,
   z_film_top_A, purpose (free text)}

## state_id pattern
`L{layers}_n{n}_{electrolyte}_dN{+/-k}_{start}_r{replica}` e.g. `L4_n8_NaF_dN-2_far_r3`;
bulk boxes: `bulk_H2O_r1`, `bulk_NaF_r1`.

## MD outputs (written by run_md_oc25.py) under `s0/md/<state_id>/`
- `traj.extxyz` (every 1 ps, appended), `thermo.csv` (step, time_ps, T, Epot, Ekin), `topology.csv`
  (per written frame: n_HF, n_OH-, n_H3O+, n_H_on_Ag, n_F_contact, n_cation_contact, n_water_dissociated),
  `snapshots/frame_XXXXX.extxyz` (DFT candidates, from production part), `restart.extxyz`, `run.log`.

## DFT inputs (written by make_vasp_inputs.py) under `s0/dft/<state_id>/<frame>/native/` (the `ext8/` variant is no longer produced as of 2026-09-17)
- POSCAR, INCAR, KPOINTS, `meta.json` (state_id, frame, slab_layers_dft, area, n_ions, delta_N).
