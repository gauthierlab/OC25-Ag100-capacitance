# S0 structure set — Ag(100)/water/NaF(CsF) for OC25 capacitance smoke test

Built 2026-09-17 at the RPBE-D3 zero-damping lattice constant (see below). This set replaces
an earlier 49-state set built at a = 3.8436 Å (BJ-damped D3, wrong damping for the OC25 recipe)
that also contained 3-layer slabs; both the 3-layer slabs and the 8-layer DFT extension were
dropped the same day, so this set is 4-layer slabs only.

## Regenerate

From `ROOT/s0/`:

```
python3 build_s0_structures.py
```

Useful flags: `--dry-run` (print the state matrix and exit), `--only '<regex>'`
(filter state_id, e.g. `--only 'L4_n8_CsF_.*'`), `--a <value>` (override the
derived lattice constant), `--contcar <path>`, `--film`, `--vacuum`, `--conc`,
`--outdir`.

This last run: 29/29 states, all passed internal validation, using the pure-Python
fallback packer (packmol not found on PATH).

## Lattice constant

`a` is derived from `../ag_rpbe_d3zero_CONTCAR` (1-atom fcc primitive cell, VASP,
RPBE + D3 zero damping = IVDW 11, ENCUT 500, 12x12x12, ISIF 3, residual pressure 0.7 kbar):
d_NN = |v1| = 2.942761 A, a = sqrt(2)*d_NN = **4.161692 A** (1.9 % above experiment,
4.086 A; plain RPBE gives 4.215 A). IVDW 11 is the OC25 training setting, and Meta
confirmed the OC25 bulks were relaxed with RPBE-D3, so the training Ag slabs are at the
same constant. The script never hard-codes a lattice constant; `--a` overrides it and
everything downstream (area, sigma, water/ion counts) rescales.

## State matrix

| Group | Electrolyte | n | layers | delta_N | replicas | ion_start (charged) | ion_start (dN=0) | Purpose |
|---|---|---|---|---|---|---|---|---|
| Interface | NaF | 8 | 4 | -2,-1,0,+1,+2 | 4 | r1,r2=surface; r3,r4=far | random | quick-look C at PZC / two-sided start / ionization check |
| Interface | CsF | 8 | 4 | -2,0,+2 | 2 | r1=surface; r2=far | random | in-distribution cation comparison: Cs+ is in the OC25 training ion set, Na+ is not |
| Bulk | H2O | — | — | 0 | 2 (`bulk_H2O_r1/r2`) | — | MLIP bulk-water sanity (density/RDF/diffusion) |
| Bulk | NaF (~1 M) | — | — | 0 | 1 (`bulk_NaF_r1`) | — | ion pairing / hydration sanity |

Totals: 20 NaF interface states + 6 CsF interface states + 2 bulk-water + 1
bulk-NaF = **29 states**.

`state_id` pattern: `L{layers}_n{n}_{electrolyte}_dN{+/-k}_{start}_r{replica}`,
e.g. `L4_n8_NaF_dN-2_far_r3`; bulk boxes: `bulk_H2O_r1`, `bulk_NaF_r1`.

Key derived numbers (n=8, a=4.161692 A):
- Cell 23.542 x 23.542 x 58.743 A (4-layer slab, top Ag layer at z = 6.24 A, 35 A water
  film to z = 43.74 A, 15 A vacuum). Lateral area = 554.23 A^2.
- sigma per single excess elementary charge = 2.891 uC/cm^2 (so |delta_N|=2 -> +/-5.782 uC/cm^2).
- n_water = 636, n_pairs (NaF or CsF at 1 M) = 12 per interface state; n_cation/n_anion
  = 12 + max(0,-delta_N) / 12 + max(0,+delta_N).
- n_Ag = 256 (bottom 2 layers fixed); n_atoms = 2188 (dN=0), 2189 (|dN|=1), 2190 (|dN|=2).
- Bulk boxes: cubic side 18.644 A (216 waters at 0.997 g/cm^3); bulk_NaF adds
  4 Na + 4 F (~1 M) in the same box.

Water displacement rule: one water molecule is removed from the target count
per ion pair (`n_water = n_water_target - n_pairs`), treating one ion pair as
displacing one water molecule's volume.

## packmol vs. fallback packer

packmol was not installed on the build machine. The script always writes a
`packmol.inp` per state (referencing the shared rigid-body templates in
`structures/templates/{water,na,cs,f}.pdb`) so the exact same fill can be
regenerated with real packmol elsewhere. When packmol is unavailable, a
built-in pure-Python rejection-sampling packer is used instead: rigid water
monomers (O-H = 1.0 A, H-O-H = 109.47 deg) and single-atom ions are inserted
at random positions/orientations, rejecting any placement that puts an atom
within 2.0 A of an existing atom (periodic minimum image in x,y for the
slab/film states, full 3-D periodic minimum image for the bulk boxes).
Rejection sampling can jam in the last 1-2% of a fill (classic random
sequential addition tail behavior); the script detects this
(`PackingStuckError`) and restarts the *entire* fill for that state with a
fresh deterministic seed and a larger per-molecule attempt budget (up to 6
retries). All 29 states converged. Each manifest entry records which engine
was actually used (`packer: "packmol" | "python_fallback"`).
