# S0 DFT potentiometer

## Purpose

OC25 (Meta FAIR MLIP) samples Ag(100)/water/NaF(or CsF) interfaces at fixed composition, but
gives no electrode potential and no guarantee that the intended nominal charge (set by the
excess Na/F ion count, delta_N) is the charge actually realized on the metal. This directory
runs RPBE-D3 single points on MD snapshots to read out, per snapshot:

1. **U_inner** (primary): the inner-potential-referenced electrode potential,
   `U_inner = <V_H>_bulk-water - E_fermi`, both terms read directly off VASP's own eV scale
   (OUTCAR E-fermi; LOCPOT with `LVHAR = .TRUE.`, i.e. Hartree+ionic electrostatic potential
   only, planar-averaged over x,y and then macro-averaged over a `--macro-window` along z),
   averaged over a `--ref-width`-thick slab centered in the middle of the water film. The
   standard deviation of the macro-averaged V_H across that window is reported as the
   reference noise (`U_inner_ref_noise_V`).
2. **Phi_top / Phi_bottom** (secondary): the vacuum work function (`Phi = V_H(vacuum plateau)
   - E_fermi`). The slabs are single-sided, so each cell has exactly ONE vacuum span. The MD
   cells put the fixed bottom Ag layer at z = 0; `make_vasp_inputs.py` translates every
   snapshot along z so the ~14 A vacuum is split evenly across the cell boundary (the effect of
   `atoms.center(axis=2)`), which puts the `DIPOL = 0.5 0.5 0.5` correction step mid-vacuum.
   Without centering the step cuts through the bottom Ag layer; this is the likely cause of the
   SCF non-convergence Meta reported on 2026-09-29 (not yet confirmed by a centered rerun). `Phi_top` is read on the near-film side of the step and
   `Phi_bottom` on the side facing the bottom Ag layer, 3 A clear of the atoms and 1.5 A clear
   of the step; each plateau is ~2.5 A wide and its slope (should be ~0) is reported.
   `dipole_step = Phi_top - Phi_bottom` equals the correction step.
3. **sigma, three independent estimators** of the actual electrode charge (uC/cm^2):
   - `sigma_a` (ion-Bader): infers the metal's charge from overall charge neutrality given the
     net ionic (Bader) charge on the Na/Cs/F ions, treating water as neutral.
   - `sigma_b` (slab-Bader): direct sum of Bader ionic charge over the Ag atoms.
   - `sigma_c` (density integral): nuclear minus electronic charge from the cell bottom up to
     the electron-density minimum between the top Ag layer and the first water layer.

Sign convention (all three estimators, and both potentials, are built to agree with this):
**sigma is negative when the metal has excess electrons** (e.g. an ionized excess Na atom,
Na -> Na+ + e-, pushes an electron onto the metal and makes sigma negative).

## Commands

```bash
cd s0

# 1a. Native-thickness inputs: last 5 production snapshots of every state.
python3 make_vasp_inputs.py --all --frames last:5
# (The 8-layer slab-extension variant, --ext8, was dropped from S0 on 2026-09-17 along with the
#  3-layer slabs: S0 is 4-layer slabs at native thickness only. The code path is retained but unused.)

# 2. Submit the SLURM array (slurm/cluster.env supplies the VASP module, POTCAR path, mpirun, bader)
sbatch --partition=<p> --nodes=1 --ntasks-per-node=<cores> --time=06:00:00 \
       --array=1-$(wc -l < dft/jobs.txt)%32 slurm/submit_dft_array.sh

# 3. Analyze every finished job (parses OUTCAR/LOCPOT/CHGCAR/ACF.dat, writes
#    <job-dir>/potentiometer.json and dft/potentiometer_summary.csv)
python3 analyze_potentiometer.py --all --ref-width 8.0 --macro-window 3.0

# 4. Summarize: group by (slab_layers_md, slab_layers_dft, electrolyte, delta_N); quick-look
#    capacitance at the PZC. (Slab-thickness comparisons dropped 2026-09-17.)
python3 summarize_s0_dft.py
```

A single job can be inspected directly:
```bash
python3 analyze_potentiometer.py --job-dir dft/<state_id>/<frame>/native --mlip-forces md/<state_id>/snapshots/<frame>.extxyz
```

## Output quantities and sign conventions

| Quantity | Definition | Units | Sign |
|---|---|---|---|
| `U_inner_V` | `<V_H>_bulk-water(macro-avg) - E_fermi` | V | more negative = more negatively charged electrode (cathodic) |
| `U_inner_ref_noise_V` | std of macro-averaged V_H across the `--ref-width` bulk-water window | V | reference-region electrostatic noise, not a fit error |
| `Phi_top_eV`, `Phi_bottom_eV` | vacuum work function on each side, `V_H(plateau) - E_fermi` | eV | absolute (SHE-referenceable via Phi_SHE = 4.44 eV) |
| `dipole_step_eV` | `Phi_top - Phi_bottom` | eV | should equal the IDIPOL/LDIPOL correction step; a large value flags an under-converged dipole correction |
| `sigma_a/b/c_uC_cm2` | see above | uC/cm^2 | negative = excess electrons on the metal |
| `min_ion_ionization` | min(\|Zval - N_bader\|) over all cation/anion ions | e | how close the least-ionized ion is to a full formal charge |
| `n_F_contact` / `F_contact_charge_mean` | count and mean signed ionic charge of F within 2.9 A of a top-layer Ag | (count), e | F_contact charge near -1 e = still essentially ionic F- even while contact-adsorbed |
| `force_MAE_if_available` | MLIP-vs-DFT force MAE, only if `--mlip-forces` given | eV/A | free MLIP-validation byproduct |

**Formulas.** `sigma_b = +(sum_Ag(Zval - N_bader)) * e / A` (the slab's own net Bader charge;
an Ag atom that gained an electron contributes negatively). `sigma_a = -(sum_ions q_Bader) * e / A`
(the metal charge inferred from the ions by overall neutrality, hence the minus sign). Both were
verified against a synthetic test case with one known excess electron on the slab.

**`sigma_c`**: reference plane is the electron-density minimum between the top Ag layer and
the first water O layer; the nuclear charge is the sum of ZVAL (point charge at each atom's
z-position, no smearing) for atoms below that plane; electron count comes from VASP's own
CHGCAR convention (grid value = rho(r)*V_cell, so cumulative sum of the planar average divided
by NGZ gives the electron count below a given z). `sigma_c = (nuclear - electronic) * e / A`,
same sign convention as a/b.

## Go/no-go criteria for S0

1. **Ionization**: `sigma_a / sigma_nominal >= 0.9` (the excess ions are behaving as
   near-fully-ionized Na+/F-, not staying as neutral adatoms/molecules).
2. **Contact-adsorbed F retains >= 0.7 e-** of its formal -1 charge (`F_contact_charge_mean`
   more negative than -0.7, i.e. `|F_contact_charge_mean| >= 0.7`), i.e. contact adsorption is
   not accompanied by spurious (near-)discharge to neutral F/AgF.
3. **Quick-look capacitance at the PZC within 40-100 uF/cm^2** (per true, not geometric, area)
   from the linear U_inner-vs-sigma_a fit in `summarize_s0_dft.py`.

`summarize_s0_dft.py` prints and writes the table for check 3 automatically; checks 1-2 are
per-job fields in `potentiometer_summary.csv` (aggregate them with a spreadsheet/pandas
groupby, or extend `summarize_s0_dft.py` -- not done here since the plan's S0 states were not
yet populated when this was written).

## Why k-points are Gamma-only

E_F convergence at the 8x8 (~22 A) lateral cell was verified separately (Gamma-only vs a denser mesh
agree), so every S0 job is Gamma-only.

## INCAR settings and the LASPH decision

See the INCAR template in `make_vasp_inputs.py` (`INCAR_TEMPLATE`). The OC25 training flags are kept
verbatim (GGA=RP, IVDW=11, ENCUT=400, ISMEAR=0, SIGMA=0.1, ISPIN=1, IDIPOL=3, LDIPOL); EDIFF is tightened
to 1e-5; LVHAR, LCHARG, LAECHG are added for the read-out. LASPH stays at the VASP default (off) because it
is not part of the OC25 recipe; revisit only if a direct comparison shows E_F or sigma shifting by more
than the per-snapshot noise floor.

## Known limitations / left undone

- `analyze_potentiometer.py`'s `--mlip-forces` comparison assumes the DFT and MLIP structures
  have identical atom ordering (true for `native`, but note MLIP forces are not defined for
  the `ext8` variant's added bulk layers, so `--mlip-forces` should only be used with `native`
  jobs).
- The bottom-side vacuum plateau (`Phi_bottom`) is identified via the periodic wraparound of
  the single vacuum gap above an asymmetric slab; if a future state uses a genuinely
  double-sided vacuum cell, revisit `compute_potentials` in `analyze_potentiometer.py`.
- `summarize_s0_dft.py` does not aggregate the per-job ionization/F-contact go/no-go checks
  (1-2 above) into its markdown table; these are per-job columns in `potentiometer_summary.csv`
  and can be aggregated with a pandas groupby on (electrolyte, delta_N).
- POTCARs are never shipped (license); `submit_dft_array.sh` concatenates them from
  `VASP_PP_PATH` using `POTCAR_ORDER.txt` at run time. `VASP_PP_PATH`, the VASP module, the
  mpirun/srun invocation, and the bader/chgsum binaries are set in `slurm/cluster.env`; the
  partition, node count and walltime go on the `sbatch` command line.
