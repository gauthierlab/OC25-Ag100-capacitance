# Follow-up: S0 diagnostics, new cell sets, and long-range tests (Ag(100)/OC25)

**From:** Gauthier group (TTU) · **Date:** 2026-10-08 · **Re:** `RESPONSE_S0_potential_diagnostics_2026-10-05.md`

Thank you for the thorough response. Every requested item came back, and the extra 196-snapshot PZC set made the
cation offset unambiguous. This note has six parts:

1. A correction to the premise of our 2026-09-30 request, which changes how the scatter should be read.
2. Seven requests on data you already have (R1–R7). None needs new DFT.
3. Questions about predicting the work function with an OC25 head, and a validation step for it.
4. Three new structure sets, shipped in `ag100_followup_2026-10-08.tar.gz`:
   - **LC**, which can run now: MD plus ~100 DFT single points.
   - **SL**, the production design, to run if a work-function head exists.
   - **LR1**, MD only.
5. A battery of six long-range tests (LR1–LR6). Together they turn the capacitance calculation into a
   demonstration that OC25 captures long-range electrostatics.
6. The decisions we need from you.

Labels used throughout:

- **σ:** surface charge density on the metal.
- **ΔN:** ion imbalance; ΔN < 0 means excess cations, so a negative metal.
- **U_inner:** the Hartree potential averaged over an 8 Å window mid-film, minus E_F.
- **Φ_top:** the vacuum level above the water minus E_F.
- **A:** the cell area; **L:** the distance from the metal to the potential reference.

---

## 1. Correction: the per-snapshot scatter is the thermal value

Our request claimed the U_inner scatter was "about 10× larger than physics allows", using
⟨δU²⟩ = kT/(C·A). That relation is wrong for a cell with a fixed metal charge, and we apologise for the detour it
caused. At fixed charge, the polarization P of the slab between the metal and the reference fluctuates freely.
For a uniform polarization mode in volume V = A·L, the free energy at fixed displacement field D is
V[(D − P)²/2ε₀ + P²/2ε₀χ], which gives

  ⟨δP²⟩ = kT·ε₀·χ / (V(1 + χ)),  so  var(U) = kT·L/(ε₀A) · (1 − 1/ε).

Two consequences follow:

- **The formula is a ceiling.** var(U) ≤ kT·L/(ε₀A) for any susceptibility. Water alone (ε ≈ 78) already puts the
  factor at 0.99, so electrolyte screening of any length cannot raise the variance above it. Ionic screening drives
  the factor to exactly 1. Screening suppresses the *mean* response to a perturbation, not the thermal fluctuation
  of the uniform field.
- **NaF sits at the ceiling.** With no fitted parameter:

| Quantity (A = 554 Å², 298 K, film ≈ 40 Å) | Ceiling | NaF | CsF |
|---|---|---|---|
| SD of V_H − E_F, window at ¼ / ½ / ¾ of the film (L ≈ 10 / 20 / 30 Å) | 0.29 / 0.41 / 0.50 V | 0.22 / 0.37 / 0.49 V | 0.30 / 0.57 / 0.72 V |
| SD of Φ_top (L ≈ whole film) | 0.54–0.58 V | 0.54 V | 0.87 V |
| SD of the screened cell dipole | 1.7–1.8 e·Å | 1.7 e·Å | 2.9 e·Å |
| SD of the film field over ~24 Å | ~19 mV/Å | 19–21 mV/Å | 24–29 mV/Å |

The per-snapshot scatter is therefore not, by itself, evidence of under-screening, and a long-range MLIP would not
reduce the NaF scatter. Two things follow for the CsF result and for the averages:

- **CsF exceeds the ceiling** (0.87 vs 0.58 V; ~2.3× in variance). In a correctly sampled ensemble that is
  impossible for any screening length. CsF would need L ≈ 90 Å, about twice the film. This is the one fluctuation
  result that needs an explanation. Possible causes are slow sampling, correlated snapshots, or a genuine model
  effect; R1 and R2 below distinguish them.
- **The averages are what can carry an ensemble error:** the mean film field, the 1.6 V NaF/CsF gap, and the slow
  ionic-dipole relaxation.

**The mean field cannot be a diffuse-layer tail at 1 M.** The Debye length is 3.0 Å, and your field fit starts
~4–5 Å beyond the first cation layer. At ΔN = 0 the diffuse layer only balances specifically adsorbed F⁻
(0.70 per cell NaF, 0.33 CsF). Gouy–Chapman then gives ≤ 3 mV/Å at the start of the diffuse layer, ≤ 0.7 mV/Å
where the fit starts, and a total drop of ≤ 9 mV. Even 2 e per cell gives only 24 mV. Producing ~0.5 V would need
a ~170 Å screening length. That said, with the effective n equal to the number of replicas, the means are
−23 ± 10 mV/Å (NaF) and +20 ± 14 mV/Å (CsF), only ~2σ and ~1.4σ. R1 and R5 settle whether they are real.

**On the cation gap.** We agree that the double layers differ: 1.11 vs 0.51 first-layer cations and 0.70 vs 0.33
contact F⁻ at ΔN = 0. That is a legitimate cation-specific seed. Under-screening, though, would act on both salts
alike, so the gap's *size* (1.6 V against an experimental cation effect of tens of mV) is the question. We would
not yet conclude either "Na⁺ chemistry error" or "not a Na⁺ error". The LC set below is designed to separate
background-salt effects from counter-ion effects.

---

## 2. Requests on existing data (no new DFT)

Please give per-state results for both the S0 and the PZC sets where applicable.

| # | Request | What it decides |
|---|---|---|
| R1 | The mean film field **per replica** with its SE, and the film field vs simulation time within each replica (MD frames, classical Poisson of formal ion charges + SPC/E water is fine if DFT is too sparse) | Whether the mean field is significant, and whether it is drift (not equilibrated) or stationary |
| R2 | Autocorrelation functions and integrated times of p_ion, p_water and p_ion + p_water from the 401-frame MD, per state. Also the same for Φ_top across the 30 fs short-spacing frames, if you extend that test to ~10 ps | Correct long-range relaxation gives the Maxwell time ε·ε₀/κ ≈ 0.1 ns at 1 M; ion diffusion across the film gives ~1 ns. This also sets every MD budget below, and is the most important number in this list |
| R3 | A control-variate slope: U_adj = U − c·(X − ⟨X⟩_state,MD), with X = p_ion + p_water and ⟨X⟩ averaged over all MD frames of the state. Report dU_adj/dσ ± SE for NaF and CsF | How much variance reduction is available now, and a first check that any bias is constant in σ |
| R4 | Bader charges of **contact** F⁻ (within the 3.3 Å Ag–F first minimum), separately from the others | Plan criterion: contact F⁻ keeps ≥ 0.7 e⁻ |
| R5 | The snapshot-**averaged** macro V_H(z) per state (from the DFT LOCPOTs), fitted to a + b·z + c·exp(−z/λ) from the cation layer to 8 Å below the surface. Also the **mean** (not only the SD) of V_H − E_F at ¼, ½ and ¾ of the film height | Diffuse tail (λ ≈ 3–5 Å, flat middle) vs a linear unscreened field |
| R6 | Net ionic charge density ρ_ion(z) and the water polarization density P_z(z), MD-averaged per state, with their classical Poisson potential | Whether a mean field coexists with ρ ≈ 0 mid-film. This also gives the classical baseline for R3 and LR5 |
| R7 | **RDFs Ag–O (water), Ag–Na, Ag–Cs, Ag–F** per state from the MD (26 states × 401 frames). Use top-layer Ag only, normalized as a surface pseudo-g(r) with the shell volume truncated to the film half-space. Please send the full curves as data files, the first-shell minimum, the coordination number to it, and the z-resolved density of each species | Cation-specific packing. Ag–O (first-layer water density and its distance) has not been reported yet |

---

## 3. A work-function head (questions, then validation)

Plan v2 §2 quotes your outlook that "it would be highly desirable to have ML models that can predict the interface
workfunctions directly". If such a head exists or is planned, it changes the whole design. The potential becomes
available on every MD frame, DFT shrinks to validation, and the thermal noise can be averaged as in Shin et al.,
Nat. Commun. 13, 174 (2022). Questions:

1. Does an OC25 head exist, or is one planned, that predicts Φ (or E_F against the vacuum, or the cell dipole) per
   frame?
2. Is it a read-out head (forces unchanged) or a model with electrostatics in the energy and forces?
3. What does it predict, and how is it pooled? A sum of per-atom charges or dipoles extrapolates to thick films;
   a mean-pooled invariant read-out has no handle on the z-integral that sets the drop across a 40 Å film.
4. Which data was it trained on? The OC25 films are 5–10 Å thick, and ours are 35–60 Å.
5. Throughput: `run.log` prints ns/day. Please send the measured S0 values (2190 atoms) and GPU type, plus an
   estimate at ~10k and ~22k atoms. Our cost tables below assume 2×10⁴–10⁵ atom-steps/s per H200, which is unverified.

**Validation (W1), once a head exists.** Run it on all 326 existing DFT snapshots and report:

- RMSE and r of Φ_pred vs Φ_DFT;
- the **mean residual per (salt, ΔN)**. Pass: the residual slope vs σ stays within ±1 mV per µC/cm², with no
  dependence on the cation;
- whether it reproduces the 1.6 V NaF/CsF gap and the constant bottom-face work function;
- the comparison with the free baseline of the screened nuclear dipole (p_ion + p_water)/ε_inf, which already
  explains 64–80 % of the dipole variance.

Random per-frame error is harmless, because it averages out with the thermal noise; σ-dependent bias is what
distorts C. The LC DFT points (§4.1) add 100 more validation snapshots at compositions the head has not seen.

---

## 4. New structure sets

All three sets were built with the updated `s0/build_s0_structures.py` (new `--matrix lc|sl|lr1` option) and passed
validation. Each set has its own `manifest.json` and `state_list.txt`. Lattice constant, slab, MD and DFT
settings are unchanged from S0, except as stated. Vacuum is now 20 Å, because the S0 films thickened and left only
8–12 Å. The counter-ion sets have **no background salt**:

- ΔN < 0 adds |ΔN| cations;
- ΔN > 0 adds ΔN F⁻;
- ΔN = 0 is pure water.

The F⁻ side is the same for both salts, so it is built once.

### 4.1 LC: same 8×8 cell, counter-ions only, MD + DFT (please run now)

`s0/structures_lc/`. 20 states, 2200–2202 atoms, 35 Å film. State IDs are `LC_n8_<ion>_dN<±k>_<start>_r<k>`.

| Composition | ΔN | Replicas | Purpose |
|---|---|---|---|
| Pure water | 0 | 4 | PZC and fluctuation ceiling of bare Ag(100)/water. With no ions, there can be no cation gap |
| F⁻ only | +1, +2 | 4 (2 surface-start, 2 far-start) | Anodic side, shared by both salts |
| Na⁺ only | −1, −2 | 2 (surface, far) | Cathodic side |
| Cs⁺ only | −1, −2 | 2 (surface, far) | Cathodic side, in-distribution cation |

**MD.**

```bash
python make_state_list.py --structures-dir structures_lc
python run_md_oc25.py --state-id <id> --structures-dir structures_lc --out-dir md_lc \
       --equil-ps 200 --prod-ps 1000 --snapshot-every-ps 200
```

The run is longer than S0 (1 ns instead of 0.4 ns of production), because single counter-ions need time to sample
the film.

**DFT** on the 5 production snapshots (200 ps apart), about 100 single points:

```bash
python make_vasp_inputs.py --structures-dir structures_lc --md-dir md_lc --out-dir dft_lc \
       --all --frames last:5
```

Please keep `LVHAR`, `LAECHG` and the Bader workflow as in S0, and the patched `analyze_potentiometer.py`.

**Predictions** (stated now so the result is interpretable):

- **The scatter will stay at the ceiling** (~0.4 V U_inner, ~0.55 V Φ_top), because the ceiling does not depend on
  salt. If pure water exceeds it, OC25 water is under-penalized. If only the Cs⁺ cells exceed it, the excess is
  Cs-specific.
- **PZC.** The pure-water cell gives one PZC with no cation dependence possible. Compare it with experiment:
  −0.62 V vs SHE, Valette 1982, which is the Ag(100) PZC without specific adsorption, −0.865 V vs SCE.
- **Extrapolate each salt's U(σ) to σ = 0.** If Na⁺ and Cs⁺ both land on the pure-water PZC, the 1.6 V gap at 1 M
  is a background-salt effect. If they do not, the counter-ion placement itself differs.

### 4.2 SL: Shin-like production cells (run only if a work-function head exists and passes W1)

The 0.1 M double hump on Ag(100) is a ~4 µF/cm² (~8 %) shoulder in Valette's data. Resolving it from U(σ) needs
~1 mV per state, which no sampling budget reaches. The features are large only at ≤ 0.01 M, and a direct 0.01 M
cell needs a ~200 Å film and 160–250k atoms. We therefore follow Shin et al., who reproduced the camel shape:

- salt-free counter-ion cells of similar area;
- ~15 ns of sampling per σ point;
- a smooth fit of σ–E;
- comparison against the dilute experiment.

`s0/structures_sl_h40/` (62 states) and `s0/structures_sl_h60/` (12 states, film-thickness check).

| | |
|---|---|
| Cell | n = 16 (47.1 Å, 2218 Å²; Shin: 2143 Å²), 4 layers, 40 Å film, 20 Å vacuum, ~9.9k atoms |
| ΔN | 0, ±1, ±2, ±3, ±4, ±6, ±8, ±12, ±16, ±20, ±25 (0.72 µC/cm² per ion, up to ±18 µC/cm²). ΔN = ±4 and ±8 match the σ of LC ΔN = ±1 and ±2 (test LR6) |
| States | pure water ×2; F⁻ ×10 ΔN × 2; Na⁺ and Cs⁺ ×10 ΔN × 2 each. 62 trajectories |
| Thickness check | 60 Å film at ΔN = +2, +4 (F⁻) and −2, −4 (Na⁺ and Cs⁺), 2 replicas each: 12 trajectories |
| MD | 1 ns equilibration, then production to ~15 ns total per state across replicas; write every 1 ps; work-function head on every written frame. Set the final length from R2 |
| DFT | None on SL cells. Validation comes from the LC set at n = 8 and from W1 |
| Cost (assumed throughput) | 0.17–0.87 ns/day per GPU at ~9.9k atoms, i.e. 17–86 GPU-days per state, all states in parallel. To trim, drop ΔN < 0 for Cs⁺ or drop ±20/±25 |

**Analysis** (ours):

1. Fit a smooth spline to the mean U(σ), with a Maxwell construction if an S-shaped region appears.
2. Extract the inner-layer capacitance C'(σ) by subtracting the counter-ion contribution, computed by Poisson from
   the measured ion profile.
3. Add Gouy–Chapman at Valette's five concentrations (0.005–0.1 M NaF) and compare with all of his Fig. 2.
4. Show the direct salt-free C(E) against 0.005 M as Shin did, with the caveat that a salt-free cell is not a
   fixed-concentration cell.

### 4.3 LR1 bulk boxes (MD only, run when convenient)

`s0/structures_lr1/`: 60 Å cubic boxes of pure water, 1 M NaF and 1 M CsF (~21.7k atoms each). Suggested run:
0.2 ns equilibration and 2 ns production, writing every 0.5 ps. Needed for test LR1.

---

## 5. Long-range tests: C as a long-range observable

Your PMFs out to 5× the 6 Å cutoff test the screened pair interaction, and we take them as evidence of correct
long-range behaviour. In water, though, that interaction is only l_B/r ≈ 7.1/30 ≈ 0.24 kT at 30 Å, so the signal
is small. The tests below use collective quantities in which many charges add coherently. A charged electrode's
field does not decay with distance, so the electrode–counter-ion attraction is long-ranged in a way a pair PMF is
not. Passing them is a stronger demonstration, and all of them except LR1 come from the same trajectories as C.

| # | Test | Reference answer | Failure signature | Data |
|---|---|---|---|---|
| LR1 | Bulk perfect screening (Stillinger–Lovett): S_ZZ(k)/k² → 1/κ_D² as k → 0. Water's longitudinal dielectric response ε_L(k → 0) ≈ 78 | Exact sum rules, independent of the model (κ_D⁻¹ = 3.0 Å at 1 M, k_min = 0.105 Å⁻¹) | Too little small-k screening | LR1 boxes |
| LR2 | Interfacial fluctuation ceiling: var(U) at fixed σ ≤ kT·L/(ε₀A) (equality in the conductor limit), for every reference distance L, both cell sizes and both cations | n = 8: 0.41 V (mid-film), 0.58 V (Φ_top). n = 16, 40 Å film: 0.20 V, 0.29 V | Variance above the ceiling. S0 CsF currently fails | S0, LC (DFT); SL (head) |
| LR3 | Zero mean field in a salt-containing bulk | 0 within the per-replica SE | Stationary non-zero field | S0 (R1, R5) |
| LR4 | **Counter-ion dose–response:** ⟨z_ion⟩ vs ΔN in salt-free cells against slab Poisson–Boltzmann. The Gouy length 1/(2π·l_B·Γ) falls from 50 to 8 Å as ΔN goes from 1 to 6 at n = 16, across and beyond the cutoff | PB, 40 Å film: ⟨z⟩ = 17.6 / 15.7 / 14.2 / 13.0 / 11.1 Å at ΔN = 1 / 2 / 3 / 4 / 6 (uniform 20). 60 Å film: 21.3 / 16.7 Å at ΔN = 2 / 4 (uniform 30) | ⟨z⟩ stays near uniform beyond the contact layer, and the trend in ΔN is flat | SL; crossing time ~1.2 ns, so SE ≈ 1.5 Å per 15 ns |
| LR5 | Poisson closure: dU/dσ from the head equals dU/dσ from the 1D Poisson integral of the MD-averaged charge densities, up to a σ-independent electronic term | Equal slopes | The potential and the structure disagree | SL (and S0/LC with DFT) |
| LR6 | Size consistency: U at matched σ is the same at n = 8 (LC, DFT) and n = 16 (SL, head) | Equal within SE | U at fixed σ depends on cell size, the signature of a receptive-field limit | LC + SL |

**Caveat on LR4.** PB is mean-field and ignores hydration structure and image forces at the metal and the
water/vacuum surface. The comparison is the trend with ΔN and the profile beyond ~5 Å from the wall, not exact
⟨z⟩ values.

---

## 6. Decisions we need from you

1. Can you run **LC** (20 MD trajectories of 1.2 ns, ~100 DFT single points of ~2200 atoms) and **R1–R7** now?
2. Does a work-function head exist or is one planned (§3)? Please also send the throughput numbers. Both decide
   whether **SL** runs.
3. Is **LR1** affordable (3 × ~22k atoms × 2.2 ns)?
4. On your option list from 2026-10-05:
   - **Many short independent replicas.** We agree.
   - **Re-weighting by DFT electrostatic energy.** We would not use it: a 0.5 V-scale per-snapshot spread
     degenerates the weights.
   - **Moving the reference toward the metal.** Useful, but a reference at ¼ of the film overlaps the Cs⁺ second
     layer.
   - **A long-range MLIP.** We would hold it until R1, R2 and LC show whether it is needed.

## Package contents (`ag100_followup_2026-10-08.tar.gz`)

| Path | Contents |
|---|---|
| `request_to_meta_followup_2026-10-08.md` | This note |
| `s0/build_s0_structures.py` | `--matrix lc|sl|lr1`, `--sl-dn`, `--sl-no-water`, `--prefix`, `--bulk-side`; S0 output unchanged |
| `s0/make_vasp_inputs.py` | Electrolyte map extended for the counter-ion sets (`H2O`, `F_ci`, `Na_ci`, `Cs_ci`, `CsF_bulk`) |
| `s0/structures_lc/` | 20 LC states, manifest, state list |
| `s0/structures_sl_h40/`, `s0/structures_sl_h60/` | 62 + 12 SL states |
| `s0/structures_lr1/` | 3 bulk boxes |
| `CHANGES_2026-10-08.md` | Build commands and validation summaries |
