# Plan v2 (reduced scope): Ag(100) differential capacitance in NaF from OC25 MLIP dynamics

Date: 2026-09-17. Supersedes v1 (`Ag100_OC25_capacitance_plan_v1_full_scope.md`, kept for reference).
Scope: this is **one supporting result** in the OC25 / CO-dimerization paper (main focus: Cu, CO dimerization),
not a standalone paper. Deliverable: one figure (main text or SI) showing that OC25 dynamics plus a DFT-trained
potential predictor reproduce the Ag(100) differential-capacitance curve in NaF (Valette 1982, Fig. 2, 0.1 M
curve as the primary target), plus the two sanity numbers behind it (ionization of excess ions; PZC).

## 0. Legend

| Label | Meaning |
|---|---|
| S0 | Smoke tests: ionization by Bader, two-sided ion start, bulk-water sanity, NaF vs CsF cation check. Files in `s0/`; sub-labels in `TESTS.md` |
| S1 | 1 M NaF, n = 8 cells, ΔN = −7…+7, 4 replicas: MLIP sampling for the DFT labels |
| S2 | DFT potentiometer (RPBE-D3, Γ, OC25 settings) on S1 snapshots → (σ_actual, U_inner) labels |
| S3 | Analysis on the 1 M labels: Cⁱ(σ) from U_inner(σ_actual), Gouy–Chapman stitching → C(E; c) family vs Valette Fig. 2 (Meta's choice, 2026-09-17); the same labels train the predictor φ̂ (Δ-Poisson, §3) |
| S4 | Follow-on, once φ̂ validates: direct 0.1 M NaF dynamics in a large cell (n ≈ 20), potential from φ̂, C(E) read directly; the 0.1 M curve is the consistency check between S3 and S4 |
| U_inner | −(E_F − ⟨V_H⟩_bulk-water)/e per DFT snapshot; the metal-to-bulk-electrolyte drop that C is defined against |
| σ_nominal / σ_actual | e·ΔN/A set by composition / metal charge measured by Bader on the DFT snapshot |

## 1. Settled premises (unchanged from v1 §1)

OC25 models output energy and forces only; the charge state is inferred from composition and double-layer
structure. Charging is by composition (excess Na → Na⁺ + e⁻ in the metal; excess F → F⁻ + hole), in integer
steps per cell. Nominal σ is an intention, verified by Bader on labelled snapshots. The potential axis comes from
DFT single points, never from the model. Use the energy-conserving checkpoint `esen-sm-conserving-all-oc25`
for MD. Meta runs all MLIP dynamics and DFT; compute width is not a constraint.

## 2. What the OC25 paper (arXiv:2509.17862v1) says that matters here

- **Ion set (Fig. 3): Cs⁺, Li⁺, H⁺, Ca²⁺, Me₄N⁺, OH⁻, F⁻, HCO₃⁻, SO₄²⁻.** F⁻ is in distribution (≈3.5 % of
  systems). **Na⁺ and K⁺ are not training ions.** Na is among the 88 elements only as a bulk/slab constituent.
  Consequence: NaF is an out-of-distribution cation; CsF (or LiF) is fully in distribution. Decision in §4.
- Charged interfaces were created by ion imbalance (an ion is sampled ≈50 % of the time, one per cell, cells
  tiled to ≥8 Å laterally), evaluated as neutral DFT cells, giving metal surface charges from ≈−80 to +60 µC/cm²
  with most systems at zero. This is exactly our composition-charging scheme, at higher |σ| than we need.
- Solvent depth 5–10 Å (mean 5.6 Å); mean ion concentration ≈2 M, minimum 0.38 M. A 35 Å film with a bulk-like
  middle and a 0.1 M diffuse layer are extrapolations in depth and concentration (not in local environment).
  Meta reports the message passing captures long-range electrostatics; S0 bulk-water and profile checks test it.
- DFT: VASP 6.3.2, RPBE, D3 zero damping (IVDW = 11), ENCUT 400, EDIFF 1e-4, ISMEAR 0 / SIGMA 0.1,
  IDIPOL 3 + LDIPOL, non-spin-polarized, k-mesh by reciprocal density 40. Our potentiometer uses the same
  settings with EDIFF tightened to 1e-5 and LVHAR/LAECHG for the potential and Bader reads.
- Outlook section: "it would be highly desirable to have ML models that can predict the interface workfunctions
  directly." The φ̂ predictor of §3 is therefore aligned with Meta's own stated interest.

## 3. Potential predictor φ̂ (replaces "DFT on every production snapshot")

Goal: a cheap per-frame potential so that the C curve comes from long MD in a large cell where the DFT
potentiometer is unaffordable, with DFT used only to label a few hundred small-cell snapshots.

Recommended form: **Δ-learning on top of an exact classical Poisson baseline**, linear in a handful of
intensive features.
1. Baseline U_cl per frame: solve 1D Poisson for the planar-averaged classical charge density (Na/Cs +1, F −1,
   water as SPC/E point charges) from the metal image plane (z_im, a fitted constant) to the bulk-water reference
   slab, with the metal as a conductor. This is exact for the diffuse layer and most of the inner layer, is
   cell-size independent by construction, and contains the 0.1 M dip physics (Gouy–Chapman) without learning.
2. Residual ΔU = U_inner(DFT) − U_cl regressed linearly (ridge) on intensive features: σ (excess ions per area,
   laterally local if needed), first-layer water density and ⟨cos θ⟩, contact-adsorbed F⁻ per area, contact
   cations per area. The residual is the electronic part (push-back, image-plane shift, water polarizability
   beyond fixed charges) plus a constant that cancels in C. Expect it smooth and ≲0.3 V in range.
3. Validation: hold out entire ΔN states (σ interpolation/extrapolation); hold out ≈20–30 snapshots from a
   12×12 or 10×10 cell at two σ values (size transfer, the property that matters for S4). Pass criterion:
   ≤30 mV RMSE on held-out states and no σ-correlated residual (slope of residual vs σ within ±1 mV per µC/cm²,
   because a σ-dependent bias is what would distort C).
Alternative Meta may prefer: a linear/MLP head on pooled OC25 node embeddings. Run both if they are willing;
the Δ-Poisson model is the physics-constrained fallback and the check on the embedding model's size transfer.
Note the dip is a 10–15 % feature in C at 0.1 M; only a σ-dependent predictor bias can hide it, and in the
Δ-Poisson form the dip sits in the exact classical part.

Precedent for the embedding-head route: Wang, Fang, Huang, Liu, J. Chem. Theory Comput. 21, 7628 (2025)
(`ct5c00784.pdf`, "CP-MLFF" in MACE). They add the electron count N_e as a global input (either appended to every
initial node feature, or as a separate global channel exchanged with the nodes at each layer), mean-pool the
invariant (l = 0) node features of every layer, and read E_F out with an MLP trained by regression alongside
forces. Labels came from CP-VASP AIMD with an implicit-solvent bulk-electrolyte reference (E_F evaluated every
5th step because the reference is costly); ≈400 initial + active-learning structures (ensemble of 4, thresholds
150 meV/Å on forces and 40 meV on E_F) to ≈1100 total, giving E_F RMSE ≈11–22 meV on training data and
≈56 meV on new structures; size transfer tested only to a ≈1.7× larger graphene cell with 72 waters and needed
a few added structures. Mapping to us: composition charging fixes N_e by neutrality, so the N_e input drops
and the head is simply pooled OC25 node features → U_inner. Caveat that motivates the Δ-Poisson form: a
mean-pooled invariant readout has no explicit handle on the z-dipole integral that sets the drop across a
35–80 Å film with a diffuse layer, which their 44-water films never exercised; putting Poisson in analytically
and learning only the local residual (or learning the residual with such a head) avoids that extrapolation.
Their implicit-solvent potential reference is not adopted: with an implicit region above the explicit film the
countercharge is placed in the continuum far from the interface, so the double layer that sets E_F is not the
one in the frame. Labels stay explicit: U_inner from the bulk-water Hartree reference. Decision 2026-09-17:
Δ-Poisson form is the primary predictor; an embedding head, if Meta trains one, is a comparison only.

## 4. Electrolyte decision

Target is NaF (Valette Fig. 2), but Na⁺ is not an OC25 training ion. Rule: S0 runs NaF and CsF side by side
(`TESTS.md` S0.5). If Na⁺ passes S0 (Bader ≥ +0.9, intact hydration shell, no anomalous contact adsorption
compared with Cs⁺, no reactive events), keep NaF for fidelity to the experiment and report the Cs comparison
as the in-distribution control. If Na⁺ fails, switch to CsF and lean on the experimental fact that the negative
branch of C(E) is cation-independent on Ag(100) (Valette Figs. 4–5: NaF, NaClO₄, KPF₆ coincide below −1.0 V
SCE), so CsF vs NaF differences are expected only at the far negative wing. Dropped from v1: KF (K⁺ is also
not in OC25, so it hedges nothing), NaClO₄ (ClO₄⁻ not in OC25), classical simulations, T-dependence,
fine-tune loop, n = 10 finite-size control (only the φ̂ size-transfer hold-out remains, §3).

## 5. System and numbers

- Ag(100), n = 8 lateral (≈23.5 Å), 4 layers with the bottom 2 fixed (3-layer option dropped 2026-09-17),
  lattice constant from `ag_rpbe_d3zero_CONTCAR`: a = 4.162 Å (NN 2.943 Å), RPBE-D3 with zero damping (IVDW = 11,
  the OC25 setting), relaxed on Quanah 2026-09-17 (ENCUT 500, 12×12×12, ISIF 3, residual pressure 0.7 kbar). This
  replaces the 3.844 Å value used earlier, which came from a BJ-damped (IVDW = 12) run; 4.162 Å is 1.9 % above
  experiment (4.086 Å), the expected direction for RPBE (plain RPBE gives 4.215 Å). With this a: area 554 Å², σ per
  ion 2.89 µC/cm², 12 NaF pairs and 636 H₂O in a 35 Å film (2188 atoms at ΔN = 0). Meta
  confirmed (2026-09-17) that OC25 bulks were relaxed with RPBE-D3, so the training Ag slabs are at the same constant.
- Water film 35 Å, vacuum 15 Å, 298 K, Langevin NVT, 1 fs.
- S1 grid: 1 M NaF, ΔN = −7…+7 (15 states), 4 replicas each (2 surface-start + 2 far-start), 0.5 ns
  equilibration + 1.5 ns production; snapshots every 100 ps → 60 per state → ≈900 DFT labels total, plus
  30 hold-out snapshots at n = 10 or 12. The replica count (4, down from 16–32 in v1) follows from the change of
  role: these cells only supply φ̂ training labels and the ionization/PZC checks, not the C curve itself; the
  between-replica spread still reports the slow ion-exchange degree of freedom.
- S4 cell: n = 20 (≈59 Å), film 60–80 Å, 0.1 M NaF (≈16 pairs in 80 Å), ≈25–29k atoms, 4-layer slab; σ per
  excess ion ≈0.5 µC/cm², so ΔN = 0…±40 covers ±20 µC/cm² with ≈20 points across the ±5 µC/cm² hump region;
  per-frame potential noise is ≈2.5× smaller than at n = 8 (area ratio). 2 replicas per state, ≥2 ns production,
  φ̂ evaluated on every frame. Concentration 0.1 M is the lowest Valette curve reachable directly; the
  0.005–0.04 M curves, if wanted at all, come from Gouy–Chapman stitching of the inner-layer Cⁱ(σ) (v1 §4.3
  Track 1) and are optional for this paper.

## 6. S0 (built 2026-09-17 in `s0/`, see `TESTS.md`)

Matrix: NaF, n = 8, L4 only, ΔN ∈ {−2…+2}, 4 replicas (two-sided starts on charged states); CsF L4 at
ΔN ∈ {−2, 0, +2}, 2 replicas; bulk H₂O ×2 and bulk 1 M NaF ×1. 0.1 ns equilibration + 0.4 ns production
each. DFT on the last 5 snapshots per replica (20 per state) at native 4-layer thickness (no slab-thickness
comparison: 3-layer and 8-layer-extended variants both dropped 2026-09-17). Go/no-go: σ_actual/σ_nominal ≥ 0.9; contact-adsorbed F⁻
retains ≥ 0.7 e⁻; quick-look C at the PZC from the ±2 states within 40–100 µF/cm² (Valette Cⁱ at the PZC per
true area ≈ 60–90); bulk water liquid at 298 K with
density within 10 % of 1 g/cm³ and D within a factor 2 of experiment; no HF, no H on Ag, no desorption into
vacuum. Per-snapshot U_inner scatter measured here sets the S1 snapshot count.

## 7. Milestones for the paper (order agreed with Meta 2026-09-17)

1. S0 verdict (ionization table, PZC vs −0.62 V SHE, NaF vs CsF decision, per-snapshot U noise).
2. Cⁱ(σ) from the 1 M labels vs Valette Fig. 9, then Gouy–Chapman stitching → C(E; 0.005–0.1 M) vs Fig. 2
   (the figure for the paper; the dip is analytic by construction, the tests are the PZC, the peak Cⁱ, the plateau
   K₀ and the hump position/asymmetry).
3. φ̂ trained on the same labels and validated (held-out ΔN states, size transfer to n = 10/12 hold-outs).
4. Follow-on: direct 0.1 M cell with φ̂; agreement with the stitched 0.1 M curve is the internal consistency test.
5. Optional SI: F⁻ contact-adsorption count per area vs Fig. 6.

## 8. Questions for Meta

1. Confirm no Na⁺/K⁺ solvated-ion data in OC25 (paper Fig. 3) and whether any Ag/water/F⁻ or Ag/water/Cs⁺
   systems exist by count.
2. Ag slab thickness used in OC25 generation (lattice constant settled: RPBE-D3 relaxed, same as ours).
3. Whether they will (a) run the S2 DFT array at ≈2200 atoms Γ-only with LVHAR/LAECHG, (b) train an
   embedding-based potential head on the S2 labels in parallel with our Δ-Poisson model.
4. Practical: their fairchem version, whether `FAIRChemCalculator` needs a task name for the OC25 checkpoints,
   HF access for the gated `facebook/OC25` repo.

## 9. References
Valette, J. Electroanal. Chem. 138, 37 (1982). Sundararaman et al., J. Chem. Phys. 148, 144105 (2018).
Ludwig, Singh, Nørskov, J. Phys. Chem. Lett. 11, 9802 (2020). Wang, Fang, Huang, Liu, J. Chem. Theory Comput.
21, 7628 (2025) (CP-MLFF, `ct5c00784.pdf`). Shin, Kim, Bae, Ringe, Choi, Lim, Choi, Kim, Nat. Commun. 13, 174
(2022), doi:10.1038/s41467-021-27909-x (DFT-CES camel-shaped C(E), Ag(111)/NaF; `s41467-021-27909-x.pdf`, SI
`41467_2021_27909_MOESM1_ESM.pdf`). OC25: arXiv:2509.17862v1; models at
huggingface.co/facebook/OC25 (gated); code github.com/facebookresearch/fairchem.

Closest precedent, Shin et al. 2022 (added 2026-10-07). Same observable (C = dσ/dE vs Valette-type NaF data) and
same charging idea (excess electrons balanced by excess Na⁺/F⁻), but a different level of theory and a different
facet. Their DFT-CES is a mean-field QM/MM: a frozen 4-layer PBE Ag(111) (√3×2) cell tiled to 46×47 Å, 2000
TIP3P-EW waters with Fyta–Netz ions and fitted Ag–water/ion Buckingham terms (vdW-DF2 and HF on an Ag₆ cluster),
coupled only through the 5 ns-averaged electrolyte potential; E is Trasatti's E_F-vs-vacuum minus 4.4 V, which
works for them because the water/vacuum surface potential is time-averaged before E_F is read, whereas our
per-snapshot DFT carries the instantaneous surface term (v1 §3.2; S0 Φ_top SD 0.52 V NaF, 1.05 V CsF); one
converged mean-field state per σ, so no per-snapshot potential noise and σ steps of ≈0.75 µC/cm². Their cells
contain only the counter-ions (no background salt), so no Gouy–Chapman diffuse layer is present. Ours: every
atom from one DFT-trained potential (OC25, RPBE-D3), per-snapshot DFT potentiometer, U_inner against bulk water,
1 M / 0.1 M salt, Ag(100). Consequences for us: (i) their S-shaped σ–E (bistable water orientation, Maxwell
construction, cathodic hump near σ ≈ −10 µC/cm²) means S3 must allow a non-monotonic U_inner(σ) rather than
assume Cⁱ > 0; (ii) their 15C5 result (larger cation–metal distance shifts the cathodic hump negative) cautions
that the §4 CsF fallback may move the cathodic hump, beyond what Valette's Na⁺/K⁺ data test; (iii) their
anodic hump is F⁻ specific adsorption set by a fitted Ag–F⁻ pair term with a fixed −1 charge, which our Bader
check (contact F⁻ ≥ 0.7 e⁻) and DFT-level F⁻–Ag interaction address directly.
