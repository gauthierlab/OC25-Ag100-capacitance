# Label legend for the Ag(100)/OC25 capacitance project

Stage labels (S0–S4) are defined in `Ag100_OC25_capacitance_plan.md` §0. S0 sub-tests, all under `s0/`:

| Label | One-line description | State IDs |
|---|---|---|
| S0.1 | Training-coverage check from the OC25 paper (done 2026-09-17: F⁻ in, Na⁺/K⁺ out, charged cells by ion imbalance, films ≤10 Å) | none (literature) |
| S0.2 | Ionization / Bader on DFT snapshots: every excess Na ≈ +1, F ≈ −1, balancing charge on the metal; three σ estimators cross-checked | all interface states, DFT `native/` |
| S0.3 | Bulk-water sanity of the MLIP at 298 K: density, O–O RDF, self-diffusion, no freezing; plus 1 M NaF bulk ion-pairing check | `bulk_H2O_r1/2`, `bulk_NaF_r1` |
| S0.4 | Two-sided ion start: replicas r1,r2 start with excess ions at the surface, r3,r4 with excess ions ≥12 Å away; convergence of ion profiles sets the equilibration time | every charged NaF state |
| S0.5 | Cation check: CsF (Cs⁺ in OC25) vs NaF (Na⁺ not in OC25) at ΔN = −2, 0, +2 | `L4_n8_CsF_*` |
| S0.6 | Quick-look C at the PZC from the ΔN = ±2 states and the per-snapshot U_inner noise floor; L4 (2 fixed + 2 free) at native thickness only. The 3-layer comparison and the 8-layer DFT extension (`ext8/`) were both dropped 2026-09-17 at the user's call | `L4_n8_NaF_*` |

## Follow-up labels (2026-10-08, `plan_meta_followup_2026-10-08.md`, `request_to_meta_followup_2026-10-08.md`)

| Label | One-line description |
|---|---|
| R1 | Mean film field per replica ± SE, and its drift with time (existing S0 data) |
| R2 | Autocorrelation times of p_ion, p_water, p_ion+p_water; Maxwell (~0.1 ns at 1 M) vs diffusive (~1 ns) |
| R3 | Control-variate slope dU_adj/dσ with X = p_ion + p_water averaged over all MD frames |
| R4 | Contact-F⁻ Bader charges (criterion ≥ 0.7 e⁻) |
| R5 | Snapshot-averaged V_H(z) fitted to linear + exponential; mean V_H − E_F at ¼, ½, ¾ film |
| R6 | MD-averaged ρ_ion(z), P_z(z) and their classical Poisson potential |
| R7 | Ag–O, Ag–Na, Ag–Cs, Ag–F surface pseudo-RDFs, first minima, coordination numbers, z-densities |
| W0 / W1 | Questions about an OC25 work-function head / its validation on the 326 S0+PZC DFT snapshots |
| LC | 8×8 counter-ion-only set (pure water ×4, F⁻ ±1,2 ×4, Na⁺/Cs⁺ −1,−2 ×2), MD + DFT; `s0/structures_lc/` |
| SL | Shin-like n = 16 salt-free set, 40 Å film (62 states) + 60 Å check (12); head only; `s0/structures_sl_h40/`, `_h60/` |
| LR1 | Bulk Stillinger–Lovett / longitudinal-dielectric sum rules; 60 Å boxes `s0/structures_lr1/` |
| LR2 | Interfacial fluctuation ceiling var(U) ≤ kT·L/(ε₀A) |
| LR3 | Zero mean field in salt-containing bulk |
| LR4 | Counter-ion ⟨z⟩ vs ΔN against slab Poisson–Boltzmann (Gouy length 50 → 8 Å) |
| LR5 | Poisson closure: dU/dσ from the head equals that from the averaged charge densities |
| LR6 | Size consistency: U at matched σ, n = 8 (LC, DFT) vs n = 16 (SL, head) |
