# OC25 Ag(100) differential capacitance

Shared workspace between the Gauthier group (Texas Tech) and Meta FAIR Chemistry for computing the
Ag(100)/aqueous NaF differential capacitance C(E) with the OC25 machine-learned interatomic potential,
benchmarked against Valette (1982).

## Layout

Each exchange gets its own folder named `MM-DD_<topic>` (the date it was sent). The folder's top-level
markdown file explains what changed, what to run, and what data to send back. Replies go in a new folder
with the same convention, rather than as edits to an earlier one.

| Folder | From | Contents |
|---|---|---|
| [`10-08_followup`](10-08_followup) | Gauthier group | Follow-up to the 2026-10-05 S0 potential diagnostics: requests R1–R7 on existing data, new structure sets LC, SL and LR1, and the long-range tests LR1–LR6. Read `request_to_meta_followup_2026-10-08.md` first, then `CHANGES_2026-10-08.md`. |

This folder is identical to `ag100_followup_2026-10-08.tar.gz`, apart from `.gitkeep` files that hold the empty
`s0/dft/` and `s0/md/` output directories in place.

## License

MIT; see [LICENSE](LICENSE).
