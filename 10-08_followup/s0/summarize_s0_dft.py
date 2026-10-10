#!/usr/bin/env python3
"""
summarize_s0_dft.py
====================

Group potentiometer_summary.csv (written by analyze_potentiometer.py) by
(slab_layers_md, slab_layers_dft, electrolyte, delta_N) and report, per group:
  - mean +/- s.e.m. of U_inner_V
  - mean +/- s.e.m. of sigma_a, sigma_b, sigma_c (uC/cm^2)
  - ionization ratio sigma_a / sigma_nominal
Then, pooling across states (mean sigma_a, mean U_inner per state), a quick-look capacitance
at the PZC from a linear fit of <U_inner> vs <sigma_a>: C = d(sigma)/d(U) in uF/cm^2, with the
fit's standard error.
(Slab-thickness comparisons were dropped 2026-09-17; only native-thickness jobs exist.) The S0
go/no-go checks from README_DFT.md.

Writes a markdown table to <out-dir>/s0_dft_summary.md (and prints it).
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

import numpy as np


def read_rows(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        for k in ("delta_N", "slab_layers_md", "slab_layers_dft"):
            if r.get(k) not in (None, ""):
                try:
                    r[k] = int(float(r[k]))
                except ValueError:
                    pass
        for k in (
            "sigma_nominal_uC_cm2", "sigma_a_uC_cm2", "sigma_b_uC_cm2", "sigma_c_uC_cm2",
            "U_inner_V", "U_inner_ref_noise_V",
        ):
            if r.get(k) not in (None, ""):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    r[k] = None
            else:
                r[k] = None
    return rows


def mean_sem(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values) / (len(values) ** 0.5)


def group_key(r):
    return (r.get("slab_layers_md"), r.get("slab_layers_dft"), r.get("electrolyte"), r.get("delta_N"))


def build_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)
    return groups


def agreement_verdict(diff, sem_a, sem_b):
    """Compare |diff| to the combined s.e.m., with an explicit label (not a bare "no") when
    one or both groups have too few replicates (n=1, sem undefined/NaN) to make the call."""
    if sem_a is None or sem_b is None or (isinstance(sem_a, float) and np.isnan(sem_a)) or (
        isinstance(sem_b, float) and np.isnan(sem_b)
    ):
        return "n/a (n=1 in a group, sem undefined)"
    sem_bound = sem_a + sem_b
    if sem_bound == 0:
        return "yes (exact, but n=1 both -- no error bar)" if diff == 0 else "no (n=1 both, no error bar)"
    return "yes" if diff <= sem_bound else "no"


def linear_fit_capacitance(x, y):
    """C = dy/dx via ordinary least squares; returns (slope, slope_stderr) in the units of
    y/x. Here x = sigma (uC/cm^2), y = U_inner (V), and reported capacitance is dsigma/dU =
    1/slope (uF/cm^2), with error propagated from the slope's standard error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2:
        return None, None
    A = np.vstack([x, np.ones(n)]).T
    (m, b), res, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    if n > 2:
        y_pred = A @ [m, b]
        resid = y - y_pred
        s2 = np.sum(resid ** 2) / (n - 2)
        Sxx = np.sum((x - x.mean()) ** 2)
        m_stderr = np.sqrt(s2 / Sxx) if Sxx > 0 else float("nan")
    else:
        m_stderr = float("nan")
    return m, m_stderr


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=None, help="default: <out-dir>/potentiometer_summary.csv")
    p.add_argument("--out-dir", default=None, help="default: ROOT/s0/dft")
    args = p.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "dft")
    csv_path = args.csv or os.path.join(out_dir, "potentiometer_summary.csv")

    rows = read_rows(csv_path)
    if not rows:
        print(f"No rows in {csv_path}", file=sys.stderr)
        return 1

    groups = build_groups(rows)

    lines = []
    lines.append(f"# S0 DFT potentiometer summary\n")
    lines.append(f"Source: `{csv_path}` ({len(rows)} single points, {len(groups)} groups)\n")

    lines.append("## Per-group means (slab_layers_md, slab_layers_dft, electrolyte, delta_N)\n")
    header = (
        "| L_md | L_dft | electrolyte | dN | n | U_inner (V) | sigma_nom | "
        "sigma_a | sigma_b | sigma_c | ioniz. ratio (a/nom) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    group_summary = {}
    for key in sorted(groups.keys(), key=lambda k: (str(k[0]), str(k[1]), str(k[2]), (k[3] if k[3] is not None else 0))):
        grp = groups[key]
        n = len(grp)
        u_mean, u_sem = mean_sem([r["U_inner_V"] for r in grp])
        a_mean, a_sem = mean_sem([r["sigma_a_uC_cm2"] for r in grp])
        b_mean, b_sem = mean_sem([r["sigma_b_uC_cm2"] for r in grp])
        c_mean, c_sem = mean_sem([r["sigma_c_uC_cm2"] for r in grp])
        nom_vals = [r["sigma_nominal_uC_cm2"] for r in grp if r["sigma_nominal_uC_cm2"]]
        nom = nom_vals[0] if nom_vals else None
        ratio = (a_mean / nom) if (a_mean is not None and nom) else None

        group_summary[key] = dict(n=n, U_inner=u_mean, U_inner_sem=u_sem, sigma_a=a_mean, sigma_a_sem=a_sem,
                                   sigma_b=b_mean, sigma_c=c_mean, sigma_nominal=nom, ionization_ratio=ratio)

        def fmt(v, s=None, nd=3):
            if v is None:
                return "n/a"
            if s is not None and not (isinstance(s, float) and np.isnan(s)):
                return f"{v:.{nd}f} +/- {s:.{nd}f}"
            return f"{v:.{nd}f}"

        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {n} | {fmt(u_mean, u_sem)} | "
            f"{fmt(nom, nd=2) if nom is not None else 'n/a'} | {fmt(a_mean, a_sem, nd=2)} | "
            f"{fmt(b_mean, nd=2)} | {fmt(c_mean, nd=2)} | {fmt(ratio, nd=3)} |"
        )
    lines.append("")

    # --- Quick-look capacitance at the PZC ---
    lines.append("## Quick-look capacitance at the PZC\n")
    lines.append(
        "Linear fit of group-mean U_inner (V) vs group-mean sigma_a (uC/cm^2) across all "
        "available states/groups (native, thinnest available slab, all delta_N): "
        "C = d(sigma)/d(U) = 1/slope.\n"
    )
    xs = [g["sigma_a"] for g in group_summary.values() if g["sigma_a"] is not None and g["U_inner"] is not None]
    ys = [g["U_inner"] for g in group_summary.values() if g["sigma_a"] is not None and g["U_inner"] is not None]
    if len(xs) >= 2:
        slope, slope_stderr = linear_fit_capacitance(xs, ys)  # slope = dU/dsigma (V per uC/cm2)
        if slope and slope != 0:
            C = 1.0 / slope  # uF/cm^2
            C_err = abs(C) * (slope_stderr / abs(slope)) if slope_stderr and not np.isnan(slope_stderr) else float("nan")
            lines.append(f"**C(PZC, quick-look) = {C:.1f} +/- {C_err:.1f} uF/cm^2** (n={len(xs)} groups; dU/dsigma slope = {slope:.5f} +/- {slope_stderr:.5f} V per uC/cm^2)\n")
        else:
            lines.append("Fit slope is zero or undefined; cannot report C.\n")
    else:
        lines.append("Fewer than 2 groups with both sigma_a and U_inner; cannot fit C.\n")

    # --- native vs ext8 comparison (L3 vs L4 dropped 2026-09-17; 4 layers only) ---
    lines.append("## Go/no-go checks\n")
    # Slab-thickness comparisons (L3 vs L4, native vs ext8) were dropped 2026-09-17: S0 runs
    # 4-layer slabs only and the DFT is done at native thickness only.

    lines.append(
        "Go/no-go criteria (from README_DFT.md): ionization sigma_a/sigma_nominal >= 0.9; "
        "contact-adsorbed F retains >= 0.7 e-; quick-look C at PZC within 40-100 uF/cm2 (true "
        "area).\n"
    )

    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s0_dft_summary.md")
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nWrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
