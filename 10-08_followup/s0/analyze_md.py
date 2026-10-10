#!/usr/bin/env python3
"""
analyze_md.py -- post-hoc analysis of S0 MD trajectories.

For interface states (Ag(100) + water film + electrolyte): planar density
profiles, first-layer water dipole orientation, ion contact statistics,
film density, bulk-region ion concentration, and a classical dipole-proxy
potential estimate. For bulk boxes (state_id starting with "bulk_"):
density, O-O RDF, O self-diffusion (Einstein/MSD), and a freezing check.

Also produces a two-sided-start ("surface" vs "far" initial ion placement)
comparison for each (slab_layers, electrolyte, delta_N) group, and a
cross-state summary.csv.

Dipole-proxy potential formula (documented here per the task spec):
  For atoms above the top Ag layer, with SPC/E-style point charges q_i
  (O: -0.8476e, H: +0.4238e, Na/Cs: +1e, F: -1e), and z_i measured relative
  to the mean position of the top Ag layer:
      P/A  = (1/A) * sum_i q_i * z_i         [e * Angstrom / Angstrom^2 = e/Angstrom]
      dV   = (P/A) * e / eps0                [V], after converting e/Angstrom to C/m
  i.e. the parallel-plate-capacitor relation for a sheet dipole density,
  the same convention used for slab dipole corrections (Neugebauer-Scheffler
  / Bengtsson-style). e and eps0 are physical constants (not SPC/E-specific).
  This is a cheap classical proxy for the interfacial potential drop; it is
  NOT a substitute for a proper Helmholtz-plane / Poisson analysis, and it
  ignores electronic polarization of the metal and the MLIP's own (implicit)
  charge distribution entirely.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list

sys.path.insert(0, str(Path(__file__).resolve().parent))
import md_common as mc  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

SCRIPT_DIR = Path(__file__).resolve().parent

AVOGADRO = 6.02214076e23
AMU_TO_G = 1.66053906660e-24
E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
# Volts per (e/Angstrom) of areal dipole density; see module docstring.
DIPOLE_PROXY_CONST_V_PER_E_PER_ANGSTROM = E_CHARGE * 1e10 / EPS0

SUMMARY_FIELDS = [
    "state_id", "prod_ps_analyzed", "T_mean", "film_density_middle",
    "n_F_contact_mean", "n_cation_contact_mean", "bulk_conc_M",
    "dipole_proxy_mean_V", "dipole_proxy_std_V", "n_HF_max", "n_OH_max",
    "n_H_on_Ag_max", "n_desorbed_max",
    # extra diagnostics beyond the INTERFACE_SPEC-mandated core columns
    "kind", "bulk_density_g_cm3", "OO_rdf_r1_A", "OO_rdf_g1",
    "O_self_diffusion_cm2_s", "freezing_second_peak_split",
    "tetrahedral_q_mean",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Post-hoc analysis of S0 MD runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--state-id", nargs="+", default=None)
    g.add_argument("--all", action="store_true", help="Analyze every state under --md-dir.")
    p.add_argument("--md-dir", default=str(SCRIPT_DIR / "md"))
    p.add_argument("--structures-dir", default=str(SCRIPT_DIR / "structures"))
    p.add_argument("--skip-ps", type=float, default=None,
                    help="Production window start (ps). Default: read the "
                         "equil length from run.log, else from thermo.csv's "
                         "phase=='equil' rows, else 0.")
    p.add_argument("--bin-width", type=float, default=0.1, help="z-bin width (Angstrom).")
    p.add_argument("--film-middle-halfwidth", type=float, default=5.0,
                    help="Half-width (Angstrom) of the 'middle of the film' region "
                         "used for film density / bulk ion concentration.")
    p.add_argument("--msd-fit-start-ps", type=float, default=10.0)
    p.add_argument("--msd-fit-end-ps", type=float, default=100.0)
    p.add_argument("--rdf-dr", type=float, default=0.05)
    p.add_argument("--cut-oh", type=float, default=mc.DEFAULT_CUTOFFS["oh"])
    p.add_argument("--convergence-tol-A", type=float, default=1.0,
                    help="Threshold (Angstrom) for the two-sided-start "
                         "equilibration-time estimate.")
    return p


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def discover_states(md_dir):
    md_dir = Path(md_dir)
    if not md_dir.exists():
        return []
    return sorted(p.name for p in md_dir.iterdir()
                  if p.is_dir() and (p / "traj.extxyz").exists())


def get_equil_ps_from_log(run_log):
    run_log = Path(run_log)
    if not run_log.exists():
        return None
    pat = re.compile(r"equil_steps=(\d+).*?dt_fs=([0-9.]+)")
    for line in run_log.read_text().splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1)) * float(m.group(2)) / 1000.0
    return None


def resolve_skip_ps(args, state_dir, thermo):
    if args.skip_ps is not None:
        return args.skip_ps
    v = get_equil_ps_from_log(state_dir / "run.log")
    if v is not None:
        return v
    eq = thermo[thermo["phase"] == "equil"]
    if len(eq):
        return float(eq["time_ps"].max())
    return 0.0


def load_state_tables(md_dir, state_id):
    d = Path(md_dir) / state_id
    thermo = pd.read_csv(d / "thermo.csv")
    topo = pd.read_csv(d / "topology.csv")
    return d, thermo, topo


def load_prod_frames(d, thermo, skip_ps):
    frames = read(d / "traj.extxyz", index=":")
    times = thermo["time_ps"].values
    n = min(len(frames), len(times))
    frames, times = frames[:n], times[:n]
    mask = times >= (skip_ps - 1e-9)
    sel_frames = [f for f, m in zip(frames, mask) if m]
    sel_times = times[mask]
    return sel_frames, sel_times


def species_of_role(atoms, role_name):
    syms = np.array(atoms.get_chemical_symbols())
    role = np.array(atoms.arrays.get("role", np.array([""] * len(atoms), dtype=object)))
    present = sorted(set(syms[role == role_name].tolist()))
    return present[0] if present else None


def write_csv(path, df):
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Interface-state analysis
# --------------------------------------------------------------------------
def analyze_interface(state_id, entry, d, thermo, topo, frames, ftimes, out_dir, args, log):
    prod_topo = topo[topo["time_ps"] >= (ftimes[0] if len(ftimes) else 0) - 1e-9]
    z_ref = float(prod_topo["z_top_Ag"].mean()) if len(prod_topo) else 0.0

    area = entry.get("area_A2") if entry else None
    if area is None:
        cell = frames[0].get_cell()
        area = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        log.warning("[%s] no area_A2 in manifest; using cell a x b = %.3f A^2", state_id, area)

    cation_sym = species_of_role(frames[0], "cation")
    anion_sym = species_of_role(frames[0], "anion")

    z_slab_top = entry.get("z_slab_top_A") if entry else z_ref
    z_film_top = entry.get("z_film_top_A") if entry else None
    if z_film_top is None:
        syms0 = np.array(frames[0].get_chemical_symbols())
        o0 = np.where(syms0 == "O")[0]
        z_film_top = float(frames[0].get_positions()[o0, 2].max()) if len(o0) else z_ref + 10.0
    mid_z = 0.5 * (z_slab_top + z_film_top)
    hw = args.film_middle_halfwidth

    cell0 = frames[0].get_cell()
    Lz = float(cell0[2, 2])
    bin_w = args.bin_width
    bins = np.arange(-5.0, Lz - z_ref + 5.0, bin_w)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    hist_O = np.zeros(len(bin_centers))
    hist_H = np.zeros(len(bin_centers))
    hist_cat = np.zeros(len(bin_centers))
    hist_an = np.zeros(len(bin_centers))
    dip_sum = np.zeros(len(bin_centers))
    dip_cnt = np.zeros(len(bin_centers))

    film_density_series = []
    bulk_conc_series = []
    dipole_proxy_series = []

    for atoms in frames:
        syms = np.array(atoms.get_chemical_symbols())
        pos = atoms.get_positions()
        z = pos[:, 2]
        zrel = z - z_ref
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        idx_O = np.where(syms == "O")[0]
        idx_H = np.where(syms == "H")[0]
        idx_cat = np.where(syms == cation_sym)[0] if cation_sym else np.array([], dtype=int)
        idx_an = np.where(syms == anion_sym)[0] if anion_sym else np.array([], dtype=int)

        hist_O += np.histogram(zrel[idx_O], bins=bins)[0]
        hist_H += np.histogram(zrel[idx_H], bins=bins)[0]
        if len(idx_cat):
            hist_cat += np.histogram(zrel[idx_cat], bins=bins)[0]
        if len(idx_an):
            hist_an += np.histogram(zrel[idx_an], bins=bins)[0]

        # first-layer water dipole orientation: bisector of H-O-H vs +z
        bonds = mc.water_bond_map(atoms, oh_cut=args.cut_oh)
        for o, hs in bonds.items():
            h1, h2 = hs
            v1, n1 = mc.mic_vectors(pos[o], pos[h1], cell, pbc)
            v2, n2 = mc.mic_vectors(pos[o], pos[h2], cell, pbc)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            u1, u2 = v1 / n1, v2 / n2
            bis = u1 + u2
            nb = np.linalg.norm(bis)
            if nb < 1e-8:
                continue
            costheta = (bis / nb)[2]
            bidx = int(np.searchsorted(bins, zrel[o]) - 1)
            if 0 <= bidx < len(bin_centers):
                dip_sum[bidx] += costheta
                dip_cnt[bidx] += 1

        # film density in the middle of the film (exclude Ag)
        sel = (z >= mid_z - hw) & (z <= mid_z + hw) & (syms != "Ag")
        vol_cm3 = (area * 1e-16) * (2 * hw * 1e-8)
        mass_amu = atoms.get_masses()[sel].sum()
        film_density_series.append(mass_amu * AMU_TO_G / vol_cm3)

        # ions in the same middle region -> "bulk" concentration
        sel_ion = sel & np.isin(syms, [s for s in (cation_sym, anion_sym) if s])
        n_ions = int(sel_ion.sum())
        bulk_conc_series.append((n_ions / AVOGADRO) / (vol_cm3 / 1000.0))

        # classical dipole-proxy potential, atoms above the top Ag layer
        sel_above = z > z_ref
        q = np.array([mc.SPCE_CHARGES.get(s, 0.0) for s in syms[sel_above]])
        p_over_A = float(np.sum(q * zrel[sel_above])) / area  # e/Angstrom
        dipole_proxy_series.append(p_over_A * DIPOLE_PROXY_CONST_V_PER_E_PER_ANGSTROM)

    rho_O = hist_O / (len(frames) * bin_w * area)
    rho_H = hist_H / (len(frames) * bin_w * area)
    rho_cat = hist_cat / (len(frames) * bin_w * area)
    rho_an = hist_an / (len(frames) * bin_w * area)
    dip_mean = np.divide(dip_sum, dip_cnt, out=np.full_like(dip_sum, np.nan), where=dip_cnt > 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "density_profile.csv", pd.DataFrame({
        "z_rel_topAg_A": bin_centers, "rho_O_per_A3": rho_O, "rho_H_per_A3": rho_H,
        "rho_cation_per_A3": rho_cat, "rho_anion_per_A3": rho_an,
    }))
    write_csv(out_dir / "water_dipole_orientation.csv", pd.DataFrame({
        "z_rel_topAg_A": bin_centers, "mean_cos_theta": dip_mean, "n_samples": dip_cnt,
    }))
    write_csv(out_dir / "film_density_timeseries.csv", pd.DataFrame({
        "time_ps": ftimes, "film_density_g_cm3": film_density_series,
    }))
    write_csv(out_dir / "bulk_conc_timeseries.csv", pd.DataFrame({
        "time_ps": ftimes, "bulk_conc_M": bulk_conc_series,
    }))
    write_csv(out_dir / "dipole_proxy_timeseries.csv", pd.DataFrame({
        "time_ps": ftimes, "dipole_proxy_V": dipole_proxy_series,
    }))
    write_csv(out_dir / "contact_counts_timeseries.csv", pd.DataFrame({
        "time_ps": prod_topo["time_ps"].values,
        "n_F_contact": prod_topo["n_F_contact"].values,
        "n_cation_contact": prod_topo["n_cation_contact"].values,
        "n_F_contact_per_A2": prod_topo["n_F_contact"].values / area,
        "n_cation_contact_per_A2": prod_topo["n_cation_contact"].values / area,
    }))

    if HAVE_MPL:
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(bin_centers, rho_O, label="O")
            ax.plot(bin_centers, rho_H, label="H")
            if cation_sym:
                ax.plot(bin_centers, rho_cat, label=cation_sym)
            if anion_sym:
                ax.plot(bin_centers, rho_an, label=anion_sym)
            ax.set_xlabel("z - z(top Ag) [A]")
            ax.set_ylabel("rho [atoms/A^3]")
            ax.legend()
            ax.set_title(state_id)
            fig.tight_layout()
            fig.savefig(out_dir / "density_profile.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(bin_centers, dip_mean)
            ax.set_xlabel("z - z(top Ag) [A]")
            ax.set_ylabel("<cos theta> (H-O-H bisector vs +z)")
            ax.set_title(state_id)
            fig.tight_layout()
            fig.savefig(out_dir / "water_dipole_orientation.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(ftimes, dipole_proxy_series)
            ax.set_xlabel("time [ps]")
            ax.set_ylabel("dipole-proxy potential [V]")
            ax.set_title(state_id)
            fig.tight_layout()
            fig.savefig(out_dir / "dipole_proxy_timeseries.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover - plotting must never fail the run
            log.warning("[%s] plotting failed: %s", state_id, e)

    summary = {
        "state_id": state_id, "kind": "interface",
        "prod_ps_analyzed": float(ftimes[-1] - ftimes[0]) if len(ftimes) else 0.0,
        "T_mean": float(thermo.loc[thermo["time_ps"] >= ftimes[0], "T_K"].mean()) if len(ftimes) else float("nan"),
        "film_density_middle": float(np.mean(film_density_series)) if film_density_series else float("nan"),
        "n_F_contact_mean": float(prod_topo["n_F_contact"].mean()) if len(prod_topo) else float("nan"),
        "n_cation_contact_mean": float(prod_topo["n_cation_contact"].mean()) if len(prod_topo) else float("nan"),
        "bulk_conc_M": float(np.mean(bulk_conc_series)) if bulk_conc_series else float("nan"),
        "dipole_proxy_mean_V": float(np.mean(dipole_proxy_series)) if dipole_proxy_series else float("nan"),
        "dipole_proxy_std_V": float(np.std(dipole_proxy_series)) if dipole_proxy_series else float("nan"),
        "n_HF_max": int(prod_topo["n_HF"].max()) if len(prod_topo) else -1,
        "n_OH_max": int(prod_topo["n_OH-"].max()) if len(prod_topo) else -1,
        "n_H_on_Ag_max": int(prod_topo["n_H_on_Ag"].max()) if len(prod_topo) else -1,
        "n_desorbed_max": int(prod_topo["n_desorbed"].max()) if len(prod_topo) else -1,
    }
    return summary, dict(
        bins=bins, bin_centers=bin_centers, rho_cat=hist_cat, rho_an=hist_an,
        cation_sym=cation_sym, anion_sym=anion_sym, z_ref=z_ref,
    )


# --------------------------------------------------------------------------
# Bulk-box analysis
# --------------------------------------------------------------------------
def compute_msd(o_pos_series, cell, pbc):
    unwrapped = [o_pos_series[0].copy()]
    for k in range(1, len(o_pos_series)):
        raw = o_pos_series[k] - o_pos_series[k - 1]
        dvec, _ = find_mic(raw, cell, pbc)
        unwrapped.append(unwrapped[-1] + dvec)
    unwrapped = np.array(unwrapped)
    disp = unwrapped - unwrapped[0]
    return np.mean(np.sum(disp ** 2, axis=2), axis=1)


def compute_tetrahedral_q(pos_samples, box_diag):
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return float("nan")
    qs = []
    for pos in pos_samples:
        n = len(pos)
        if n < 5:
            continue
        wrapped = pos % box_diag
        tree = cKDTree(wrapped, boxsize=box_diag)
        _, idx = tree.query(wrapped, k=5)
        for i in range(n):
            vecs = []
            for j in idx[i, 1:5]:
                raw = pos[j] - pos[i]
                raw -= box_diag * np.round(raw / box_diag)
                nrm = np.linalg.norm(raw)
                if nrm < 1e-6:
                    continue
                vecs.append(raw / nrm)
            if len(vecs) < 4:
                continue
            s = 0.0
            for a in range(3):
                for b in range(a + 1, 4):
                    s += (np.dot(vecs[a], vecs[b]) + 1.0 / 3.0) ** 2
            qs.append(1.0 - (3.0 / 8.0) * s)
    return float(np.mean(qs)) if qs else float("nan")


def analyze_bulk(state_id, entry, d, thermo, topo, frames, ftimes, out_dir, args, log):
    cell = frames[0].get_cell()
    pbc = frames[0].get_pbc()
    vol_A3 = frames[0].get_volume()
    vol_cm3 = vol_A3 * 1e-24
    out_dir.mkdir(parents=True, exist_ok=True)

    masses = np.array([atoms.get_masses().sum() for atoms in frames])
    density = float(np.mean(masses) * AMU_TO_G / vol_cm3)

    o_pos_series = [atoms.get_positions()[np.array(atoms.get_chemical_symbols()) == "O"]
                     for atoms in frames]
    n_o = len(o_pos_series[0]) if o_pos_series else 0

    # O-O RDF
    dr = args.rdf_dr
    rmax = min(cell[0, 0], cell[1, 1], cell[2, 2]) / 2.0 * 0.95
    rbins = np.arange(0.0, rmax, dr)
    hist = np.zeros(len(rbins) - 1)
    if n_o >= 2:
        for pos in o_pos_series:
            o_atoms = Atoms("O" * n_o, positions=pos, cell=cell, pbc=pbc)
            _, _, dd = neighbor_list("ijd", o_atoms, rmax)
            hist += np.histogram(dd, bins=rbins)[0]
        rho_o = n_o / vol_A3
        shell_vol = (4.0 / 3.0) * np.pi * (rbins[1:] ** 3 - rbins[:-1] ** 3)
        g_r = hist / (len(frames) * n_o * rho_o * shell_vol)
    else:
        g_r = np.full(len(rbins) - 1, np.nan)
    r_centers = 0.5 * (rbins[1:] + rbins[:-1])

    r1 = g1 = float("nan")
    peak_region = (r_centers >= 2.2) & (r_centers <= 3.3)
    if peak_region.any() and np.isfinite(g_r[peak_region]).any():
        loc = np.nanargmax(g_r[peak_region])
        r1 = float(r_centers[peak_region][loc])
        g1 = float(g_r[peak_region][loc])

    split = False
    try:
        from scipy.signal import find_peaks
        second_region = (r_centers >= 3.8) & (r_centers <= 5.8)
        if second_region.sum() > 5 and np.isfinite(g_r[second_region]).all():
            pk, _ = find_peaks(g_r[second_region], prominence=0.03)
            split = len(pk) >= 2
    except ImportError:
        pass

    # O self-diffusion, Einstein relation on the fit window
    D_cm2_s = float("nan")
    if n_o >= 1 and len(ftimes) >= 3:
        msd = compute_msd(o_pos_series, cell, pbc)
        t_rel_ps = ftimes - ftimes[0]
        fit_mask = (t_rel_ps >= args.msd_fit_start_ps) & (t_rel_ps <= args.msd_fit_end_ps)
        if fit_mask.sum() >= 2:
            t_fit_fs = t_rel_ps[fit_mask] * 1000.0
            slope, _ = np.polyfit(t_fit_fs, msd[fit_mask], 1)  # A^2/fs
            D_cm2_s = float(slope / 6.0 * 1e-16 / 1e-15)  # -> cm^2/s
        write_csv(out_dir / "msd_O.csv", pd.DataFrame({"time_ps": ftimes, "msd_A2": msd}))

    # tetrahedral order parameter (subsample to keep this cheap)
    if np.allclose(cell, np.diag(np.diag(cell))) and n_o >= 5:
        idxs = np.linspace(0, len(o_pos_series) - 1, min(5, len(o_pos_series))).astype(int)
        q_mean = compute_tetrahedral_q([o_pos_series[i] for i in idxs], np.diag(cell))
    else:
        q_mean = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "OO_rdf.csv", pd.DataFrame({"r_A": r_centers, "g_r": g_r}))

    if HAVE_MPL:
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(r_centers, g_r)
            ax.set_xlabel("r [A]")
            ax.set_ylabel("g_OO(r)")
            ax.set_title(state_id)
            fig.tight_layout()
            fig.savefig(out_dir / "OO_rdf.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            log.warning("[%s] plotting failed: %s", state_id, e)

    summary = {
        "state_id": state_id, "kind": "bulk",
        "prod_ps_analyzed": float(ftimes[-1] - ftimes[0]) if len(ftimes) else 0.0,
        "T_mean": float(thermo.loc[thermo["time_ps"] >= ftimes[0], "T_K"].mean()) if len(ftimes) else float("nan"),
        "film_density_middle": float("nan"),
        "n_F_contact_mean": float("nan"), "n_cation_contact_mean": float("nan"),
        "bulk_conc_M": float("nan"),
        "dipole_proxy_mean_V": float("nan"), "dipole_proxy_std_V": float("nan"),
        "n_HF_max": int(topo["n_HF"].max()) if len(topo) else -1,
        "n_OH_max": int(topo["n_OH-"].max()) if len(topo) else -1,
        "n_H_on_Ag_max": int(topo["n_H_on_Ag"].max()) if len(topo) else -1,
        "n_desorbed_max": -1,
        "bulk_density_g_cm3": density, "OO_rdf_r1_A": r1, "OO_rdf_g1": g1,
        "O_self_diffusion_cm2_s": D_cm2_s, "freezing_second_peak_split": split,
        "tetrahedral_q_mean": q_mean,
    }
    return summary


# --------------------------------------------------------------------------
# Two-sided-start ("surface" vs "far") comparison
# --------------------------------------------------------------------------
def two_sided_comparison(entries, per_state_profile, md_dir, args, out_root, log):
    groups = defaultdict(lambda: defaultdict(list))
    for sid, e in entries.items():
        if e is None:
            continue
        ion_start = e.get("ion_start")
        if ion_start not in ("surface", "far"):
            continue
        key = (e.get("slab_layers"), e.get("electrolyte"), e.get("delta_N"))
        groups[key][ion_start].append(sid)

    rows = []
    for key, sides in groups.items():
        if "surface" not in sides or "far" not in sides:
            continue
        layers, electrolyte, dN = key
        surf_ids = [s for s in sides["surface"] if s in per_state_profile]
        far_ids = [s for s in sides["far"] if s in per_state_profile]
        if not surf_ids or not far_ids:
            continue

        ref = per_state_profile[surf_ids[0]]
        bins = ref["bins"]
        bin_centers = ref["bin_centers"]

        def pooled(ids, key_arr):
            tot = np.zeros(len(bin_centers))
            for sid in ids:
                p = per_state_profile[sid]
                if len(p[key_arr]) == len(tot):
                    tot += p[key_arr]
            s = tot.sum()
            return tot / s if s > 0 else tot

        surf_cat = pooled(surf_ids, "rho_cat")
        far_cat = pooled(far_ids, "rho_cat")
        surf_an = pooled(surf_ids, "rho_an")
        far_an = pooled(far_ids, "rho_an")
        l1_cat = float(np.sum(np.abs(surf_cat - far_cat)) * (bin_centers[1] - bin_centers[0]))
        l1_an = float(np.sum(np.abs(surf_an - far_an)) * (bin_centers[1] - bin_centers[0]))

        tag = f"L{layers}_{electrolyte}_dN{dN:+d}" if isinstance(dN, int) else f"L{layers}_{electrolyte}_dN{dN}"
        comp_dir = Path(out_root) / "_comparisons"
        comp_dir.mkdir(parents=True, exist_ok=True)
        write_csv(comp_dir / f"{tag}_ion_profiles.csv", pd.DataFrame({
            "z_rel_topAg_A": bin_centers,
            "rho_cation_surface_norm": surf_cat, "rho_cation_far_norm": far_cat,
            "rho_anion_surface_norm": surf_an, "rho_anion_far_norm": far_an,
        }))

        # equilibration time: excess-ion mean-z convergence between the two
        # start conditions, using the first replica of each side.
        excess_role = "cation" if (isinstance(dN, int) and dN < 0) else (
            "anion" if (isinstance(dN, int) and dN > 0) else None)
        t_conv = float("nan")
        if excess_role is not None:
            t_conv = estimate_equilibration_time(
                md_dir, surf_ids[0], far_ids[0], excess_role, args, log)

        if HAVE_MPL:
            try:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(bin_centers, surf_cat, label="cation, surface-start")
                ax.plot(bin_centers, far_cat, label="cation, far-start")
                ax.plot(bin_centers, surf_an, label="anion, surface-start", ls="--")
                ax.plot(bin_centers, far_an, label="anion, far-start", ls="--")
                ax.set_xlabel("z - z(top Ag) [A]")
                ax.set_ylabel("normalized density")
                ax.legend(fontsize=8)
                ax.set_title(tag)
                fig.tight_layout()
                fig.savefig(comp_dir / f"{tag}_ion_profiles.png", dpi=150)
                plt.close(fig)
            except Exception as e:  # pragma: no cover
                log.warning("comparison plot failed for %s: %s", tag, e)

        rows.append({
            "group": tag, "slab_layers": layers, "electrolyte": electrolyte,
            "delta_N": dN, "surface_states": ";".join(surf_ids),
            "far_states": ";".join(far_ids),
            "L1_diff_cation_profile": l1_cat, "L1_diff_anion_profile": l1_an,
            "equilibration_time_ps": t_conv,
        })
        log.info("Two-sided comparison %s: L1(cation)=%.4f L1(anion)=%.4f t_eq=%s ps",
                 tag, l1_cat, l1_an, t_conv)

    if rows:
        write_csv(Path(out_root) / "_comparisons" / "two_sided_comparison.csv", pd.DataFrame(rows))
    return rows


def estimate_equilibration_time(md_dir, sid_surface, sid_far, excess_role, args, log):
    try:
        d_s, thermo_s, _ = load_state_tables(md_dir, sid_surface)
        d_f, thermo_f, _ = load_state_tables(md_dir, sid_far)
        skip_s = resolve_skip_ps(args, d_s, thermo_s)
        skip_f = resolve_skip_ps(args, d_f, thermo_f)
        frames_s, times_s = load_prod_frames(d_s, thermo_s, skip_s)
        frames_f, times_f = load_prod_frames(d_f, thermo_f, skip_f)
    except Exception as e:
        log.warning("equilibration-time estimate failed to load trajectories: %s", e)
        return float("nan")

    def mean_z_series(frames):
        out = []
        for atoms in frames:
            role = np.array(atoms.arrays.get("role", []))
            z = atoms.get_positions()[:, 2]
            sel = role == excess_role
            out.append(float(z[sel].mean()) if sel.any() else float("nan"))
        return np.array(out)

    mz_s = mean_z_series(frames_s)
    mz_f = mean_z_series(frames_f)
    n = min(len(mz_s), len(mz_f))
    if n == 0:
        return float("nan")
    diff = np.abs(mz_s[:n] - mz_f[:n])
    below = np.where(diff < args.convergence_tol_A)[0]
    if len(below) == 0:
        return float("nan")
    t_common = times_s[:n]
    return float(t_common[below[0]] - t_common[0])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    log = logging.getLogger("analyze_md")

    if args.all:
        state_ids = discover_states(args.md_dir)
        if not state_ids:
            log.error("No states with traj.extxyz found under %s", args.md_dir)
            return
    else:
        state_ids = args.state_id

    manifest = mc.load_manifest(args.structures_dir)
    entries = {sid: manifest.get(sid) for sid in state_ids}

    summaries = []
    profiles = {}
    for sid in state_ids:
        d = Path(args.md_dir) / sid
        if not (d / "traj.extxyz").exists():
            log.error("Skipping %s: no traj.extxyz under %s", sid, d)
            continue
        try:
            d, thermo, topo = load_state_tables(args.md_dir, sid)
            skip_ps = resolve_skip_ps(args, d, thermo)
            frames, ftimes = load_prod_frames(d, thermo, skip_ps)
            if not frames:
                log.warning("[%s] no production frames (skip_ps=%.3f >= last frame time); skipping.",
                            sid, skip_ps)
                continue
            out_dir = d / "analysis"
            entry = entries.get(sid)
            is_bulk = sid.startswith("bulk_") or (entry and "n_Ag" in entry and entry.get("n_Ag", 1) == 0)
            if is_bulk:
                summary = analyze_bulk(sid, entry, d, thermo, topo, frames, ftimes, out_dir, args, log)
            else:
                summary, prof = analyze_interface(sid, entry, d, thermo, topo, frames, ftimes,
                                                    out_dir, args, log)
                profiles[sid] = prof
            summaries.append(summary)
            log.info("Analyzed %s (%s, %d production frames, %.2f ps)",
                     sid, "bulk" if is_bulk else "interface", len(frames),
                     summary["prod_ps_analyzed"])
        except Exception as e:
            log.exception("Failed to analyze %s: %s", sid, e)

    two_sided_comparison(entries, profiles, args.md_dir, args, args.md_dir, log)

    if summaries:
        df = pd.DataFrame(summaries)
        for f in SUMMARY_FIELDS:
            if f not in df.columns:
                df[f] = np.nan
        df = df[SUMMARY_FIELDS]
        write_csv(Path(args.md_dir) / "summary.csv", df)
        log.info("Wrote summary for %d states to %s", len(df), Path(args.md_dir) / "summary.csv")
    else:
        log.warning("No states successfully analyzed; summary.csv not written.")


if __name__ == "__main__":
    main()
