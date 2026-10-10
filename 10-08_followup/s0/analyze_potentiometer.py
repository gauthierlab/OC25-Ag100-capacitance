#!/usr/bin/env python3
"""
analyze_potentiometer.py
=========================

Post-process one (or all) RPBE-D3 single-point DFT jobs produced by make_vasp_inputs.py into
the S0 "potentiometer" readout: electrode potential U (inner-potential-referenced, primary;
vacuum work function, secondary) and actual metal charge sigma (three estimators).

Per job directory (must contain OUTCAR, LOCPOT, CHGCAR, ACF.dat, POSCAR or CONTCAR, meta.json)
writes <job-dir>/potentiometer.json, and with --all aggregates
<out-dir>/potentiometer_summary.csv (one row per job).

Sign convention throughout: electrode charge sigma is NEGATIVE when the metal has excess
electrons (e.g. from an ionized excess Na -> Na+ + e-). Potentials (U_inner, Phi_top,
Phi_bottom) are computed directly from VASP's own eV-scale quantities (E-fermi from OUTCAR,
planar-averaged LOCPOT with LVHAR=.TRUE., i.e. Hartree+ionic electrostatic potential only) as
  potential_relative_to_X = V_H(X, macro-averaged) - E_fermi
which is the standard "vacuum level minus Fermi level" work-function convention, applied here
also to the bulk-water reference (U_inner) instead of vacuum. No extra explicit division by e
is applied: VASP's OUTCAR/LOCPOT are both already on the eV=volt-for-e=1 scale used throughout
the computational-electrochemistry literature for this exact formula.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ELEMENTARY_CHARGE_C = 1.602176634e-19  # C (magnitude; sign handled explicitly in formulas)
ANGSTROM2_TO_CM2 = 1.0e-16

# Nominal PAW ZVAL (valence electron count) for the recommended ASE-default setups used by
# make_vasp_inputs.py. Must stay consistent with RECOMMENDED_SETUPS there.
ZVAL = {
    "Ag": 11,
    "O": 6,
    "H": 1,
    "Na": 9,   # Na_pv
    "Cs": 9,   # Cs_sv
    "F": 7,
}

F_CONTACT_CUTOFF_A = 2.9  # F within this distance of a top-layer Ag counts as contact-adsorbed


# ---------------------------------------------------------------------------
# Small VASP grid-file parser (LOCPOT / CHGCAR), self-contained: no dependence on
# ase.calculators.vasp.VaspChargeDensity so behaviour is transparent and testable on a
# synthetic file we generate ourselves.
# ---------------------------------------------------------------------------


def read_vasp_grid_file(path):
    """Parse a VASP LOCPOT/CHGCAR-format file.

    Returns dict with: cell (3x3 ndarray, Angstrom), symbols (list[str]), counts (list[int]),
    positions_direct (Nx3 ndarray), ngrid (nx,ny,nz), data (nx,ny,nz ndarray, first data set
    only -- sufficient for ISPIN=1 non-spin-polarized runs).

    VASP writes the grid with x fastest, then y, then z (flat index = ix + iy*nx + iz*nx*ny).
    CHGCAR-style files may be followed by "augmentation occupancies" blocks and, for ISPIN=2,
    a second spin-density grid; both are ignored here (not needed for any estimator used).
    """
    with open(path) as f:
        lines = f.readlines()

    idx = 0
    comment = lines[idx]; idx += 1
    scale = float(lines[idx].split()[0]); idx += 1
    cell = np.array([[float(x) for x in lines[idx + i].split()[:3]] for i in range(3)]) * scale
    idx += 3

    # VASP5: a line of element symbols before the counts line. Detect by trying to parse ints.
    tokens = lines[idx].split()
    try:
        counts = [int(t) for t in tokens]
        symbols_line = None
    except ValueError:
        symbols_line = tokens
        idx += 1
        counts = [int(t) for t in lines[idx].split()]
    idx += 1

    symbols = []
    if symbols_line:
        for sp, c in zip(symbols_line, counts):
            symbols.extend([sp] * c)
    natoms = sum(counts)

    mode_line = lines[idx].strip().lower(); idx += 1
    if mode_line.startswith("s"):  # "Selective dynamics"
        mode_line = lines[idx].strip().lower(); idx += 1
    is_direct = mode_line.startswith("d")

    positions = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = lines[idx + i].split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    idx += natoms

    positions_direct = positions if is_direct else None
    positions_cart = positions if not is_direct else None
    if positions_direct is not None:
        positions_cart = positions_direct @ cell
    else:
        positions_direct = positions_cart @ np.linalg.inv(cell)

    # blank line then grid dims
    while lines[idx].strip() == "":
        idx += 1
    nx, ny, nz = (int(v) for v in lines[idx].split()[:3]); idx += 1

    ntot = nx * ny * nz
    flat = []
    while len(flat) < ntot:
        flat.extend(float(v) for v in lines[idx].split())
        idx += 1
    flat = np.array(flat[:ntot])
    data = flat.reshape((nz, ny, nx)).transpose(2, 1, 0)  # -> (nx, ny, nz), x fastest as read

    return {
        "cell": cell,
        "symbols": symbols,
        "counts": counts,
        "positions_direct": positions_direct,
        "positions_cart": positions_cart,
        "ngrid": (nx, ny, nz),
        "data": data,
    }


def planar_average_z(data):
    """Mean over x,y for each z-plane. data: (nx,ny,nz) -> (nz,) array."""
    return data.mean(axis=(0, 1))


def macro_average(profile, z_len, window):
    """Periodic (circular) boxcar (moving) average of a 1-D profile sampled on a uniform,
    periodic z-grid spanning z_len, with window width `window` (Angstrom)."""
    n = len(profile)
    dz = z_len / n
    w = max(1, int(round(window / dz)))
    kernel = np.ones(w) / w
    padded = np.concatenate([profile[-w:], profile, profile[:w]])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[w:-w]


def electron_count_cumulative(planar_avg_chg, nz):
    """N(< iz) via VASP's CHGCAR convention (grid value = rho*Vcell): cumulative sum of the
    planar average divided by nz gives the number of electrons below each z-plane boundary."""
    return np.cumsum(planar_avg_chg) / nz


# ---------------------------------------------------------------------------
# OUTCAR / ACF.dat / meta.json / structure parsing
# ---------------------------------------------------------------------------


def parse_efermi(outcar_path):
    pattern = re.compile(r"E-fermi\s*:\s*([-\d.]+)")
    efermi = None
    with open(outcar_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                efermi = float(m.group(1))  # keep last occurrence (final SCF step)
    if efermi is None:
        raise ValueError(f"Could not find 'E-fermi' in {outcar_path}")
    return efermi


def parse_total_energy(outcar_path):
    pattern = re.compile(r"free  energy   TOTEN\s*=\s*([-\d.]+)")
    energy = None
    with open(outcar_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                energy = float(m.group(1))
    return energy


def parse_acf(acf_path):
    """Parse Henkelman bader-code ACF.dat. Returns list of dicts with index, x,y,z, charge
    (electron count assigned by Bader partitioning), min_dist, vol."""
    rows = []
    with open(acf_path) as f:
        lines = f.readlines()
    data_lines = [l for l in lines if l.strip() and l.strip()[0].isdigit()]
    for line in data_lines:
        parts = line.split()
        rows.append(
            {
                "index": int(parts[0]) - 1,  # ACF.dat is 1-indexed
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
                "charge": float(parts[4]),
                "min_dist": float(parts[5]) if len(parts) > 5 else None,
                "vol": float(parts[6]) if len(parts) > 6 else None,
            }
        )
    return rows


def read_poscar_like(path):
    """Minimal POSCAR/CONTCAR reader returning cell, symbols (expanded per-atom), positions
    (Cartesian, Angstrom)."""
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].split()[0])
    cell = np.array([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)]) * scale
    tokens = lines[5].split()
    try:
        counts = [int(t) for t in tokens]
        idx = 6
        symbols_line = None
    except ValueError:
        symbols_line = tokens
        counts = [int(t) for t in lines[6].split()]
        idx = 7
    symbols = []
    if symbols_line:
        for sp, c in zip(symbols_line, counts):
            symbols.extend([sp] * c)
    mode = lines[idx].strip().lower()
    has_sd = mode.startswith("s")
    if has_sd:
        idx += 1
        mode = lines[idx].strip().lower()
    idx += 1
    is_direct = mode.startswith("d")
    natoms = sum(counts)
    positions = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = lines[idx + i].split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    if is_direct:
        positions = positions @ cell
    return cell, symbols, positions


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def find_top_ag_z(symbols, positions, meta):
    ag_z = [p[2] for s, p in zip(symbols, positions) if s == "Ag"]
    if ag_z:
        return max(ag_z)
    if meta.get("z_top_Ag") is not None:
        return meta["z_top_Ag"]
    raise ValueError("Cannot determine z_top_Ag: no Ag atoms and no meta.json fallback")


def compute_potentials(locpot, efermi, meta, ref_width, macro_window):
    z_len = locpot["cell"][2, 2]
    profile = planar_average_z(locpot["data"])
    nz = len(profile)
    macro = macro_average(profile, z_len, macro_window)
    z_grid = (np.arange(nz) + 0.5) * (z_len / nz)

    symbols = locpot["symbols"] if locpot["symbols"] else None
    if not symbols:
        symbols = []
    z_top_Ag = find_top_ag_z(symbols, locpot["positions_cart"], meta) if symbols else meta.get("z_top_Ag")
    if z_top_Ag is None:
        raise ValueError("z_top_Ag unavailable from structure or meta.json")

    film = meta.get("film_thickness")
    if film is None:
        # fall back: use the water/ion atoms' z-extent above the slab if available; else assume
        # the bulk-water window is centered halfway between topmost Ag and the highest atom.
        z_all = locpot["positions_cart"][:, 2]
        film = float(np.max(z_all) - z_top_Ag)

    z_center = z_top_Ag + film / 2.0
    lo, hi = z_center - ref_width / 2.0, z_center + ref_width / 2.0
    in_window = (z_grid >= lo) & (z_grid <= hi)
    if not np.any(in_window):
        raise ValueError(f"ref-width window [{lo:.2f},{hi:.2f}] A contains no grid points (z_len={z_len:.2f})")
    V_bulk_mean = float(np.mean(macro[in_window]))
    V_bulk_std = float(np.std(macro[in_window]))

    # Vacuum plateau(s). The slabs are single-sided, so there is ONE vacuum span per cell: the
    # largest periodic gap in atomic z, running from above the water film to below the bottom Ag
    # layer. It may cross the cell boundary: make_vasp_inputs.py centers the atoms so that the
    # DIPOL = 0.5 correction step sits at the boundary, mid-vacuum; older uncentered cells had the
    # bottom Ag layer at z~0 and the whole span below z_len. Work in an unwrapped coordinate u
    # along the span. "Top" is the near-film half and "bottom" the half facing the bottom Ag
    # layer. For a converged IDIPOL=3/LDIPOL=.TRUE. correction each half is flat and
    # dipole_step = Phi_top - Phi_bottom equals the correction step.
    fz = np.sort(np.mod(locpot["positions_cart"][:, 2], z_len))
    gaps = np.diff(np.concatenate([fz, [fz[0] + z_len]]))
    ig = int(np.argmax(gaps))
    gap_lo, gap_hi = fz[ig], fz[ig] + gaps[ig]  # unwrapped; gap_hi may exceed z_len
    u_grid = np.where(z_grid < gap_lo, z_grid + z_len, z_grid)
    # The vacuum plateaus are read from the RAW planar average, not the macro average: there are no
    # atoms in vacuum, so there is nothing to smooth, and the boxcar would otherwise drag the film's
    # (or the metal's) potential up to macro_window/2 into the plateau and smear the dipole step.
    # 3 A clearance from the outermost atom on each side clears the electron density tail.
    vac_lo = gap_lo + 3.0
    vac_hi = gap_hi - 3.0
    # Split at the actual correction step. With DIPOL = 0.5 the step sits at the cell boundary
    # (u = z_len in the unwrapped coordinate), whether or not the atoms are centered. A stray
    # molecule in the vacuum changes the largest gap and can leave the step outside it; that
    # snapshot is flagged (step_in_gap = False) instead of being read across the step.
    u_step = z_len
    step_in_gap = bool(gap_lo < u_step < gap_hi)
    if not step_in_gap:
        raise ValueError(f"dipole step (u={u_step:.2f} A) not inside the vacuum gap [{gap_lo:.2f},{gap_hi:.2f}] A: "
                         "stray atom in vacuum or uncentered cell; inspect this snapshot")
    buf = 0.5
    top_mask = (u_grid >= vac_lo) & (u_grid <= u_step - buf)
    bot_mask = (u_grid >= u_step + buf) & (u_grid <= vac_hi)

    if not np.any(top_mask):
        raise ValueError("No grid points in top (near-film) vacuum plateau region (film/vacuum too thin for ref check)")
    Phi_top = float(profile[top_mask].mean() - efermi)
    top_slope = float(np.polyfit(u_grid[top_mask], profile[top_mask], 1)[0]) if top_mask.sum() >= 2 else float("nan")

    if np.any(bot_mask):
        Phi_bottom = float(profile[bot_mask].mean() - efermi)
        bottom_slope = float(np.polyfit(u_grid[bot_mask], profile[bot_mask], 1)[0]) if bot_mask.sum() >= 2 else float("nan")
    else:
        Phi_bottom = None
        bottom_slope = None

    U_inner = V_bulk_mean - efermi

    dipole_step = (Phi_top - Phi_bottom) if Phi_bottom is not None else None

    return {
        "U_inner_V": U_inner,
        "U_inner_ref_noise_V": V_bulk_std,
        "Phi_top_eV": Phi_top,
        "Phi_top_slope_eV_per_A": top_slope,
        "Phi_bottom_eV": Phi_bottom,
        "Phi_bottom_slope_eV_per_A": bottom_slope,
        "dipole_step_eV": dipole_step,
        "vacuum_gap_A": float(gap_hi - gap_lo),
        "top_plateau_width_A": float(u_step - buf - vac_lo),
        "z_top_Ag": z_top_Ag,
        "film_thickness_A": film,
    }


def compute_sigma_a(acf_rows, symbols, area_A2):
    """Ion-Bader estimator: sigma_a = -(sum of ionic charges on cation/anion atoms) * e / A.
    Ionic charge q_i = Zval_i - N_bader_i (standard Bader convention: positive = cation).
    Water is treated as neutral (excluded)."""
    total_q = 0.0
    for row, sym in zip(acf_rows, symbols):
        if sym in ("Na", "Cs", "F"):
            zval = ZVAL[sym]
            q = zval - row["charge"]
            total_q += q
    area_cm2 = area_A2 * ANGSTROM2_TO_CM2
    sigma_a = -(total_q * ELEMENTARY_CHARGE_C) / area_cm2 * 1e6  # uC/cm^2
    return sigma_a


def compute_sigma_b(acf_rows, symbols, area_A2):
    """Slab-Bader estimator: direct sum of Bader ionic charge over Ag atoms (no extra sign
    flip -- see README_DFT.md 'sign conventions' section for the derivation). This differs
    from the literal spec formula sigma_b = -(sum(Zval-Nbader)) * e / A by omitting the
    leading minus sign: that literal form was verified (by hand and against the synthetic
    test) to give the WRONG sign relative to the stated convention (sigma negative when the
    metal has excess electrons). The Bader charge of an atom, Zval - N_bader, is already the
    atom's net ionic charge in units of e with the conventional sign (positive = lost
    electrons); summing it directly over the slab gives the slab's own net charge with no
    additional inference step (unlike estimator a, which infers the metal's charge from the
    ions via overall charge neutrality and therefore needs the extra minus sign)."""
    total_q = sum(ZVAL["Ag"] - row["charge"] for row, sym in zip(acf_rows, symbols) if sym == "Ag")
    area_cm2 = area_A2 * ANGSTROM2_TO_CM2
    sigma_b = (total_q * ELEMENTARY_CHARGE_C) / area_cm2 * 1e6  # uC/cm^2
    return sigma_b


def compute_sigma_c(chgcar, symbols, meta):
    """Density-integral estimator. Reference: cell bottom (z=0) up to the electron-density
    minimum between the top Ag layer and the first water layer. Nuclear charge in that region
    is the sum of ZVAL over atoms (treated as point/delta charges at their z-position) with
    z below that minimum; electron count is CHGCAR's planar-averaged density integrated over
    the same span (VASP convention: grid value = rho*Vcell, so N(<z) = cumsum(planar_avg)/nz).
    sigma_c = (nuclear_charge_in_region - electron_count_in_region) * e / A, i.e. positive
    when the region has a net deficit of electrons (consistent with sigma_a/b: negative when
    the metal side has excess electrons)."""
    data = chgcar["data"]
    nz = data.shape[2]
    z_len = chgcar["cell"][2, 2]
    planar = planar_average_z(data)
    cum_N = electron_count_cumulative(planar, nz)
    z_grid = (np.arange(nz) + 0.5) * (z_len / nz)

    positions = chgcar["positions_cart"]
    ag_z = [p[2] for s, p in zip(symbols, positions) if s == "Ag"]
    if not ag_z:
        raise ValueError("No Ag atoms found for sigma_c reference search")
    z_top_Ag = max(ag_z)
    water_o_z = [p[2] for s, p in zip(symbols, positions) if s == "O" and p[2] > z_top_Ag]
    z_first_water = min(water_o_z) if water_o_z else z_top_Ag + 3.0

    search_mask = (z_grid >= z_top_Ag) & (z_grid <= z_first_water)
    if not np.any(search_mask):
        # fall back to a fixed window just above the slab
        search_mask = (z_grid >= z_top_Ag) & (z_grid <= z_top_Ag + 3.0)
    search_idx = np.where(search_mask)[0]
    if len(search_idx) == 0:
        raise ValueError("Could not locate a density-minimum search window for sigma_c")
    local_min_idx = search_idx[np.argmin(planar[search_idx])]
    z_min = z_grid[local_min_idx]

    below = positions[:, 2] < z_min
    nuclear_charge = sum(ZVAL[s] for s, b in zip(symbols, below) if b)
    n_electrons = cum_N[local_min_idx]

    area_cm2 = meta["area_A2"] * ANGSTROM2_TO_CM2
    sigma_c = ((nuclear_charge - n_electrons) * ELEMENTARY_CHARGE_C) / area_cm2 * 1e6  # uC/cm^2
    return sigma_c, z_min


def contact_adsorbed_fluoride(symbols, positions, cutoff=F_CONTACT_CUTOFF_A):
    """F atoms within `cutoff` Angstrom (in-plane+z, minimum-image not applied: fine for a
    single top surface check) of a top-Ag-layer atom. Returns list of F indices."""
    ag_idx = [i for i, s in enumerate(symbols) if s == "Ag"]
    if not ag_idx:
        return []
    ag_z = positions[ag_idx, 2]
    top_ag_z = ag_z.max()
    top_ag_idx = [i for i in ag_idx if positions[i, 2] >= top_ag_z - 0.5]
    f_idx = [i for i, s in enumerate(symbols) if s == "F"]
    contact = []
    for fi in f_idx:
        d = np.linalg.norm(positions[top_ag_idx] - positions[fi], axis=1)
        if d.min() <= cutoff:
            contact.append(fi)
    return contact


def mlip_force_comparison(job_dir, mlip_forces_path, dft_forces, dft_symbols, dft_positions):
    from ase.io import read as ase_read

    frame = ase_read(mlip_forces_path, index=0)
    if "forces" in frame.arrays:
        mlip_f = frame.arrays["forces"]
    elif frame.calc is not None and frame.calc.results.get("forces") is not None:
        mlip_f = frame.calc.results["forces"]
    else:
        return None

    if len(mlip_f) != len(dft_forces):
        return {"error": f"atom count mismatch: mlip={len(mlip_f)} dft={len(dft_forces)}"}

    diff = mlip_f - dft_forces
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    e_mlip = None
    if frame.calc is not None:
        try:
            e_mlip = float(frame.calc.get_potential_energy())
        except Exception:
            e_mlip = frame.info.get("energy")
    else:
        e_mlip = frame.info.get("energy")

    return {"force_rmse_eV_per_A": rmse, "force_mae_eV_per_A": mae, "mlip_energy_eV": e_mlip}


def parse_forces_from_outcar(outcar_path, natoms):
    """Parse the final 'TOTAL-FORCE' block from OUTCAR."""
    with open(outcar_path) as f:
        lines = f.readlines()
    last_start = None
    for i, line in enumerate(lines):
        if "TOTAL-FORCE" in line:
            last_start = i
    if last_start is None:
        return None
    forces = []
    i = last_start + 2  # skip header + dashed line
    while i < len(lines) and len(forces) < natoms:
        parts = lines[i].split()
        if len(parts) < 6:
            break
        forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        i += 1
    return np.array(forces) if len(forces) == natoms else None


# ---------------------------------------------------------------------------
# Per-job driver
# ---------------------------------------------------------------------------


def analyze_job(job_dir, ref_width, macro_window, mlip_forces_path=None):
    meta_path = os.path.join(job_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    outcar = os.path.join(job_dir, "OUTCAR")
    locpot_path = os.path.join(job_dir, "LOCPOT")
    chgcar_path = os.path.join(job_dir, "CHGCAR")
    acf_path = os.path.join(job_dir, "ACF.dat")

    struct_path = os.path.join(job_dir, "CONTCAR")
    if not os.path.exists(struct_path):
        struct_path = os.path.join(job_dir, "POSCAR")
    cell, symbols, positions = read_poscar_like(struct_path)

    efermi = parse_efermi(outcar)
    e_total = parse_total_energy(outcar)

    locpot = read_vasp_grid_file(locpot_path)
    if not locpot["symbols"]:
        locpot["symbols"] = symbols
        locpot["positions_cart"] = positions
    pot = compute_potentials(locpot, efermi, meta, ref_width, macro_window)

    acf_rows = parse_acf(acf_path)
    if len(acf_rows) != len(symbols):
        raise ValueError(f"ACF.dat has {len(acf_rows)} atoms but structure has {len(symbols)}")

    area_A2 = meta.get("area_A2")
    if area_A2 is None:
        area_A2 = abs(np.linalg.det(cell[:2, :2])) if cell.shape[0] >= 2 else None
        if area_A2 is None:
            raise ValueError("area_A2 not in meta.json and could not be derived from cell")

    sigma_a = compute_sigma_a(acf_rows, symbols, area_A2)
    sigma_b = compute_sigma_b(acf_rows, symbols, area_A2)

    chgcar = read_vasp_grid_file(chgcar_path)
    if not chgcar["symbols"]:
        chgcar["symbols"] = symbols
        chgcar["positions_cart"] = positions
    sigma_c, z_min_c = compute_sigma_c(chgcar, symbols, meta)

    per_ion_charges = {}
    for sp in ("Na", "Cs", "F"):
        qs = [ZVAL[sp] - row["charge"] for row, sym in zip(acf_rows, symbols) if sym == sp]
        if qs:
            per_ion_charges[sp] = {"mean": float(np.mean(qs)), "min": float(np.min(qs)), "n": len(qs), "values": [float(q) for q in qs]}

    min_ion_ionization = None
    if per_ion_charges:
        all_abs = [abs(q) for d in per_ion_charges.values() for q in d["values"]]
        min_ion_ionization = float(min(all_abs)) if all_abs else None

    positions_arr = np.array(positions)
    contact_idx = contact_adsorbed_fluoride(symbols, positions_arr)
    f_contact_charges = [ZVAL["F"] - acf_rows[i]["charge"] for i in contact_idx]  # signed ionic charge; ~-1 = fully ionic F-
    n_F_contact = len(contact_idx)
    f_contact_charge_mean = float(np.mean(f_contact_charges)) if f_contact_charges else None

    force_info = None
    if mlip_forces_path:
        dft_forces = parse_forces_from_outcar(outcar, len(symbols))
        if dft_forces is not None:
            force_info = mlip_force_comparison(job_dir, mlip_forces_path, dft_forces, symbols, positions_arr)

    sigma_nominal = meta.get("sigma_nominal_uC_cm2")

    result = {
        "state_id": meta.get("state_id"),
        "frame": meta.get("frame"),
        "variant": meta.get("variant"),
        "delta_N": meta.get("delta_N"),
        "electrolyte": meta.get("electrolyte"),
        "slab_layers_md": meta.get("slab_layers_md"),
        "slab_layers_dft": meta.get("slab_layers_dft"),
        "sigma_nominal_uC_cm2": sigma_nominal,
        "sigma_a_uC_cm2": sigma_a,
        "sigma_b_uC_cm2": sigma_b,
        "sigma_c_uC_cm2": sigma_c,
        "sigma_c_z_min_A": z_min_c,
        "U_inner_V": pot["U_inner_V"],
        "U_inner_ref_noise_V": pot["U_inner_ref_noise_V"],
        "Phi_top_eV": pot["Phi_top_eV"],
        "Phi_top_slope_eV_per_A": pot["Phi_top_slope_eV_per_A"],
        "Phi_bottom_eV": pot["Phi_bottom_eV"],
        "Phi_bottom_slope_eV_per_A": pot["Phi_bottom_slope_eV_per_A"],
        "dipole_step_eV": pot["dipole_step_eV"],
        "vacuum_gap_A": pot["vacuum_gap_A"],
        "top_plateau_width_A": pot["top_plateau_width_A"],
        "EF_eV": efermi,
        "E_total_eV": e_total,
        "per_ion_bader_charge": per_ion_charges,
        "min_ion_ionization": min_ion_ionization,
        "n_F_contact": n_F_contact,
        "F_contact_charge_mean": f_contact_charge_mean,
        "force_comparison": force_info,
        "job_dir": job_dir,
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--job-dir", default=None)
    g.add_argument("--all", action="store_true")
    p.add_argument("--jobs-file", default=None, help="default: <out-dir>/jobs.txt, out-dir inferred from --job-dir's grandparent or ROOT/s0/dft")
    p.add_argument("--out-dir", default=None, help="where potentiometer_summary.csv is written (default: ROOT/s0/dft)")
    p.add_argument("--ref-width", type=float, default=8.0)
    p.add_argument("--macro-window", type=float, default=3.0)
    p.add_argument("--mlip-forces", default=None, help="extxyz path with MLIP forces/energy to diff against (single job only)")
    return p


CSV_FIELDS = [
    "state_id", "frame", "variant", "delta_N", "sigma_nominal_uC_cm2",
    "sigma_a_uC_cm2", "sigma_b_uC_cm2", "sigma_c_uC_cm2",
    "U_inner_V", "U_inner_ref_noise_V", "Phi_top_eV", "Phi_bottom_eV", "dipole_step_eV",
    "Phi_top_slope_eV_per_A", "Phi_bottom_slope_eV_per_A", "vacuum_gap_A", "top_plateau_width_A",
    "EF_eV", "min_ion_ionization", "n_F_contact", "F_contact_charge_mean", "E_total_eV",
    "force_MAE_if_available",
    # Added beyond the literal column list in the task spec: summarize_s0_dft.py's required
    # groupby is (slab_layers_md, slab_layers_dft, electrolyte, delta_N), which is impossible
    # without these columns in the CSV.
    "slab_layers_md", "slab_layers_dft", "electrolyte",
]


def to_csv_row(r):
    row = {k: r.get(k) for k in CSV_FIELDS if k != "force_MAE_if_available"}
    fc = r.get("force_comparison")
    row["force_MAE_if_available"] = fc.get("force_mae_eV_per_A") if fc else None
    return row


def main(argv=None):
    args = build_parser().parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, "dft")

    if args.job_dir:
        job_dirs = [args.job_dir]
    else:
        jobs_file = args.jobs_file or os.path.join(out_dir, "jobs.txt")
        with open(jobs_file) as f:
            job_dirs = [l.strip() for l in f if l.strip()]

    results = []
    for jd in job_dirs:
        try:
            r = analyze_job(jd, args.ref_width, args.macro_window, mlip_forces_path=args.mlip_forces if args.job_dir else None)
        except Exception as e:
            print(f"FAILED {jd}: {e}", file=sys.stderr)
            continue
        with open(os.path.join(jd, "potentiometer.json"), "w") as f:
            json.dump(r, f, indent=2)
        results.append(r)
        print(f"OK {jd}: U_inner={r['U_inner_V']:.4f} V  sigma_a={r['sigma_a_uC_cm2']:.2f} uC/cm2")

    if args.all or len(job_dirs) > 1:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "potentiometer_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for r in results:
                writer.writerow(to_csv_row(r))
        print(f"Wrote {len(results)} rows to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
