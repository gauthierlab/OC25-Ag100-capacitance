#!/usr/bin/env python3
"""
make_vasp_inputs.py
====================

S0 potentiometer: generate RPBE-D3 VASP single-point inputs from MD (OC25) snapshots,
per s0/INTERFACE_SPEC.md.

Reads:
  - s0/structures/manifest.json           (state metadata; written by build_s0_structures.py)
  - s0/structures/<state_id>/init.extxyz  (only with --from-init)
  - s0/md/<state_id>/snapshots/frame_*.extxyz  (per-atom arrays 'fixed', 'role'; written by run_md_oc25.py)

Writes, under <out-dir>/<state_id>/<frame>/native/ (ext8/ only with the retired --ext8 flag):
  POSCAR, INCAR, KPOINTS, meta.json, POTCAR_ORDER.txt
and <out-dir>/jobs.txt (one job directory per line, for the SLURM array).

Does NOT ship POTCARs (license). POTCAR_ORDER.txt names the required species order and the
recommended PAW setups so a submission script can `cat` them together on the cluster.

Species order in every POSCAR: Ag O H <cation> F, where <cation> is Na or Cs depending on the
state's electrolyte (species with zero count are omitted, as VASP requires nonzero counts).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from ase.io import read

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RPBE-D3(zero damping, IVDW=11)-relaxed Ag lattice constant (bulk fcc), read from
# ag_rpbe_d3zero_CONTCAR in ROOT (Quanah run 2026-09-17, ENCUT 500, 12x12x12, ISIF 3):
# primitive-cell vectors are permutations of (0, a/2, a/2) with magnitude component
# 2.0808459533273047 -> a = 4.161691906654609 A. The earlier 3.8436 A value came from
# IVDW=12 (BJ damping) and is superseded; OC25 used IVDW=11. Used only as a fallback when a
# state's manifest entry has no lattice_constant_A field.
FALLBACK_RPBE_D3_A_AG = 4.161691906654609

# Species priority order for POSCAR / POTCAR concatenation (per task spec: Ag O H Na|Cs F).
SPECIES_PRIORITY = ["Ag", "O", "H", "Na", "Cs", "F"]

# Recommended ASE-default PAW_PBE setups (task-specified) and their nominal ZVAL
# (valence electron counts), used later by analyze_potentiometer.py for Bader-charge
# referencing. Kept here too (duplicated in POTCAR_ORDER.txt) so the two scripts agree.
RECOMMENDED_SETUPS = {
    "Ag": ("Ag", 11),
    "O": ("O", 6),
    "H": ("H", 1),
    "Na": ("Na_pv", 9),
    "Cs": ("Cs_sv", 9),
    "F": ("F", 7),
}

ELECTROLYTE_CATION = {
    "NaF": "Na",
    "NaF_bulk": "Na",
    "CsF": "Cs",
    "H2O": None,
    # counter-ion-only sets (LC/SL, 2026-10-08)
    "F_ci": None,
    "Na_ci": "Na",
    "Cs_ci": "Cs",
    "CsF_bulk": "Cs",
}

# ---------------------------------------------------------------------------
# Manifest / state handling
# ---------------------------------------------------------------------------


def load_manifest(structures_dir):
    path = os.path.join(structures_dir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        entries = json.load(f)
    return {e["state_id"]: e for e in entries}


def cation_symbol_for_state(entry, atoms=None):
    """Cation element symbol for this state (Na or Cs), from manifest electrolyte field,
    falling back to inspecting the atoms' role=='cation' symbols if manifest is absent/unknown."""
    electrolyte = (entry or {}).get("electrolyte")
    if electrolyte in ELECTROLYTE_CATION:
        return ELECTROLYTE_CATION[electrolyte]
    if atoms is not None and "role" in atoms.arrays:
        roles = atoms.arrays["role"]
        syms = np.array(atoms.get_chemical_symbols())
        cat_syms = set(syms[roles == "cation"])
        cat_syms -= {""}
        if len(cat_syms) == 1:
            return next(iter(cat_syms))
    return None


def species_order_for_atoms(atoms, cation_symbol):
    present = set(atoms.get_chemical_symbols())
    order = [s for s in SPECIES_PRIORITY if s in present]
    if cation_symbol and cation_symbol not in order and cation_symbol in present:
        order.append(cation_symbol)
    leftover = sorted(present - set(order))
    order += leftover
    return order


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------


def discover_frames(state_id, args):
    """Return list of (frame_name, Atoms) for a state, honoring --from-init or --frames."""
    if args.from_init:
        init_path = os.path.join(args.structures_dir, state_id, "init.extxyz")
        if not os.path.exists(init_path):
            raise FileNotFoundError(f"--from-init given but no init.extxyz for {state_id}: {init_path}")
        atoms = read(init_path, index=0)
        return [("init", atoms)]

    snap_dir = os.path.join(args.md_dir, state_id, "snapshots")
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"No MD snapshots dir for {state_id}: {snap_dir}")

    all_frames = sorted(glob.glob(os.path.join(snap_dir, "frame_*.extxyz")))
    if not all_frames:
        raise FileNotFoundError(f"No frame_*.extxyz files found in {snap_dir}")

    spec = args.frames
    m = re.match(r"^last:(\d+)$", spec.strip()) if spec else None
    if m:
        n = int(m.group(1))
        chosen = all_frames[-n:]
    elif spec:
        chosen = sorted(glob.glob(os.path.join(snap_dir, spec)))
        if not chosen:
            # allow a bare glob relative to snap_dir even if user passed an absolute-looking one
            chosen = sorted(glob.glob(spec))
        if not chosen:
            raise FileNotFoundError(f"--frames pattern '{spec}' matched nothing in {snap_dir}")
    else:
        chosen = all_frames

    out = []
    for path in chosen:
        name = os.path.splitext(os.path.basename(path))[0]  # frame_00000012
        atoms = read(path, index=0)
        out.append((name, atoms))
    return out


def all_state_ids(args, manifest):
    ids = set(manifest.keys())
    if os.path.isdir(args.md_dir):
        ids |= {
            d for d in os.listdir(args.md_dir)
            if os.path.isdir(os.path.join(args.md_dir, d))
        }
    if os.path.isdir(args.structures_dir):
        ids |= {
            d for d in os.listdir(args.structures_dir)
            if os.path.isdir(os.path.join(args.structures_dir, d))
        }
    return sorted(ids)


# ---------------------------------------------------------------------------
# POSCAR writer (explicit, does not rely on ASE constraint machinery)
# ---------------------------------------------------------------------------


def write_poscar(atoms, fixed, species_order, path, comment):
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    cell = np.array(atoms.get_cell())

    order_idx = []
    for sp in species_order:
        order_idx.extend(np.where(symbols == sp)[0].tolist())
    if len(order_idx) != len(atoms):
        missing = set(range(len(atoms))) - set(order_idx)
        raise ValueError(
            f"species_order {species_order} does not cover all atoms; "
            f"unaccounted symbols: {sorted(set(symbols[list(missing)]))}"
        )

    counts = [int(np.sum(symbols == sp)) for sp in species_order]
    keep = [(sp, c) for sp, c in zip(species_order, counts) if c > 0]

    pos_sorted = positions[order_idx]
    fixed_sorted = np.asarray(fixed)[order_idx]

    with open(path, "w") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for v in cell:
            f.write(f"  {v[0]:.16f}  {v[1]:.16f}  {v[2]:.16f}\n")
        f.write("  " + "  ".join(sp for sp, _ in keep) + "\n")
        f.write("  " + "  ".join(str(c) for _, c in keep) + "\n")
        f.write("Selective dynamics\n")
        f.write("Cartesian\n")
        for pos, fx in zip(pos_sorted, fixed_sorted):
            flag = "F F F" if int(fx) == 1 else "T T T"
            f.write(f"  {pos[0]:.16f}  {pos[1]:.16f}  {pos[2]:.16f}   {flag}\n")


# ---------------------------------------------------------------------------
# ext8 slab extension
# ---------------------------------------------------------------------------


def cluster_layers(z_values, tol=0.5):
    """Cluster 1-D z-coordinates into layers; returns list of (mean_z, index_array), sorted by z."""
    order = np.argsort(z_values)
    z_sorted = z_values[order]
    groups = []
    current = [order[0]]
    for i in range(1, len(z_sorted)):
        if z_sorted[i] - z_sorted[i - 1] > tol:
            groups.append(current)
            current = []
        current.append(order[i])
    groups.append(current)
    out = [(float(np.mean(z_values[g])), np.array(g)) for g in groups]
    out.sort(key=lambda t: t[0])
    return out


def extend_to_8_layers(atoms, entry, target_layers=8):
    """Append bulk Ag(100) layers underneath the slab so the DFT cell has `target_layers`
    Ag layers total. New layers (and the pre-existing fixed layers) are all marked fixed.
    Free (MD-relaxed) layers keep their MD positions, only translated up in z. The cell is
    grown in z by exactly the added thickness so the vacuum region above the film is
    unchanged.

    Registry: fcc(100) stacks ABAB with successive layers offset by half the surface square
    lattice constant s = a/sqrt(2) in both x and y (verified against ase.build.fcc100).
    """
    slab_layers_md = int(entry.get("slab_layers"))
    n_add = target_layers - slab_layers_md
    if n_add <= 0:
        return None, slab_layers_md  # already at/above target; caller should skip ext8

    a = entry.get("lattice_constant_A") or FALLBACK_RPBE_D3_A_AG
    d = a / 2.0  # (100) interlayer spacing
    s = a / np.sqrt(2.0)  # surface square-lattice constant
    offset = s / 2.0

    roles = atoms.arrays.get("role")
    if roles is None:
        raise ValueError("atoms missing per-atom 'role' array; cannot identify slab atoms")
    symbols = np.array(atoms.get_chemical_symbols())
    is_slab = roles == "slab"
    if not np.any(is_slab):
        raise ValueError("no atoms with role=='slab' found")

    slab_idx = np.where(is_slab)[0]
    slab_z = atoms.get_positions()[slab_idx, 2]
    layers = cluster_layers(slab_z, tol=d / 2.0)
    if len(layers) != slab_layers_md:
        print(
            f"  WARNING: manifest says slab_layers={slab_layers_md} but clustering found "
            f"{len(layers)} Ag layers (tol={d/2.0:.2f} A); proceeding with clustering result.",
            file=sys.stderr,
        )
        slab_layers_md = len(layers)
        n_add = target_layers - slab_layers_md
        if n_add <= 0:
            return None, slab_layers_md

    bottom_mean_z, bottom_group_local = layers[0]
    bottom_global_idx = slab_idx[bottom_group_local]
    bottom_xy = atoms.get_positions()[bottom_global_idx][:, :2]
    n_per_layer = len(bottom_global_idx)

    cell = np.array(atoms.get_cell())
    if not (np.allclose(cell[0, 1:], 0) and np.allclose(cell[1, [0, 2]], 0) and np.allclose(cell[2, :2], 0)):
        print(
            "  WARNING: cell is not diagonal/orthorhombic; ext8 registry assumes an axis-aligned "
            "lateral cell (a along x, b along y, c along z). Proceeding anyway.",
            file=sys.stderr,
        )
    ax, by, cz = cell[0, 0], cell[1, 1], cell[2, 2]

    shift = n_add * d
    new_atoms = atoms.copy()
    # IMPORTANT: MD-driver snapshots (and this synthetic test fixture) carry an ASE FixAtoms
    # constraint mirroring the 'fixed' array. Atoms.set_positions() is constraint-aware and
    # silently reverts any requested change for constrained (fixed) indices via
    # FixAtoms.adjust_positions -- so shifting *all* atoms up via set_positions() would
    # silently leave the fixed (bulk-position) Ag layers at their old z and only move the free
    # atoms, corrupting the geometry. We track fixed/free ourselves via the 'fixed' array (per
    # INTERFACE_SPEC) and write POSCAR selective dynamics from that directly, so the ASE-level
    # constraint is redundant here; drop it before touching positions.
    new_atoms.set_constraint()
    pos = new_atoms.get_positions()
    pos[:, 2] += shift
    new_atoms.set_positions(pos)

    new_cell = cell.copy()
    new_cell[2, 2] = cz + shift
    new_atoms.set_cell(new_cell)

    add_symbols, add_pos, add_fixed, add_role = [], [], [], []
    for j in range(1, n_add + 1):
        z = bottom_mean_z + shift - j * d
        if j % 2 == 1:
            xy = np.mod(bottom_xy + offset, [ax, by])
        else:
            xy = bottom_xy.copy()
        for k in range(n_per_layer):
            add_pos.append([xy[k, 0], xy[k, 1], z])
            add_symbols.append("Ag")
            add_fixed.append(1)
            add_role.append("slab")

    from ase import Atoms as ASEAtoms

    extra = ASEAtoms(symbols=add_symbols, positions=add_pos, cell=new_cell, pbc=True)

    # Fixed flags for the original atoms are untouched: the pre-existing fixed (bulk-position)
    # layers were already 1, and the pre-existing free layer(s) stay 0 and keep their MD
    # positions (only translated up in z above). Only the newly appended layers are forced fixed.
    fixed_old = np.array(new_atoms.arrays.get("fixed"))
    role_old = np.array(new_atoms.arrays.get("role"))

    combined = new_atoms + extra
    fixed_combined = np.concatenate([fixed_old, np.array(add_fixed, dtype=int)])
    role_combined = np.concatenate([role_old, np.array(add_role, dtype=object)])
    combined.arrays["fixed"] = fixed_combined
    combined.arrays["role"] = role_combined

    return combined, target_layers


def center_vacuum_z(atoms):
    """Translate atoms along z so the vacuum gap is split evenly across the cell boundary,
    i.e. the slab+film sits in the middle of the cell (the effect of atoms.center(axis=2), but
    robust to MD snapshots whose z coordinates were wrapped). With DIPOL = 0.5 0.5 0.5 the
    dipole-correction step then lands mid-vacuum. The MD cells put the bottom Ag layer at z=0,
    which places that step inside the metal and stalls the SCF.
    Returns (centered copy, z shift in Angstrom, vacuum gap in Angstrom)."""
    z_len = atoms.cell[2, 2]
    fz = np.sort(np.mod(atoms.get_positions()[:, 2] / z_len, 1.0))
    gaps = np.diff(np.concatenate([fz, [fz[0] + 1.0]]))
    i = int(np.argmax(gaps))
    gap_mid = fz[i] + gaps[i] / 2.0  # fractional, may exceed 1
    shift = (1.0 - gap_mid) * z_len
    out = atoms.copy()
    pos = out.get_positions()
    pos[:, 2] = np.mod(pos[:, 2] + shift, z_len)
    out.set_positions(pos, apply_constraint=False)  # FixAtoms would otherwise pin the fixed Ag layers
    return out, float(shift), float(gaps[i] * z_len)


# ---------------------------------------------------------------------------
# INCAR / KPOINTS / POTCAR_ORDER writers
# ---------------------------------------------------------------------------

INCAR_TEMPLATE = """\
SYSTEM = {system}

# --- OC25-training-matched settings (authoritative; must not be changed) ---
GGA    = RP        # RPBE exchange-correlation
IVDW   = 11         # D3 zero-damping van der Waals correction
ENCUT  = 400
ISMEAR = 0
SIGMA  = 0.1
ISPIN  = 1          # non-spin-polarized, as trained
IDIPOL = 3
LDIPOL = .TRUE.
DIPOL  = 0.5 0.5 0.5  # atoms are centered in z (center_vacuum_z), so the correction step at the cell boundary sits mid-vacuum

# --- Tightened relative to OC25 training (potentiometer needs a converged E_F/V_H) ---
EDIFF  = {ediff}    # training used 1e-4; tightened here
NELM   = {nelm}
ALGO   = {algo}
LREAL  = {lreal}
PREC   = {prec}

# --- Single-point / output flags added for the potentiometer (do not affect the PES) ---
NSW    = 0          # single point
IBRION = -1
ISYM   = 0          # dipole-corrected asymmetric slab has no exact symmetry; avoid symmetrization artifacts
LWAVE  = .FALSE.
LCHARG = .TRUE.     # CHGCAR for estimator (c) and for Bader reference sum
LAECHG = .TRUE.     # AECCAR0/AECCAR2 for Henkelman Bader (bader -ref CHGCAR_sum)
LVHAR  = .TRUE.     # LOCPOT = Hartree + ionic electrostatic potential only (no XC) for U_inner/Phi

# NOTE on LASPH: intentionally left at its default (.FALSE.), i.e. NOT enabled here even though
# it is common practice for D3/GGA total-energy accuracy. Rationale: the task list marks LASPH
# as a decision point ("LASPH=.TRUE.? -- decide and justify"). The instruction immediately after
# it is to "keep the OC25 flags authoritative and only add output flags" (LVHAR/LCHARG/LAECHG/
# LWAVE/NSW/IBRION/ISYM). LASPH changes the PAW one-center energy/potential (non-spherical
# contribution), which is not part of the verified OC25 training recipe and was not requested as
# an output flag. We are not optimizing energies here (NSW=0, no relaxation, no energy ranking
# across structures) -- the quantities we read out (E_F, LOCPOT, CHGCAR, Bader charges) are only
# very weakly sensitive to the aspherical PAW correction, so the risk of a small, undocumented
# shift relative to the MLIP's own training reference outweighs the benefit. Revisit only if a
# direct RPBE-D3 vs RPBE-D3+LASPH comparison on a few S0 snapshots shows E_F or sigma shifting by
# more than the per-snapshot noise floor (~0.1-0.15 V / a few %).

# --- Placeholders, set per cluster/job ---
# NCORE = <set at submission time, e.g. NCORE = 8 -- depends on Meta cluster node layout>
{kpar_line}
"""


def write_incar(path, args, system_name):
    kpar_line = f"KPAR   = {args.kpar}" if args.kpar else "# KPAR   = <unset>"
    content = INCAR_TEMPLATE.format(
        system=system_name,
        ediff=args.ediff,
        nelm=args.nelm,
        algo=args.algo,
        lreal=args.lreal,
        prec=args.prec,
        kpar_line=kpar_line,
    )
    with open(path, "w") as f:
        f.write(content)


def write_kpoints(path):
    with open(path, "w") as f:
        f.write("Gamma-only (8x8, ~23.5 A lateral; convergence verified at ~22 A, larger cell is safer)\n")
        f.write("0\n")
        f.write("Gamma\n")
        f.write("1 1 1\n")
        f.write("0 0 0\n")


def write_potcar_order(path, species_order):
    with open(path, "w") as f:
        f.write("# Species order for POTCAR concatenation (must match POSCAR order exactly).\n")
        f.write("# `cat POTCAR_Ag POTCAR_O ... > POTCAR` in this order. POTCARs are not shipped here.\n")
        for sp in species_order:
            setup, zval = RECOMMENDED_SETUPS.get(sp, (sp, None))
            zval_s = str(zval) if zval is not None else "unknown"
            f.write(f"{sp}\tsetup={setup}\tZVAL={zval_s}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-id", nargs="+", default=None, help="one or more state_ids")
    p.add_argument("--all", action="store_true", help="process every state found in manifest/md-dir/structures-dir")
    p.add_argument("--md-dir", default=None, help="default: <ROOT>/s0/md")
    p.add_argument("--structures-dir", default=None, help="default: <ROOT>/s0/structures")
    p.add_argument("--out-dir", default=None, help="default: <ROOT>/s0/dft")
    p.add_argument("--frames", default="last:5", help='glob (relative to snapshots/) or "last:N" (default: last:5)')
    p.add_argument("--ext8", action="store_true", help="(retained but NOT part of S0 since 2026-09-17) also write an ext8/ variant extended to 8 Ag layers")
    p.add_argument("--ediff", type=float, default=1e-5)
    p.add_argument("--nelm", type=int, default=200)
    p.add_argument("--algo", default="Fast")
    p.add_argument("--lreal", default="Auto")
    p.add_argument("--prec", default="Normal")
    p.add_argument("--ncore", default=None, help="written only as an INCAR comment placeholder")
    p.add_argument("--kpar", type=int, default=None)
    p.add_argument("--from-init", action="store_true", help="use structures/<state_id>/init.extxyz as the (sole) frame, named 'init'")
    return p


def resolve_root_dirs(args):
    here = os.path.dirname(os.path.abspath(__file__))  # .../ag_100/s0
    if args.md_dir is None:
        args.md_dir = os.path.join(here, "md")
    if args.structures_dir is None:
        args.structures_dir = os.path.join(here, "structures")
    if args.out_dir is None:
        args.out_dir = os.path.join(here, "dft")
    return args


def process_state(state_id, manifest, args, job_dirs):
    entry = manifest.get(state_id)
    if entry is None:
        print(f"WARNING: no manifest entry for {state_id}; proceeding with atoms-only metadata", file=sys.stderr)

    try:
        frames = discover_frames(state_id, args)
    except FileNotFoundError as e:
        print(f"SKIP {state_id}: {e}", file=sys.stderr)
        return

    for frame_name, atoms in frames:
        if "fixed" not in atoms.arrays:
            print(f"SKIP {state_id}/{frame_name}: missing per-atom 'fixed' array", file=sys.stderr)
            continue
        cation = cation_symbol_for_state(entry, atoms)
        species_order = species_order_for_atoms(atoms, cation)

        variants = [("native", atoms, entry.get("slab_layers") if entry else None)]
        if args.ext8:
            if entry is None:
                print(f"  SKIP ext8 for {state_id}/{frame_name}: no manifest entry (need slab_layers, lattice_constant_A)", file=sys.stderr)
            else:
                try:
                    ext_atoms, dft_layers = extend_to_8_layers(atoms, entry, target_layers=8)
                    if ext_atoms is None:
                        print(f"  ext8 skipped for {state_id}/{frame_name}: slab already >= 8 layers", file=sys.stderr)
                    else:
                        variants.append(("ext8", ext_atoms, dft_layers))
                except Exception as e:
                    print(f"  ext8 FAILED for {state_id}/{frame_name}: {e}", file=sys.stderr)

        for variant_name, variant_atoms, dft_layers in variants:
            job_dir = os.path.join(args.out_dir, state_id, frame_name, variant_name)
            os.makedirs(job_dir, exist_ok=True)

            variant_atoms, z_shift, vacuum_gap = center_vacuum_z(variant_atoms)
            if vacuum_gap < 10.0:
                print(f"  WARNING {state_id}/{frame_name}: vacuum gap only {vacuum_gap:.1f} A", file=sys.stderr)

            fixed = variant_atoms.arrays["fixed"]
            comment = f"{state_id} {frame_name} {variant_name}"
            write_poscar(variant_atoms, fixed, species_order, os.path.join(job_dir, "POSCAR"), comment)
            write_incar(os.path.join(job_dir, "INCAR"), args, system_name=comment)
            write_kpoints(os.path.join(job_dir, "KPOINTS"))
            write_potcar_order(os.path.join(job_dir, "POTCAR_ORDER.txt"), species_order)

            cell = np.array(variant_atoms.get_cell()).tolist()
            symbols = np.array(variant_atoms.get_chemical_symbols())
            n_cation = int(np.sum(symbols == cation)) if cation else 0
            n_anion = int(np.sum(symbols == "F"))

            slab_z = None
            if "role" in variant_atoms.arrays:
                roles = variant_atoms.arrays["role"]
                is_slab = roles == "slab"
                if np.any(is_slab):
                    slab_z = float(np.max(variant_atoms.get_positions()[is_slab, 2]))

            z_top_Ag = slab_z if slab_z is not None else (entry.get("z_slab_top_A") if entry else None)
            film_thickness = None
            if entry and entry.get("z_film_top_A") is not None and entry.get("z_slab_top_A") is not None:
                film_thickness = entry["z_film_top_A"] - entry["z_slab_top_A"]

            meta = {
                "state_id": state_id,
                "frame": frame_name,
                "variant": variant_name,
                "slab_layers_md": entry.get("slab_layers") if entry else None,
                "slab_layers_dft": dft_layers,
                "area_A2": entry.get("area_A2") if entry else None,
                "n_ions": n_cation + n_anion,
                "n_cation": n_cation,
                "n_anion": n_anion,
                "delta_N": entry.get("delta_N") if entry else None,
                "sigma_nominal_uC_cm2": entry.get("sigma_nominal_uC_cm2") if entry else None,
                "electrolyte": entry.get("electrolyte") if entry else None,
                "cation_species": cation,
                "z_top_Ag": z_top_Ag,
                "film_thickness": film_thickness,
                "z_shift_A": z_shift,  # added to MD z to center the DFT cell
                "vacuum_gap_A": vacuum_gap,
                "cell": cell,
                "n_atoms": len(variant_atoms),
                "species_order": species_order,
            }
            with open(os.path.join(job_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            job_dirs.append(job_dir)


def main(argv=None):
    args = build_parser().parse_args(argv)
    args = resolve_root_dirs(args)

    manifest = load_manifest(args.structures_dir)

    if args.all:
        state_ids = all_state_ids(args, manifest)
    elif args.state_id:
        state_ids = args.state_id
    else:
        print("ERROR: must give --state-id <...> or --all", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    job_dirs = []
    for state_id in state_ids:
        print(f"Processing {state_id} ...")
        process_state(state_id, manifest, args, job_dirs)

    jobs_path = os.path.join(args.out_dir, "jobs.txt")
    existing = []
    if os.path.exists(jobs_path):
        with open(jobs_path) as f:
            existing = [l.strip() for l in f if l.strip()]
    merged = sorted(set(existing) | set(job_dirs))
    with open(jobs_path, "w") as f:
        for j in merged:
            f.write(j + "\n")

    print(f"Wrote {len(job_dirs)} job dirs this run; jobs.txt now lists {len(merged)} total: {jobs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
