"""
Shared helpers for the S0 MD stage (run_md_oc25.py, analyze_md.py).

Not part of INTERFACE_SPEC.md's file contract itself -- this is an internal
implementation module imported by the two driver scripts so the topology
definitions and cutoffs are identical between the on-the-fly MD monitor and
the post-hoc analysis.

Conventions (see INTERFACE_SPEC.md):
  - atoms.arrays['fixed']: int array, 1 = held fixed, 0 = free.
  - atoms.arrays['role']: string array, one of slab/water/cation/anion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list

# ---------------------------------------------------------------------------
# Default reactive/contact cutoffs (Angstrom). CLI-tunable in both scripts;
# these are just the fallback defaults, per the task spec.
# ---------------------------------------------------------------------------
DEFAULT_CUTOFFS = {
    "oh": 1.25,       # O-H covalent bond (defines water/hydronium/hydroxide)
    "hf": 1.10,       # H-F bond (HF(aq) formation)
    "agh": 2.00,      # Ag-H contact ("H on Ag")
    "agf": 2.90,      # Ag-F contact
    "agcat": 3.60,    # Ag-cation (Na/Cs) contact
    "ago_search": 6.00,  # generous search radius for the min(Ag-O) diagnostic
}

DEFAULT_TOP_LAYER_TOL = 1.0   # Angstrom; window used to define "top Ag layer"
DEFAULT_DESORB_OFFSET = 5.0   # Angstrom above z_film_top counted as desorbed

CATION_SYMBOLS = ("Na", "Cs")

# Column order written to topology.csv. The first block
# (n_HF, n_OH-, n_H3O+, n_H_on_Ag, n_F_contact, n_cation_contact,
# n_water_dissociated) matches INTERFACE_SPEC.md's topology.csv contract
# exactly (n_water_dissociated := n_OH- + n_H3O+). Remaining columns are
# additional diagnostics requested by the S0 task spec (min Ag-O distance,
# topmost-Ag z, desorbed-water count, intact-water count) appended after the
# contract columns so any consumer reading by column name is unaffected.
TOPOLOGY_FIELDS = [
    "step", "time_ps", "phase",
    "n_HF", "n_OH-", "n_H3O+", "n_H_on_Ag", "n_F_contact", "n_cation_contact",
    "n_water_dissociated",
    "n_water", "min_Ag_O_dist", "z_top_Ag", "n_desorbed",
]

THERMO_FIELDS = ["step", "time_ps", "T_K", "Epot_eV", "Ekin_eV", "phase"]

# SPC/E-style point charges used only for the classical dipole-proxy
# diagnostic in analyze_md.py (NOT used by the MLIP potential itself).
SPCE_CHARGES = {
    "O": -0.8476,
    "H": 0.4238,
    "Na": 1.0,
    "Cs": 1.0,
    "F": -1.0,
    "Ag": 0.0,
}


def attach_fixatoms(atoms):
    """(Re)build the FixAtoms constraint from atoms.arrays['fixed'].

    Never trust that a constraint survived a round trip through extxyz I/O --
    always rebuild it explicitly from the per-atom 'fixed' column.
    """
    if "fixed" not in atoms.arrays:
        raise ValueError(
            "atoms is missing the required per-atom 'fixed' array "
            "(see INTERFACE_SPEC.md)."
        )
    fixed = np.asarray(atoms.arrays["fixed"]).astype(int).astype(bool)
    atoms.set_constraint()  # drop whatever came in over I/O, if anything
    idx = np.where(fixed)[0]
    if len(idx):
        atoms.set_constraint(FixAtoms(indices=idx))
    return atoms


def free_mask(atoms):
    return ~np.asarray(atoms.arrays["fixed"]).astype(int).astype(bool)


def load_manifest(structures_dir):
    """Return {state_id: entry_dict} from structures/manifest.json, or {} if absent."""
    path = Path(structures_dir) / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        entries = json.load(fh)
    return {e["state_id"]: e for e in entries}


def load_manifest_entry(structures_dir, state_id):
    return load_manifest(structures_dir).get(state_id)


def top_ag_indices(atoms, syms=None, z=None, top_layer_tol=DEFAULT_TOP_LAYER_TOL):
    """Indices of Ag atoms within `top_layer_tol` of the highest Ag z."""
    if syms is None:
        syms = np.array(atoms.get_chemical_symbols())
    if z is None:
        z = atoms.get_positions()[:, 2]
    ag_idx = np.where(syms == "Ag")[0]
    if len(ag_idx) == 0:
        return ag_idx, np.nan
    z_top = float(z[ag_idx].max())
    top_idx = ag_idx[z[ag_idx] > z_top - top_layer_tol]
    return top_idx, z_top


def compute_topology(atoms, cutoffs=None, z_film_top=None,
                      top_layer_tol=DEFAULT_TOP_LAYER_TOL,
                      desorb_offset=DEFAULT_DESORB_OFFSET):
    """Classify per-frame reactive/contact chemistry.

    Returns a dict with all keys in TOPOLOGY_FIELDS except step/time_ps/phase
    (those are added by the caller).
    """
    cutoffs = dict(DEFAULT_CUTOFFS if cutoffs is None else cutoffs)
    n = len(atoms)
    syms = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    z = pos[:, 2]

    o_idx = np.where(syms == "O")[0]
    top_ag_idx, z_top_ag = top_ag_indices(atoms, syms, z, top_layer_tol)

    out = {
        "n_HF": 0, "n_OH-": 0, "n_H3O+": 0, "n_H_on_Ag": 0,
        "n_F_contact": 0, "n_cation_contact": 0, "n_water_dissociated": 0,
        "n_water": 0, "min_Ag_O_dist": float("nan"),
        "z_top_Ag": z_top_ag, "n_desorbed": -1,
    }

    if n < 2:
        return out

    react_cut = max(cutoffs["oh"], cutoffs["hf"], cutoffs["agh"],
                     cutoffs["agf"], cutoffs["agcat"])
    i, j, d = neighbor_list("ijd", atoms, react_cut)

    # O-H bonds -> classify each O by its H count
    oh_mask = (syms[i] == "O") & (syms[j] == "H") & (d < cutoffs["oh"])
    h_count_per_o = {int(o): 0 for o in o_idx}
    o_of_h = {}
    for ii, jj in zip(i[oh_mask], j[oh_mask]):
        h_count_per_o[int(ii)] = h_count_per_o.get(int(ii), 0) + 1
        o_of_h[int(jj)] = int(ii)
    n_ohminus = sum(1 for c in h_count_per_o.values() if c == 1)
    n_h3o = sum(1 for c in h_count_per_o.values() if c >= 3)
    n_water = sum(1 for c in h_count_per_o.values() if c == 2)

    # H-F bonds -> n_HF = number of F atoms with an H within cutoff
    hf_mask = (syms[i] == "H") & (syms[j] == "F") & (d < cutoffs["hf"])
    n_hf = len(set(int(x) for x in j[hf_mask]))

    # H on Ag: H close to ANY Ag and not already covalently bonded to an O
    agh_mask = (syms[i] == "Ag") & (syms[j] == "H") & (d < cutoffs["agh"])
    h_near_ag = set(int(x) for x in j[agh_mask])
    n_h_on_ag = sum(1 for h in h_near_ag if h not in o_of_h)

    # Contact ions: within cutoff of the TOP Ag layer specifically
    if len(top_ag_idx):
        top_set = set(int(x) for x in top_ag_idx)
        in_top = np.isin(i, np.fromiter(top_set, dtype=int))
        agf_mask = in_top & (syms[j] == "F") & (d < cutoffs["agf"])
        n_f_contact = len(set(int(x) for x in j[agf_mask]))
        agcat_mask = in_top & np.isin(syms[j], CATION_SYMBOLS) & (d < cutoffs["agcat"])
        n_cat_contact = len(set(int(x) for x in j[agcat_mask]))
    else:
        n_f_contact = 0
        n_cat_contact = 0

    # min Ag-O distance: separate, larger-radius search (diagnostic only)
    if len(top_ag_idx) or (syms == "Ag").any():
        i2, j2, d2 = neighbor_list("ijd", atoms, cutoffs["ago_search"])
        ago_mask = (syms[i2] == "Ag") & (syms[j2] == "O")
        min_ago = float(d2[ago_mask].min()) if ago_mask.any() else float("nan")
    else:
        min_ago = float("nan")

    # desorbed water: O far above the nominal film top (vacuum region)
    if z_film_top is not None and len(o_idx):
        n_desorbed = int((z[o_idx] > (z_film_top + desorb_offset)).sum())
    else:
        n_desorbed = -1

    out.update({
        "n_HF": n_hf, "n_OH-": n_ohminus, "n_H3O+": n_h3o,
        "n_H_on_Ag": n_h_on_ag, "n_F_contact": n_f_contact,
        "n_cation_contact": n_cat_contact,
        "n_water_dissociated": n_ohminus + n_h3o,
        "n_water": n_water, "min_Ag_O_dist": min_ago,
        "z_top_Ag": z_top_ag, "n_desorbed": n_desorbed,
    })
    return out


def water_bond_map(atoms, oh_cut=DEFAULT_CUTOFFS["oh"]):
    """Return {o_index: [h_index, ...]} for O atoms with exactly 2 bonded H
    (intact water molecules), using the same O-H cutoff as compute_topology.
    """
    syms = np.array(atoms.get_chemical_symbols())
    n = len(atoms)
    if n < 2:
        return {}
    i, j, d = neighbor_list("ijd", atoms, oh_cut + 0.05)
    mask = (syms[i] == "O") & (syms[j] == "H") & (d < oh_cut)
    bonds = {}
    for ii, jj in zip(i[mask], j[mask]):
        bonds.setdefault(int(ii), []).append(int(jj))
    return {o: hs for o, hs in bonds.items() if len(hs) == 2}


def mic_vectors(p_from, p_to, cell, pbc):
    """Minimum-image vectors p_to - p_from (arrays of shape (N,3))."""
    raw = np.asarray(p_to) - np.asarray(p_from)
    vmic, dmic = find_mic(raw, cell, pbc)
    return vmic, dmic


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)
