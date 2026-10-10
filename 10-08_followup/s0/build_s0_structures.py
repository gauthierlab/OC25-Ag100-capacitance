#!/usr/bin/env python3
"""
build_s0_structures.py -- generate the S0 smoke-test structure set for
Ag(100)/water/NaF(CsF) capacitance calculations with the OC25 MLIP.

See s0/INTERFACE_SPEC.md for the file conventions this script must follow.
Do not edit INTERFACE_SPEC.md; this script conforms to it.

Usage (run from ROOT/s0):
    python3 build_s0_structures.py
    python3 build_s0_structures.py --dry-run
    python3 build_s0_structures.py --only 'L4_n8_NaF_dN-2_.*'
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
from ase import Atoms
from ase.build import fcc100
from ase.constraints import FixAtoms
from ase.io import write as ase_write

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
E_CHARGE_C = 1.602176634e-19          # elementary charge, coulombs
N_A = 6.02214076e23                   # Avogadro's number
WATER_NUMBER_DENSITY = 0.03342        # molecules / A^3 (~0.997 g/cm3 water)
WATER_MOLAR_MASS = 18.01528           # g/mol
BULK_WATER_DENSITY = 0.997            # g/cm3
MIN_DIST_A = 1.8                      # Angstrom, hard-sphere floor
PACK_TOLERANCE = 2.0                  # Angstrom, packmol/fallback tolerance
O_H_BOND_A = 1.0
HOH_ANGLE_DEG = 109.47
GAP_ABOVE_SLAB_A = 2.5                # film starts this far above top Ag layer
MAX_PACK_RETRIES = 6                  # full-fill restarts on RSA jamming

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Lattice constant from CONTCAR
# ----------------------------------------------------------------------
def lattice_constant_from_contcar(path):
    """Read a 1-atom fcc primitive-cell CONTCAR/POSCAR and return
    (d_NN, a) where d_NN = |v1| and a = sqrt(2)*d_NN (conventional fcc a)."""
    with open(path) as fh:
        lines = fh.readlines()
    scale = float(lines[1].split()[0])
    v1 = np.array([float(x) for x in lines[2].split()[:3]]) * scale
    d_nn = np.linalg.norm(v1)
    a = np.sqrt(2.0) * d_nn
    return d_nn, a


# ----------------------------------------------------------------------
# Slab construction
# ----------------------------------------------------------------------
def build_slab(a, n, layers):
    """fcc100 Ag slab, n x n x layers, orthogonal cell, vacuum=0 so the
    bottom layer sits at z=0 and the cell's c-vector length equals the
    slab thickness. Returns (atoms, z_top)."""
    slab = fcc100('Ag', size=(n, n, layers), a=a, vacuum=0.0, orthogonal=True)
    slab.positions[:, 2] -= slab.positions[:, 2].min()  # bottom at z=0
    z_top = slab.positions[:, 2].max()

    zvals = slab.positions[:, 2]
    uniq_z = np.sort(np.unique(np.round(zvals, 3)))
    bottom_two = set(uniq_z[:2].tolist())
    fixed = np.array(
        [1 if round(z, 3) in bottom_two else 0 for z in zvals], dtype=int
    )
    slab.set_array('fixed', fixed)
    slab.set_array('role', np.array(['slab'] * len(slab), dtype=object))
    return slab, z_top


def area_from_cell(atoms):
    """Lateral (x,y) area of an orthogonal cell, in A^2."""
    return atoms.cell[0, 0] * atoms.cell[1, 1]


# ----------------------------------------------------------------------
# Water / ion count formulas
# ----------------------------------------------------------------------
def water_and_ion_counts(area_A2, film_A, conc_M, delta_N):
    volume_A3 = area_A2 * film_A
    n_water_target = round(WATER_NUMBER_DENSITY * volume_A3)
    volume_L = volume_A3 * 1e-27
    n_pairs = round(conc_M * N_A * volume_L)
    # subtract one water per NaF pair to allow volume for the ions
    n_water = n_water_target - n_pairs
    n_cation = n_pairs + max(0, -delta_N)
    n_anion = n_pairs + max(0, delta_N)
    return dict(
        n_water_target=n_water_target,
        n_pairs=n_pairs,
        n_water=n_water,
        n_cation=n_cation,
        n_anion=n_anion,
    )


def sigma_per_ion_uC_cm2(area_A2):
    area_cm2 = area_A2 * 1e-16  # 1 A^2 = 1e-16 cm^2
    return (E_CHARGE_C / area_cm2) * 1e6


# ----------------------------------------------------------------------
# Rigid water monomer geometry
# ----------------------------------------------------------------------
def random_rotation_matrix(rng):
    """Uniformly random rotation matrix (Shoemake / QR method via
    random unit quaternion)."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    w, x, y, z = q4, q1, q2, q3
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return R


def water_template():
    """O at origin, H1/H2 at O-H = 1.0 A, HOH = 109.47 deg, in the xy plane,
    centered on the molecule's centroid-ish (O at origin is fine; we place
    by O position)."""
    half = np.radians(HOH_ANGLE_DEG / 2.0)
    h1 = np.array([O_H_BOND_A * np.sin(half), O_H_BOND_A * np.cos(half), 0.0])
    h2 = np.array([-O_H_BOND_A * np.sin(half), O_H_BOND_A * np.cos(half), 0.0])
    o = np.zeros(3)
    return np.array([o, h1, h2])  # order O, H, H


def place_water(center, rng):
    template = water_template()
    R = random_rotation_matrix(rng)
    return (template @ R.T) + center


# ----------------------------------------------------------------------
# Periodic minimum-image packing (python fallback)
# ----------------------------------------------------------------------
class PeriodicPacker:
    """Rejection-sampling rigid-body packer with a minimum-image
    neighbor check via scipy cKDTree (numpy chunked fallback if scipy is
    absent). periodic_z=True for fully periodic bulk boxes; False for the
    slab/film geometry (vacuum above/below in z, only x,y are periodic)."""

    def __init__(self, cell_xyz, periodic_z, min_dist=PACK_TOLERANCE):
        self.Lx, self.Ly, self.Lz = cell_xyz
        self.periodic_z = periodic_z
        self.min_dist = min_dist
        self.positions = np.zeros((0, 3))
        self._tree = None
        self._dirty = True

    def _images(self, pts):
        shifts_xy = [-1, 0, 1]
        shifts_z = [-1, 0, 1] if self.periodic_z else [0]
        out = []
        for ix in shifts_xy:
            for iy in shifts_xy:
                for iz in shifts_z:
                    out.append(pts + np.array([ix * self.Lx, iy * self.Ly,
                                                iz * self.Lz]))
        return np.vstack(out)

    def _ensure_tree(self):
        if not self._dirty:
            return
        if len(self.positions) == 0:
            self._tree = None
        elif HAVE_SCIPY:
            tiled = self._images(self.positions)
            self._tree = cKDTree(tiled)
        else:
            self._tree = self._images(self.positions)
        self._dirty = False

    def min_dist_to_existing(self, candidate_pts):
        self._ensure_tree()
        if self._tree is None:
            return np.inf
        if HAVE_SCIPY:
            d, _ = self._tree.query(candidate_pts, k=1)
            return np.min(d)
        else:
            # numpy chunked brute force fallback
            diffs = self._tree[None, :, :] - candidate_pts[:, None, :]
            d = np.sqrt(np.sum(diffs ** 2, axis=-1))
            return np.min(d)

    def try_add(self, candidate_pts):
        d = self.min_dist_to_existing(candidate_pts)
        if d < self.min_dist:
            return False
        self.positions = np.vstack([self.positions, candidate_pts])
        self._dirty = True
        return True


class PackingStuckError(RuntimeError):
    """Raised when rejection sampling cannot place a molecule after
    max_attempts tries -- caller may retry the whole state with a fresh
    seed (classic RSA-jamming tail behaviour near target density)."""
    pass


def pack_molecules(packer, rng, kinds, z_lo, z_hi, max_attempts=20000):
    """kinds: list of ('water'|'ion', label) tuples to place, each with its
    own z-window given by (z_lo, z_hi) (same window for the whole call --
    callers make one call per distinct region/species group).
    Returns list of (positions[Natoms,3], label) in placement order."""
    placed = []
    for kind, label in kinds:
        ok = False
        for _attempt in range(max_attempts):
            x = rng.uniform(0.0, packer.Lx)
            y = rng.uniform(0.0, packer.Ly)
            z = rng.uniform(z_lo, z_hi)
            center = np.array([x, y, z])
            if kind == 'water':
                pts = place_water(center, rng)
            else:
                pts = center.reshape(1, 3)
            if packer.try_add(pts):
                placed.append((pts, label))
                ok = True
                break
        if not ok:
            raise PackingStuckError(
                f"Failed to place a '{label}' ({kind}) after {max_attempts} "
                f"attempts in z window [{z_lo:.2f}, {z_hi:.2f}] "
                f"({len(packer.positions)} atoms already placed)."
            )
    return placed


# ----------------------------------------------------------------------
# packmol.inp writer (always written, even if packmol is not run)
# ----------------------------------------------------------------------
def write_pdb_water(path):
    half = np.radians(HOH_ANGLE_DEG / 2.0)
    o = np.zeros(3)
    h1 = np.array([O_H_BOND_A * np.sin(half), O_H_BOND_A * np.cos(half), 0.0])
    h2 = np.array([-O_H_BOND_A * np.sin(half), O_H_BOND_A * np.cos(half), 0.0])
    lines = []
    lines.append("HEADER    WATER TEMPLATE (SPC/E-like rigid monomer)")
    fmt = "HETATM{serial:>5} {name:<4} HOH A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}"
    lines.append(fmt.format(serial=1, name="O", x=o[0], y=o[1], z=o[2], elem="O"))
    lines.append(fmt.format(serial=2, name="H1", x=h1[0], y=h1[1], z=h1[2], elem="H"))
    lines.append(fmt.format(serial=3, name="H2", x=h2[0], y=h2[1], z=h2[2], elem="H"))
    lines.append("END")
    with open(path, 'w') as fh:
        fh.write("\n".join(lines) + "\n")


def write_pdb_ion(path, element):
    fmt = "HETATM{serial:>5} {name:<4} ION A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2}"
    line = fmt.format(serial=1, name=element.upper(), x=0.0, y=0.0, z=0.0,
                       elem=element.upper())
    with open(path, 'w') as fh:
        fh.write("HEADER    ION TEMPLATE\n" + line + "\nEND\n")


def build_packmol_inp(state_dir, templates_rel, seed, boxes, output_name="packed.pdb"):
    """boxes: list of dicts {template, number, box (xlo,ylo,zlo,xhi,yhi,zhi), comment}"""
    lines = []
    lines.append(f"tolerance {PACK_TOLERANCE:.1f}")
    lines.append("filetype pdb")
    lines.append(f"output {output_name}")
    lines.append(f"seed {seed}")
    lines.append("")
    for b in boxes:
        lines.append(f"# {b['comment']}")
        lines.append(f"structure {templates_rel}/{b['template']}")
        lines.append(f"  number {b['number']}")
        x0, y0, z0, x1, y1, z1 = b['box']
        lines.append(
            f"  inside box {x0:.3f} {y0:.3f} {z0:.3f} {x1:.3f} {y1:.3f} {z1:.3f}"
        )
        lines.append("end structure")
        lines.append("")
    with open(os.path.join(state_dir, 'packmol.inp'), 'w') as fh:
        fh.write("\n".join(lines))


# ----------------------------------------------------------------------
# Deterministic seeding
# ----------------------------------------------------------------------
def seed_for_state(state_id):
    h = hashlib.sha256(state_id.encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


# ----------------------------------------------------------------------
# Species reordering for POSCAR (Ag, O, H, Na/Cs, F)
# ----------------------------------------------------------------------
def reorder_by_species(atoms, order):
    symbols = np.array(atoms.get_chemical_symbols())
    idx = []
    for sp in order:
        idx.extend(np.where(symbols == sp)[0].tolist())
    remaining = [i for i in range(len(atoms)) if i not in set(idx)]
    idx = idx + remaining
    new = atoms[idx]
    return new


# ----------------------------------------------------------------------
# State matrix
# ----------------------------------------------------------------------
def build_state_matrix(n, layers_list=(4,)):  # L3 dropped 2026-09-17: 4 layers only
    states = []

    # 1. NaF interfaces
    for layers in layers_list:
        for delta_N in (-2, -1, 0, 1, 2):
            for r in (1, 2, 3, 4):
                if delta_N == 0:
                    ion_start = 'random'
                else:
                    ion_start = 'surface' if r in (1, 2) else 'far'
                sign = '+' if delta_N > 0 else ('-' if delta_N < 0 else '')
                mag = abs(delta_N)
                dn_str = f"{sign}{mag}" if delta_N != 0 else "0"
                state_id = f"L{layers}_n{n}_NaF_dN{dn_str}_{ion_start}_r{r}"
                states.append(dict(
                    state_id=state_id, kind='interface', electrolyte='NaF',
                    cation_symbol='Na', anion_symbol='F',
                    slab_layers=layers, n_lateral=n, delta_N=delta_N,
                    ion_start=ion_start, replica=r,
                    purpose="quick-look C at PZC / two-sided start",
                ))

    # 2. CsF interfaces, n=8, layers=4
    for delta_N in (-2, 0, 2):
        for r in (1, 2):
            if delta_N == 0:
                ion_start = 'random'
            else:
                ion_start = 'surface' if r == 1 else 'far'
            sign = '+' if delta_N > 0 else ('-' if delta_N < 0 else '')
            mag = abs(delta_N)
            dn_str = f"{sign}{mag}" if delta_N != 0 else "0"
            state_id = f"L4_n{n}_CsF_dN{dn_str}_{ion_start}_r{r}"
            states.append(dict(
                state_id=state_id, kind='interface', electrolyte='CsF',
                cation_symbol='Cs', anion_symbol='F',
                slab_layers=4, n_lateral=n, delta_N=delta_N,
                ion_start=ion_start, replica=r,
                purpose=("in-distribution cation comparison: Cs+ is in the "
                          "OC25 training ion set, Na+ is not"),
            ))

    # 3. Bulk water
    for r in (1, 2):
        state_id = f"bulk_H2O_r{r}"
        states.append(dict(
            state_id=state_id, kind='bulk_water', electrolyte='H2O',
            cation_symbol=None, anion_symbol=None,
            slab_layers=None, n_lateral=None, delta_N=0,
            ion_start=None, replica=r,
            purpose="MLIP bulk-water sanity (density/RDF/diffusion)",
        ))

    # 4. Bulk NaF
    state_id = "bulk_NaF_r1"
    states.append(dict(
        state_id=state_id, kind='bulk_naf', electrolyte='NaF_bulk',
        cation_symbol='Na', anion_symbol='F',
        slab_layers=None, n_lateral=None, delta_N=0,
        ion_start=None, replica=1,
        purpose="ion pairing / hydration sanity",
    ))

    return states


def _dn_str(delta_N):
    if delta_N == 0:
        return "0"
    return f"{'+' if delta_N > 0 else '-'}{abs(delta_N)}"


# Counter-ion-only sets (added 2026-10-08, plan_meta_followup_2026-10-08.md Parts B and E).
# No background salt (build with --conc 0): dN < 0 adds |dN| cations, dN > 0 adds dN F-, dN = 0 is pure
# water. The F- side is the same for both salts, so it is built once (electrolyte 'F_ci').
COUNTERION = {
    'H2O': (None, None),
    'F_ci': (None, 'F'),
    'Na_ci': ('Na', None),
    'Cs_ci': ('Cs', None),
}


def _counterion_spec(prefix, n, delta_N, electrolyte, replica, purpose):
    cation, anion = COUNTERION[electrolyte]
    if delta_N == 0:
        ion_start = 'random'
    else:
        ion_start = 'surface' if replica % 2 == 1 else 'far'
    tag = electrolyte.replace('_ci', '')
    state_id = f"{prefix}_n{n}_{tag}_dN{_dn_str(delta_N)}_{ion_start}_r{replica}"
    return dict(state_id=state_id, kind='interface', electrolyte=electrolyte,
                cation_symbol=cation, anion_symbol=anion,
                slab_layers=4, n_lateral=n, delta_N=delta_N,
                ion_start=ion_start, replica=replica, purpose=purpose)


def build_lc_matrix(n=8):
    """LC set: same 8x8 cell as S0, counter-ions only. Gets DFT single points.
    Pure water dN=0 x4; F- dN=+1,+2 x4; Na+ and Cs+ dN=-1,-2 x2. 20 trajectories."""
    states = []
    for r in (1, 2, 3, 4):
        states.append(_counterion_spec('LC', n, 0, 'H2O', r,
                                       "PZC and fluctuation ceiling of Ag(100)/pure water"))
    for dN in (1, 2):
        for r in (1, 2, 3, 4):
            states.append(_counterion_spec('LC', n, dN, 'F_ci', r,
                                           "counter-ion-only anodic side (shared by NaF and CsF)"))
    for el in ('Na_ci', 'Cs_ci'):
        for dN in (-1, -2):
            for r in (1, 2):
                states.append(_counterion_spec('LC', n, dN, el, r,
                                               "counter-ion-only cathodic side, cation comparison"))
    return states


SL_DELTA_N = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25)


def build_sl_matrix(n=16, dn_list=SL_DELTA_N, cations=('Na_ci', 'Cs_ci'), replicas=(1, 2),
                    include_water=True, prefix='SL'):
    """SL set: Shin-like n=16 salt-free cells for the work-function-head production run.
    No DFT. dn_list holds magnitudes; F- for dN > 0, each cation in `cations` for dN < 0."""
    states = []
    if include_water:
        for r in replicas:
            states.append(_counterion_spec(prefix, n, 0, 'H2O', r, "SL pure-water PZC"))
    for dN in dn_list:
        for r in replicas:
            states.append(_counterion_spec(prefix, n, dN, 'F_ci', r, "SL anodic counter-ion state"))
        for el in cations:
            for r in replicas:
                states.append(_counterion_spec(prefix, n, -dN, el, r, "SL cathodic counter-ion state"))
    return states


def build_lr1_matrix():
    """LR1: bulk boxes (~60 A cubic) for the Stillinger-Lovett / longitudinal-dielectric sum rules."""
    out = []
    for kind, el, cat in (('bulk_water', 'H2O', None), ('bulk_naf', 'NaF_bulk', 'Na'),
                          ('bulk_csf', 'CsF_bulk', 'Cs')):
        out.append(dict(state_id=f"LR1_{el.replace('_bulk', '')}_1M_r1" if cat else "LR1_H2O_r1",
                        kind=kind, electrolyte=el, cation_symbol=cat,
                        anion_symbol='F' if cat else None, slab_layers=None, n_lateral=None,
                        delta_N=0, ion_start=None, replica=1,
                        purpose="LR1 bulk perfect-screening sum rule / longitudinal dielectric"))
    return out


# ----------------------------------------------------------------------
# Interface-state builder
# ----------------------------------------------------------------------
def build_interface_state(spec, a, film_A, vacuum_A, conc_M, templates_dir_rel):
    n = spec['n_lateral']
    layers = spec['slab_layers']
    delta_N = spec['delta_N']
    ion_start = spec['ion_start']
    cation_sym = spec['cation_symbol']
    anion_sym = spec['anion_symbol']

    slab, z_top = build_slab(a, n, layers)
    area = area_from_cell(slab)
    counts = water_and_ion_counts(area, film_A, conc_M, delta_N)

    z_film_lo = z_top + GAP_ABOVE_SLAB_A
    z_film_hi = z_film_lo + film_A
    cell_z = z_film_hi + vacuum_A

    n_excess = abs(delta_N)
    excess_species = None
    if delta_N < 0:
        excess_species = ('cation', cation_sym)
    elif delta_N > 0:
        excess_species = ('anion', anion_sym)

    n_pairs = counts['n_pairs']
    n_water = counts['n_water']
    base_seed = seed_for_state(spec['state_id'])

    # Naive rejection-sampling packing can jam in its last percent or two
    # (classic RSA tail behaviour); retry the *whole* fill with a fresh
    # seed and a larger per-molecule attempt budget rather than trying to
    # patch a single stuck molecule into an already-frozen configuration.
    last_err = None
    for retry in range(MAX_PACK_RETRIES):
        rng = np.random.default_rng(base_seed + retry * 104729)
        packer = PeriodicPacker((slab.cell[0, 0], slab.cell[1, 1], cell_z),
                                 periodic_z=False)
        # seed packer with existing slab atoms so nothing overlaps the metal
        packer.positions = slab.positions.copy()
        packer._dirty = True
        max_attempts = 20000 * (retry + 1)

        placed_all = []  # (pts, role, symbol)
        excess_ion_z = []
        try:
            # (a) excess ions in constrained sub-region
            if n_excess > 0:
                role, sym = excess_species
                if ion_start == 'surface':
                    lo, hi = z_film_lo, z_film_lo + 3.0
                elif ion_start == 'far':
                    lo, hi = z_top + 12.0, z_film_hi
                else:
                    lo, hi = z_film_lo, z_film_hi
                kinds = [('ion', (role, sym))] * n_excess
                placed = pack_molecules(packer, rng, kinds, lo, hi, max_attempts)
                for pts, label in placed:
                    placed_all.append((pts, label[0], label[1]))
                    excess_ion_z.append(float(pts[0, 2]))

            # (b) neutral pairs, randomly across the whole film
            neutral_kinds = ([('ion', ('cation', cation_sym))] * n_pairs +
                              [('ion', ('anion', anion_sym))] * n_pairs)
            if len(neutral_kinds):
                rng.shuffle(neutral_kinds)
            placed = pack_molecules(packer, rng, neutral_kinds, z_film_lo, z_film_hi,
                                     max_attempts)
            for pts, label in placed:
                placed_all.append((pts, label[0], label[1]))

            # (c) bulk waters, randomly across the whole film
            water_kinds = [('water', ('water', 'H2O'))] * n_water
            placed = pack_molecules(packer, rng, water_kinds, z_film_lo, z_film_hi,
                                     max_attempts)
            for pts, label in placed:
                placed_all.append((pts, label[0], label[1]))
            break  # success
        except PackingStuckError as e:
            last_err = e
            continue
    else:
        raise PackingStuckError(
            f"{spec['state_id']}: packing did not converge after "
            f"{MAX_PACK_RETRIES} full-fill retries; last error: {last_err}"
        )

    # assemble Atoms
    symbols = list(slab.get_chemical_symbols())
    positions = [slab.positions[i] for i in range(len(slab))]
    fixed = list(slab.get_array('fixed'))
    role = list(slab.get_array('role'))
    mol_id = [-1 - i for i in range(len(slab))]  # unique, no exclusion needed

    next_mol = 0
    for pts, r, sym in placed_all:
        if r == 'water':
            symbols.extend(['O', 'H', 'H'])
            positions.extend(pts.tolist())
            fixed.extend([0, 0, 0])
            role.extend(['water', 'water', 'water'])
            mol_id.extend([next_mol, next_mol, next_mol])
        else:
            symbols.append(sym)
            positions.append(pts[0].tolist())
            fixed.append(0)
            role.append(r)
            mol_id.append(next_mol)
        next_mol += 1

    cell = np.array([
        [slab.cell[0, 0], 0, 0],
        [0, slab.cell[1, 1], 0],
        [0, 0, cell_z],
    ])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.set_array('fixed', np.array(fixed, dtype=int))
    atoms.set_array('role', np.array(role, dtype=object))
    fix_idx = np.where(np.array(fixed) == 1)[0]
    atoms.set_constraint(FixAtoms(indices=fix_idx))

    meta = dict(
        area_A2=area, z_slab_top_A=z_top, z_film_top_A=z_film_hi,
        cell_z_A=cell_z, n_water=n_water, n_pairs=n_pairs,
        n_cation=counts['n_cation'], n_anion=counts['n_anion'],
        n_Ag=len(slab), n_atoms=len(atoms), mol_id=np.array(mol_id),
        excess_ion_z=excess_ion_z,
    )
    return atoms, meta


# ----------------------------------------------------------------------
# Bulk-box builders
# ----------------------------------------------------------------------
def bulk_box_side(n_water):
    vol_cm3 = n_water * WATER_MOLAR_MASS / (N_A * BULK_WATER_DENSITY)
    side_cm = vol_cm3 ** (1.0 / 3.0)
    return side_cm * 1e8  # -> Angstrom


def build_bulk_state(spec, bulk_side=None, conc_M=1.0):
    if bulk_side is None:
        n_water = 216
        side = bulk_box_side(n_water)
        n_pairs = 4 if spec['kind'] == 'bulk_naf' else 0
    else:
        # LR1 boxes: fixed side, water at bulk number density, conc_M of salt
        side = bulk_side
        vol = side ** 3
        n_pairs = round(conc_M * N_A * vol * 1e-27) if spec['cation_symbol'] else 0
        n_water = round(WATER_NUMBER_DENSITY * vol) - n_pairs
    base_seed = seed_for_state(spec['state_id'])

    last_err = None
    for retry in range(MAX_PACK_RETRIES):
        rng = np.random.default_rng(base_seed + retry * 104729)
        packer = PeriodicPacker((side, side, side), periodic_z=True)
        max_attempts = 20000 * (retry + 1)

        symbols, positions, role, mol_id = [], [], [], []
        next_mol = 0
        try:
            if n_pairs:
                kinds = ([('ion', ('cation', spec['cation_symbol']))] * n_pairs +
                          [('ion', ('anion', spec['anion_symbol']))] * n_pairs)
                rng.shuffle(kinds)
                placed = pack_molecules(packer, rng, kinds, 0.0, side, max_attempts)
                for pts, label in placed:
                    symbols.append(label[1])
                    positions.append(pts[0].tolist())
                    role.append(label[0])
                    mol_id.append(next_mol)
                    next_mol += 1

            water_kinds = [('water', ('water', 'H2O'))] * n_water
            placed = pack_molecules(packer, rng, water_kinds, 0.0, side, max_attempts)
            for pts, label in placed:
                symbols.extend(['O', 'H', 'H'])
                positions.extend(pts.tolist())
                role.extend(['water', 'water', 'water'])
                mol_id.extend([next_mol, next_mol, next_mol])
                next_mol += 1
            break
        except PackingStuckError as e:
            last_err = e
            continue
    else:
        raise PackingStuckError(
            f"{spec['state_id']}: packing did not converge after "
            f"{MAX_PACK_RETRIES} full-fill retries; last error: {last_err}"
        )

    cell = np.diag([side, side, side])
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    fixed = np.zeros(len(atoms), dtype=int)
    atoms.set_array('fixed', fixed)
    atoms.set_array('role', np.array(role, dtype=object))
    atoms.set_constraint(FixAtoms(indices=[]))

    meta = dict(
        area_A2=side * side, z_slab_top_A=0.0, z_film_top_A=side,
        cell_z_A=side, n_water=n_water, n_pairs=n_pairs,
        n_cation=n_pairs, n_anion=n_pairs, n_Ag=0, n_atoms=len(atoms),
        mol_id=np.array(mol_id),
    )
    return atoms, meta


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def min_periodic_distance(atoms, mol_id, periodic_z):
    pos = atoms.positions
    Lx, Ly, Lz = atoms.cell[0, 0], atoms.cell[1, 1], atoms.cell[2, 2]
    shifts_xy = [-1, 0, 1]
    shifts_z = [-1, 0, 1] if periodic_z else [0]
    n = len(pos)
    best = np.inf
    if HAVE_SCIPY:
        tiled_pts = []
        tiled_orig = []
        for ix in shifts_xy:
            for iy in shifts_xy:
                for iz in shifts_z:
                    shift = np.array([ix * Lx, iy * Ly, iz * Lz])
                    tiled_pts.append(pos + shift)
                    tiled_orig.append(np.arange(n))
        tiled_pts = np.vstack(tiled_pts)
        tiled_orig = np.concatenate(tiled_orig)
        tree = cKDTree(tiled_pts)
        pairs = tree.query_pairs(r=MIN_DIST_A + 0.3, output_type='ndarray')
        if len(pairs) == 0:
            return best
        for i, j in pairs:
            oi, oj = tiled_orig[i], tiled_orig[j]
            if oi == oj:
                continue
            if mol_id[oi] == mol_id[oj]:
                continue  # intramolecular, excluded from the floor check
            d = np.linalg.norm(tiled_pts[i] - tiled_pts[j])
            if d < best:
                best = d
    else:
        # brute-force fallback, chunked over shift images
        for ix in shifts_xy:
            for iy in shifts_xy:
                for iz in shifts_z:
                    shift = np.array([ix * Lx, iy * Ly, iz * Lz])
                    shifted = pos + shift
                    for i in range(n):
                        d = np.linalg.norm(shifted[i] - pos, axis=1)
                        d[i] = np.inf
                        mask = mol_id != mol_id[i]
                        if not np.any(mask):
                            continue
                        dmin = np.min(d[mask])
                        if dmin < best:
                            best = dmin
    return best


def validate_state(atoms, meta, spec, film_A, vacuum_A):
    issues = []
    periodic_z = spec['kind'].startswith('bulk')

    dmin = min_periodic_distance(atoms, meta['mol_id'], periodic_z)
    if dmin < MIN_DIST_A:
        issues.append(f"min interatomic distance {dmin:.3f} A < {MIN_DIST_A} A")

    if spec['kind'] == 'interface':
        target = meta['n_water']
        # placed count is exact by construction; sanity check anyway
        actual = np.sum(np.array(atoms.get_array('role')) == 'water') // 3
        if target > 0 and abs(actual - target) / target > 0.02:
            issues.append(f"n_water {actual} deviates >2% from target {target}")

        cation_actual = np.sum(np.array(atoms.get_array('role')) == 'cation')
        anion_actual = np.sum(np.array(atoms.get_array('role')) == 'anion')
        if cation_actual != meta['n_cation'] or anion_actual != meta['n_anion']:
            issues.append(
                f"ion counts mismatch: cation {cation_actual}/{meta['n_cation']}, "
                f"anion {anion_actual}/{meta['n_anion']}"
            )

        fixed = atoms.get_array('fixed')
        role = atoms.get_array('role')
        slab_z = atoms.positions[role == 'slab', 2]
        uniq_z = np.sort(np.unique(np.round(slab_z, 3)))
        bottom_two = set(uniq_z[:2].tolist())
        for i in range(len(atoms)):
            if fixed[i] == 1:
                if role[i] != 'slab' or round(atoms.positions[i, 2], 3) not in bottom_two:
                    issues.append(f"fixed atom {i} not in bottom 2 slab layers")
                    break

        required_cell_z = meta['z_slab_top_A'] + GAP_ABOVE_SLAB_A + film_A + vacuum_A
        if atoms.cell[2, 2] < required_cell_z - 1e-6:
            issues.append(
                f"cell z {atoms.cell[2,2]:.3f} < required {required_cell_z:.3f}"
            )

    return dmin, issues


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------
def write_state_files(state_dir, atoms, species_order):
    os.makedirs(state_dir, exist_ok=True)
    ase_write(os.path.join(state_dir, 'init.extxyz'), atoms, format='extxyz')
    poscar_atoms = reorder_by_species(atoms, species_order)
    ase_write(os.path.join(state_dir, 'POSCAR'), poscar_atoms, format='vasp',
              direct=False, sort=False, vasp5=True)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contcar', default=os.path.join(SCRIPT_DIR, '..', 'ag_rpbe_d3zero_CONTCAR'))
    p.add_argument('--a', type=float, default=None, help='override lattice constant (A)')
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--film', type=float, default=35.0)
    p.add_argument('--vacuum', type=float, default=15.0)
    p.add_argument('--conc', type=float, default=1.0)
    p.add_argument('--outdir', default=os.path.join(SCRIPT_DIR, 'structures'))
    p.add_argument('--only', default=None, help='regex filter on state_id')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--matrix', choices=('s0', 'lc', 'sl', 'lr1'), default='s0',
                   help="s0 = original S0 set (default); lc = 8x8 counter-ion-only set; "
                        "sl = Shin-like n=16 counter-ion-only set; lr1 = ~60 A bulk boxes. "
                        "Build lc/sl with --conc 0 and a separate --outdir.")
    p.add_argument('--sl-dn', default=None,
                   help="comma-separated |dN| list for --matrix sl (default 1,2,3,4,6,8,12,16,20,25)")
    p.add_argument('--sl-no-water', action='store_true', help="--matrix sl: omit the dN=0 pure-water states")
    p.add_argument('--prefix', default=None, help="state_id prefix for lc/sl (default LC / SL)")
    p.add_argument('--bulk-side', type=float, default=60.0, help="--matrix lr1: cubic box side (A)")
    args = p.parse_args()

    d_nn, a_derived = lattice_constant_from_contcar(args.contcar)
    a = args.a if args.a is not None else a_derived
    print(f"[lattice] CONTCAR: {args.contcar}")
    print(f"[lattice] d_NN = {d_nn:.6f} A, derived a = {a_derived:.6f} A"
          + (f" (OVERRIDDEN -> using a = {a:.6f} A)" if args.a is not None else " (using this value)"))

    if args.matrix == 's0':
        states = build_state_matrix(args.n)
    elif args.matrix == 'lc':
        states = build_lc_matrix(args.n)
        if args.prefix:
            states = [dict(s, state_id=s['state_id'].replace('LC_', args.prefix + '_', 1)) for s in states]
    elif args.matrix == 'sl':
        dn = tuple(int(x) for x in args.sl_dn.split(',')) if args.sl_dn else SL_DELTA_N
        states = build_sl_matrix(args.n, dn, include_water=not args.sl_no_water,
                                 prefix=args.prefix or 'SL')
    else:
        states = build_lr1_matrix()
    if args.matrix in ('lc', 'sl') and args.conc != 0:
        print(f"WARNING: --matrix {args.matrix} is meant to be built with --conc 0 (got {args.conc})",
              file=sys.stderr)
    if args.only:
        pat = re.compile(args.only)
        states = [s for s in states if pat.search(s['state_id'])]

    print(f"\n[matrix] {len(states)} states selected")
    if args.dry_run:
        for s in states:
            print(f"  {s['state_id']:45s} kind={s['kind']:12s} dN={s['delta_N']:+d} "
                  f"ion_start={str(s['ion_start']):8s} replica={s['replica']}")
        by_kind = {}
        for s in states:
            by_kind[s['kind']] = by_kind.get(s['kind'], 0) + 1
        print("\n[matrix] counts by kind:", by_kind)
        return

    import shutil
    packmol_path = shutil.which('packmol')
    packer_engine = 'packmol' if packmol_path else 'python_fallback'
    print(f"[packer] engine = {packer_engine}"
          + (f" ({packmol_path})" if packmol_path else " (packmol not found on PATH)"))

    templates_dir = os.path.join(args.outdir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    write_pdb_water(os.path.join(templates_dir, 'water.pdb'))
    write_pdb_ion(os.path.join(templates_dir, 'na.pdb'), 'Na')
    write_pdb_ion(os.path.join(templates_dir, 'cs.pdb'), 'Cs')
    write_pdb_ion(os.path.join(templates_dir, 'f.pdb'), 'F')

    manifest = []
    t0 = time.time()
    validation_rows = []
    spotcheck = {}

    for spec in states:
        sid = spec['state_id']
        state_dir = os.path.join(args.outdir, sid)
        seed = seed_for_state(sid)

        if spec['kind'] == 'interface':
            atoms, meta = build_interface_state(spec, a, args.film, args.vacuum,
                                                 args.conc, '../templates')
            species_order = [s for s in ('Ag', 'O', 'H', spec['cation_symbol'], spec['anion_symbol']) if s]
            sigma_ion = sigma_per_ion_uC_cm2(meta['area_A2'])
            sigma_nom = spec['delta_N'] * sigma_ion

            os.makedirs(state_dir, exist_ok=True)
            cat_pdb = 'na.pdb' if spec['cation_symbol'] == 'Na' else 'cs.pdb'
            boxes = []
            z_lo = meta['z_slab_top_A'] + GAP_ABOVE_SLAB_A
            z_hi = meta['z_film_top_A']
            n_excess = abs(spec['delta_N'])
            if n_excess:
                if spec['ion_start'] == 'surface':
                    lo, hi = z_lo, z_lo + 3.0
                elif spec['ion_start'] == 'far':
                    lo, hi = meta['z_slab_top_A'] + 12.0, z_hi
                else:
                    lo, hi = z_lo, z_hi
                tmpl = cat_pdb if spec['delta_N'] < 0 else 'f.pdb'
                boxes.append(dict(template=tmpl, number=n_excess,
                                   box=(0, 0, lo, atoms.cell[0, 0], atoms.cell[1, 1], hi),
                                   comment=f"excess {'cation' if spec['delta_N']<0 else 'anion'} "
                                           f"({spec['ion_start']} start)"))
            boxes.append(dict(template=cat_pdb, number=meta['n_pairs'],
                               box=(0, 0, z_lo, atoms.cell[0, 0], atoms.cell[1, 1], z_hi),
                               comment="neutral cation pairs (random)"))
            boxes.append(dict(template='f.pdb', number=meta['n_pairs'],
                               box=(0, 0, z_lo, atoms.cell[0, 0], atoms.cell[1, 1], z_hi),
                               comment="neutral anion pairs (random)"))
            boxes.append(dict(template='water.pdb', number=meta['n_water'],
                               box=(0, 0, z_lo, atoms.cell[0, 0], atoms.cell[1, 1], z_hi),
                               comment="bulk waters (random)"))
            build_packmol_inp(state_dir, '../templates', seed, boxes)

            dmin, issues = validate_state(atoms, meta, spec, args.film, args.vacuum)
            write_state_files(state_dir, atoms, species_order)

            manifest.append(dict(
                state_id=sid, slab_layers=spec['slab_layers'], n_lateral=spec['n_lateral'],
                electrolyte=spec['electrolyte'], delta_N=spec['delta_N'],
                n_cation=meta['n_cation'], n_anion=meta['n_anion'], n_water=meta['n_water'],
                n_Ag=meta['n_Ag'], n_atoms=meta['n_atoms'], ion_start=spec['ion_start'],
                replica=spec['replica'], lattice_constant_A=a, area_A2=meta['area_A2'],
                sigma_nominal_uC_cm2=sigma_nom, cell=atoms.cell[:].tolist(),
                z_slab_top_A=meta['z_slab_top_A'], z_film_top_A=meta['z_film_top_A'],
                purpose=spec['purpose'], packer=packer_engine,
            ))
            validation_rows.append((sid, meta['n_atoms'], dmin, issues))

            if spec['ion_start'] == 'surface' and 'surface' not in spotcheck:
                spotcheck['surface'] = (sid, meta['excess_ion_z'])
            if spec['ion_start'] == 'far' and 'far' not in spotcheck:
                spotcheck['far'] = (sid, meta['excess_ion_z'])

        else:
            if args.matrix == 'lr1':
                atoms, meta = build_bulk_state(spec, bulk_side=args.bulk_side, conc_M=args.conc)
            else:
                atoms, meta = build_bulk_state(spec)
            species_order = ['O', 'H']
            if meta['n_pairs']:
                species_order = ['O', 'H', spec['cation_symbol'], spec['anion_symbol']]
            sigma_nom = 0.0

            os.makedirs(state_dir, exist_ok=True)
            boxes = []
            side = atoms.cell[0, 0]
            if meta['n_pairs']:
                boxes.append(dict(template='na.pdb' if spec['cation_symbol'] == 'Na' else 'cs.pdb',
                                   number=meta['n_pairs'], box=(0, 0, 0, side, side, side),
                                   comment=f"{spec['cation_symbol']} cations"))
                boxes.append(dict(template='f.pdb', number=meta['n_pairs'],
                                   box=(0, 0, 0, side, side, side),
                                   comment="F anions"))
            boxes.append(dict(template='water.pdb', number=meta['n_water'],
                               box=(0, 0, 0, side, side, side),
                               comment="bulk waters"))
            build_packmol_inp(state_dir, '../templates', seed, boxes)

            dmin, issues = validate_state(atoms, meta, spec, 0.0, 0.0)
            write_state_files(state_dir, atoms, species_order)

            manifest.append(dict(
                state_id=sid, slab_layers=None, n_lateral=None,
                electrolyte=spec['electrolyte'], delta_N=0,
                n_cation=meta['n_cation'], n_anion=meta['n_anion'], n_water=meta['n_water'],
                n_Ag=0, n_atoms=meta['n_atoms'], ion_start=None,
                replica=spec['replica'], lattice_constant_A=a, area_A2=meta['area_A2'],
                sigma_nominal_uC_cm2=0.0, cell=atoms.cell[:].tolist(),
                z_slab_top_A=0.0, z_film_top_A=meta['z_film_top_A'],
                purpose=spec['purpose'], packer=packer_engine,
            ))
            validation_rows.append((sid, meta['n_atoms'], dmin, issues))

        print(f"  built {sid} ({meta['n_atoms']} atoms)"
              + ("  ISSUES: " + "; ".join(issues) if issues else ""))

    wall = time.time() - t0

    with open(os.path.join(args.outdir, 'manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\n[done] {len(states)} states written in {wall:.1f} s "
          f"(packer={packer_engine})")

    print("\n[validation summary]")
    n_fail = 0
    for sid, natoms, dmin, issues in validation_rows:
        status = "OK" if not issues else "FAIL"
        if issues:
            n_fail += 1
        print(f"  {sid:45s} n_atoms={natoms:5d}  min_dist={dmin:6.3f} A  {status}")
        for iss in issues:
            print(f"      - {iss}")
    print(f"\n[validation] {len(validation_rows) - n_fail}/{len(validation_rows)} states passed")

    if spotcheck:
        print("\n[spot check] excess-ion z coordinates")
        for k, (sid, zs) in spotcheck.items():
            print(f"  {k}: {sid} -> z = {['%.2f' % z for z in zs]}")

    if n_fail:
        print("\nFAILED validation on one or more states; see issues above.", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
