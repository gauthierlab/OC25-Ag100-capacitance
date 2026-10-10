"""
ToyCalculator: a cheap, dependency-free ASE calculator used ONLY to test the
MD plumbing (run_md_oc25.py --calculator dummy) on a machine without
fairchem-core / a GPU. It is NOT a physically meaningful potential and must
never be used for production MD.

Model:
  - Harmonic "bonds" for O-H pairs and Ag-Ag nearest-neighbour pairs,
    detected ONCE from the starting geometry (topology is then frozen for
    the whole run -- this is a toy, it is not meant to allow bond breaking).
  - A short-ranged repulsive-only (WCA-like) pairwise term between all other
    atom pairs, to keep everything from overlapping.
Both terms use ASE's periodic neighbour list / minimum-image convention, so
random water displacement, Langevin kicks, etc. all produce finite,
well-behaved forces on a slab + water + ions cell.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list


class ToyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms, oh_cut=1.25, ag_cut=3.2, k_bond=8.0,
                 rep_sigma=1.6, rep_eps=0.02, rep_cut=3.2, **kwargs):
        super().__init__(**kwargs)
        self.k_bond = k_bond
        self.rep_sigma = rep_sigma
        self.rep_eps = rep_eps
        self.rep_cut = rep_cut
        self._build_bonds(atoms, oh_cut, ag_cut)

    def _build_bonds(self, atoms, oh_cut, ag_cut):
        syms = np.array(atoms.get_chemical_symbols())
        n = len(atoms)
        max_cut = max(oh_cut, ag_cut)
        if n > 1:
            i, j, d = neighbor_list("ijd", atoms, max_cut)
        else:
            i = j = d = np.array([], dtype=int)

        oh_mask = (syms[i] == "O") & (syms[j] == "H") & (d < oh_cut)
        agag_mask = (syms[i] == "Ag") & (syms[j] == "Ag") & (d < ag_cut)

        pairs, r0, seen = [], [], set()
        for mask in (oh_mask, agag_mask):
            for ii, jj, dd in zip(i[mask], j[mask], d[mask]):
                key = (min(int(ii), int(jj)), max(int(ii), int(jj)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(key)
                r0.append(dd)

        self.bond_pairs = np.array(pairs, dtype=int).reshape(-1, 2)
        self.bond_r0 = np.array(r0, dtype=float)
        self.bonded_set = seen

    def calculate(self, atoms=None, properties=("energy", "forces"),
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        n = len(self.atoms)
        forces = np.zeros((n, 3))
        energy = 0.0

        # --- bonded harmonic terms ---
        if len(self.bond_pairs):
            ii = self.bond_pairs[:, 0]
            jj = self.bond_pairs[:, 1]
            raw = pos[jj] - pos[ii]
            dvec, dist = find_mic(raw, cell, pbc)
            dist = np.maximum(dist, 1e-6)
            dr = dist - self.bond_r0
            energy += float(np.sum(0.5 * self.k_bond * dr ** 2))
            fmag = -self.k_bond * dr / dist
            fvec = fmag[:, None] * dvec  # force on jj along dvec (i->j)
            np.add.at(forces, jj, fvec)
            np.add.at(forces, ii, -fvec)

        # --- nonbonded short-range repulsion (WCA-like), skip bonded pairs ---
        if n > 1:
            i, j, d, D = neighbor_list("ijdD", self.atoms, self.rep_cut)
            if len(i):
                uniq = i < j
                i, j, d, D = i[uniq], j[uniq], d[uniq], D[uniq]
                if len(i):
                    drop = np.array([
                        (min(int(a), int(b)), max(int(a), int(b))) in self.bonded_set
                        for a, b in zip(i, j)
                    ])
                    keep = ~drop
                    i, j, d, D = i[keep], j[keep], d[keep], D[keep]
                if len(i):
                    d = np.maximum(d, 1e-3)
                    sr6 = (self.rep_sigma / d) ** 6
                    sr12 = sr6 ** 2
                    e_pair = 4.0 * self.rep_eps * sr12  # repulsive part only
                    energy += float(np.sum(e_pair))
                    fmag = 4.0 * self.rep_eps * 12.0 * sr12 / d
                    unit = D / d[:, None]
                    fi = fmag[:, None] * unit  # points i->j; repels i backward
                    np.add.at(forces, j, fi)
                    np.add.at(forces, i, -fi)

        self.results["energy"] = energy
        self.results["forces"] = forces
