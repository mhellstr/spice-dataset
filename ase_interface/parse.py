#!/usr/bin/env python
from __future__ import annotations

import ase
import h5py
import numpy as np

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write
from typing import Optional, Iterable
from pathlib import Path


def load_hdf5_data(
    filename: str | Path,
    N: int = 10000,
    allowed_atomic_numbers: Iterable[int] | None = None,
    fraction: float = 0.08,
    max_force: float = 15.0,  # eV/angstrom
    allowed_charges: Iterable[int] | None = None,
    energy_key: Literal["formation_energy", "dft_total_energy"] = "formation_energy",
) -> list[ase.Atoms]:
    """
    filename: str
        Path to .hdf5 file

    N: int
        Maximum number of structures to extract

    allowed_atomic_numbers: Iterable[int] or None
        Set of allowed aotmic numbers. For example: ``allowed_atomic_numbers={1, 6, 7, 8}``.
        If None, allow all atomic numbers.

    fraction: float
        Probability to include a structure in the returned list

    max_force: float
        Maximum allowed force component (in eV/angstrom)

    allowed_charges: Iterable[int] or None
        Set of allowed total charges. For example ``allowed_charges={-1, 0, 1}``. If None, allow all charges.

    energy_key: str
        Whether to use "formation_energy" or "dft_total_energy" for the Atoms.get_potential_energy()

    """

    data = dict()

    check_atomic_numbers = True
    if allowed_atomic_numbers is None:
        check_atomic_numbers = False
    else:
        allowed_atomic_numbers = set(allowed_atomic_numbers)

    check_total_charges = True
    if allowed_charges is None:
        check_total_charges = False
    else:
        allowed_charges = set(allowed_charges)

    atoms_list: list[ase.Atoms] = []  # will be returned

    with h5py.File(filename, "r") as f:
        for group_name in f.keys():
            if len(atoms_list) >= N:
                break

            group = f[group_name]

            atomic_numbers = group["atomic_numbers"][:]
            if check_atomic_numbers and not set(atomic_numbers) <= allowed_atomic_numbers:
                continue

            conformations = group["conformations"][:] * ase.units.Bohr
            dft_total_energy = group["dft_total_energy"][:] * ase.units.Hartree
            dft_total_gradient = group["dft_total_gradient"][:]
            dft_total_force = -dft_total_gradient * ase.units.Hartree / ase.units.Bohr
            formation_energy = group["formation_energy"][:] * ase.units.Hartree
            mbis_charges = group["mbis_charges"][:]
            scf_dipole = group["scf_dipole"][:] * ase.units.Bohr
            smiles = group["smiles"][0].decode("utf-8")
            subset = group["subset"][0].decode("utf-8")
            for i in range(len(conformations)):
                if np.random.rand() > fraction:
                    continue

                total_charge = np.sum(mbis_charges[i])
                total_charge = int(np.round(total_charge))

                if check_total_charges and total_charge not in allowed_charges:
                    print(f"Ignoring SMILES {smiles} - total charge {total_charge}")
                    continue

                this_max_force = np.max(np.abs(dft_total_force[i]))
                if this_max_force > max_force:
                    print(
                        f"Ignoring SMILES {smiles} - "
                        f"max force {this_max_force:.3f} eV/ang (threshold {max_force:.3f} eV/ang)"
                    )
                    continue

                print(f"Configuration {len(atoms_list)}")
                atoms = ase.Atoms(positions=conformations[i], numbers=atomic_numbers)
                atoms.set_calculator(SinglePointCalculator(atoms))
                atoms.calc.results["energy"] = formation_energy[i]  # eV
                atoms.calc.results["forces"] = dft_total_force[i]  # eV/angstrom
                atoms.calc.results["dipole"] = scf_dipole[i]  # e*angstrom
                atoms.calc.results["charges"] = mbis_charges[i]  # atomic units
                atoms.info["charge"] = total_charge  # atomic units
                atoms.info["formation_energy"] = formation_energy[i]  # eV
                atoms.info["dft_total_energy"] = dft_total_energy[i]  # eV
                atoms.info["smiles"] = smiles
                atoms.info["subset"] = subset
                atoms_list.append(atoms)

    return atoms_list
