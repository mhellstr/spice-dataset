#!/usr/bin/env python
from parse import load_hdf5_data
from ase.io import write as ase_write, read as ase_read


if __name__ == "__main__":
    filename = "SPICE.hdf5"
    atoms_list = load_hdf5_data(
        filename, 
        N=1000, 
        allowed_atomic_numbers={1, 6, 8, 7},  # H, C, O, N
        allowed_charges={0},  # only neutral molecules
        fraction=0.02
    )
    # write to disk
    ase_write("training_set.xyz", atoms_list)

    for i, atoms in enumerate(atoms_list):
        print(f"Structure {i}")
        print(f"{atoms.get_potential_energy()=}")  # eV
        print(f"{atoms.get_forces()=}")  # eV/angstrom
        print(f"{atoms.get_dipole_moment()=}")  # e*angstrom
        print(f"{atoms.get_charges()=}")  # atomic units
        print(f"{atoms.info['charge']=}")  # total charge, atomic units
        print(f"{atoms.info['smiles']=}")

        break

    # restore the atoms list from disk
    new_atoms_list = ase_read("training_set.xyz", ":")
        
