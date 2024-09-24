#!/usr/bin/env python
from parse import load_hdf5_data
from ase.io import write as ase_write


if __name__ == "__main__":
    filename = "SPICE.hdf5"
    atoms_list = load_hdf5_data(
        filename, 
        N=1000, 
        allowed_atomic_numbers={1, 6, 8, 7},  # H, C, O, N
        allowed_charges={0},  # only neutral molecules
        fraction=0.02
    )
    ase_write("training_set.xyz", atoms_list)
