import itertools
import numpy as np
import pandas as pd
import mdtraj as md


# ####################
# calculate contacts #
# ####################
def calculate_contacts(traj, selection, selection2=None, cutoff=0.4):
    top = traj.topology
    # define possible salt bridges
    sel1_atoms = top.select(selection)
    sel2_atoms = top.select(selection)
    if selection2 is not None:
        sel2_atoms = top.select(selection2)
    # create list of atom pairs
    atom_pairs = np.array(
        list(itertools.product(sel1_atoms, sel2_atoms))
    )
    # calculate pairwise distances
    distances = pd.DataFrame(
        md.compute_distances(traj, atom_pairs=atom_pairs, periodic=False),
        index=traj.time,
    )
    # return boolean matrix of contacts in each frame
    contacts = (distances < cutoff)
    return contacts, atom_pairs
