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


# ##########
# do dssp #
# ##########
def structure_contact_fraction(traj, contacts, atom_pairs, simplified=True):
    structure = pd.DataFrame(
        md.compute_dssp(traj),
        index=traj.time,
    )

    helical_mask = (structure == "H")  # for fraction, add: .mean(axis=1)
    extended_mask = (structure == "E")  # for fraction, add: .mean(axis=1)
    coil_mask = (structure == "C")  # for fraction, add: .mean(axis=1)
    # residue-wise salt bridge + structure
    contacts_mask = pd.DataFrame(
        np.zeros_like(structure, dtype=bool),
        index=traj.time,
    )
    # for each frame, set True where salt bridge occurs
    for t in contacts.index:
        # TODO: vectorize once i figure out how to slice mdtraj topology
        for atom_idx in atom_pairs[np.where(contacts.loc[t, :])]:
            for atom in atom_idx:
                res_id = traj.top.atom(atom).residue.index
                contacts_mask.loc[t, res_id] = True
    # find where salt bridge AND given structure is true
    helix_and_salt_bridge = (helical_mask * contacts_mask).sum(axis=1)
    extended_and_salt_bridge = (extended_mask * contacts_mask).sum(axis=1)
    coil_and_salt_bridge = (coil_mask * contacts_mask).sum(axis=1)

    # filter to only helical frames and reweight fraction
    nonzero_helix_frames = helical_mask.sum(axis=1).nonzero()
    helix_salt_bridge_frxn = (
            helix_and_salt_bridge.iloc[nonzero_helix_frames]
            / helical_mask.sum(axis=1).iloc[nonzero_helix_frames]
    )
    nonzero_extended_frames = extended_mask.sum(axis=1).nonzero()
    extended_salt_bridge_frxn = (
        extended_and_salt_bridge.iloc[nonzero_extended_frames]
        / extended_mask.sum(axis=1).iloc[nonzero_extended_frames]
    )
    nonzero_coil_frames = coil_mask.sum(axis=1).nonzero()
    coil_salt_bridge_frxn = (
        coil_and_salt_bridge.iloc[nonzero_coil_frames]
        / coil_mask.sum(axis=1).iloc[nonzero_coil_frames]
    )
    # TODO: instead, loop through structure types and build df to return
    return helix_salt_bridge_frxn, extended_salt_bridge_frxn, coil_salt_bridge_frxn


def do_dssp(traj, simplified=True):
    structure = pd.DataFrame(
        md.compute_dssp(traj, simplified=simplified),
        index=traj.time,
    )

    code_set = frozenset(
        frozenset(structure[col].unique()) for col in structure.columns
    )
    structure_codes = list({code for codes in code_set for code in codes})

    structure_frxn = pd.DataFrame(index=traj.time, columns=structure_codes)
    for code in structure_frxn.columns:
        structure_frxn[code] = (structure == code).mean(axis=1)
    return structure_frxn
