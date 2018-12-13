import numpy as np
import mdtraj as md


def conformational_entropy(traj, bins=45, weights=None, density=True):
    #####################
    # calculate entropy #
    #####################
    phi = md.compute_phi(traj)
    psi = md.compute_psi(traj)

    ramachandrans = []
    for idx in range(phi[1].shape[1] - 1):
        x = phi[1][:, idx]
        y = psi[1][:, idx + 1]
        hist, _, _ = np.histogram2d(
            x, y, bins=bins, weights=weights, density=density,
        )
        s_j = -hist * np.ma.log(hist)
        ramachandrans.append(s_j.sum())
    # return residue-wise backbone conformational entropy
    return np.array(ramachandrans)
