import os.path as op
import numpy as np
import pandas as pd


def read_plumed_file(filename, cv_columns=None, bias_column=None):
    """
    Read PLUMED output files into pandas DataFrame.

    Column names are parsed from the header of the Plumed file (e.g. COLVAR or HILLS)
    and indices are taken from the time column of the Plumed file.

    Parameters
    ----------
    filename : string
        Name of the plumed file that contains collective variable data
        (e.g. HILLS or COLVAR).

    Returns
    -------
    data : pd.DataFrame
        Pandas DataFrame with column names parsed from CV/bias labels in the plumed
        file header and time column used for the index.

    Examples
    --------
    Read COLVAR file from a MetaD simulation into a DataFrame named metad_traj.

    >>> metad_traj = read_plumed_file('COLVAR')
    >>> metad_traj.head()
    # todo: finish example with output
    """
    if filename is None:
        return None

    filename = op.abspath(filename)
    with open(filename, "r") as f:
        header = f.readline().strip().split(" ")[2:]

    data = pd.read_csv(
        filename, comment="#", names=header, delimiter="\s+", index_col=0
    )

    if cv_columns is None and bias_column is None:
        return data

    if bias_column is None:
        return data[cv_columns]

    if cv_columns is None:
        return data[bias_column]

    return data[cv_columns], data[bias_column]


def reweight_ct(rbias, temp=300):
    """
    Calculate frame weights, per Tiwary and Parinello, JCPB 2015 (c(t) method)

    Reweighting biased MD trajectory to unbiased probabilty along
    a given reaction coordinate. Using rbias column from COLVAR to
    perform reweighting per Tiwary and Parinello

    Parameters
    ----------
    rbias : np.ndarray
        Array of Vbias - c(t) values associated with each timestep in a metadynamis
        biased simulation. Calculated automatically in PLUMED when the
        REWEIGHTING_NGRID and associated arguments are added to MetaD.
    temp : float, 300
        Temperature of the simulation, in K.

    Returns
    -------
    np.ndarray
        Weight for each frame associated with rbias array supplied to the funciton.
    """
    k = 8.314e-3
    kt = temp * k

    # ensure rbias is an ndarray
    rbias = np.array(rbias)
    return np.exp(rbias / kt)
