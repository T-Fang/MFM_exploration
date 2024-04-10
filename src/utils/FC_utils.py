import numpy as np
from src.utils.CBIG_func import CBIG_StableAtanh


def fisher_average(corr_mat):
    """Average by Fisher transformation

    Args:
        corr_mat (ndarray): [number, ...]
    Returns:
        corr_ave: [...]
    """
    z_corr_mat = CBIG_StableAtanh(corr_mat)
    z_corr_ave = np.nanmean(z_corr_mat, axis=0)
    corr_ave = np.tanh(z_corr_ave)
    return corr_ave
