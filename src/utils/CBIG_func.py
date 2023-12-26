import numpy as np
import scipy.optimize as soptimize
import scipy.stats as stats


def CBIG_corr(s_series, t_series=None, need_pvalue=False):
    """Compute the Pearson correlation for s_series (self_correlation) or between s and t.
    [IMPORTANT] To remain the same as Matlab version, features are in axis=0 (intrinsically vertical in matrix) and labels are in axis=1

    Args:
        s_series (array): [features, labels]
        t_series (array, optional): [features, t_labels]. Defaults to None.
        need_pvalue (boolean, optional): whether to return p_value. Defaults to False.

    Returns:
        array: [labels, t_labels]
    """
    s_series = s_series - np.mean(s_series, axis=0)
    s_series = s_series / np.linalg.norm(s_series, axis=0)
    if t_series is None:
        corr_matrix = np.matmul(np.transpose(s_series), s_series)
    else:
        t_series = t_series - np.mean(t_series, axis=0)
        t_series = t_series / np.linalg.norm(t_series, axis=0)
        corr_matrix = np.matmul(np.transpose(s_series), t_series)
    if need_pvalue:
        n = s_series.shape[0]
        dist = stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
        p_matrix = np.zeros_like(corr_matrix)
        for i in range(p_matrix.shape[0]):
            for j in range(p_matrix.shape[1]):
                p_matrix[i, j] = 2 * dist.cdf(-abs(corr_matrix[i, j]))
        return corr_matrix, p_matrix
    else:
        return corr_matrix


def CBIG_HungarianClusterMatch(ref_labels, input_labels, disp_flag=False):
    num_ref_labels = len(np.unique(ref_labels))
    num_input_labels = len(np.unique(input_labels))
    ref_labels = ref_labels - 1
    input_labels = input_labels - 1

    # Build matching matrix
    mat = np.zeros((num_input_labels, num_ref_labels))
    for i in range(num_ref_labels):
        for j in range(num_input_labels):
            mat[j, i] = -np.sum(
                np.float64(ref_labels == i) * np.float64(input_labels == j))

    row_ind, col_ind = soptimize.linear_sum_assignment(mat)

    output_labels = np.zeros_like(input_labels)
    for i in range(num_input_labels):
        output_labels[input_labels == i] = col_ind[i]

    cost = np.sum(mat[row_ind, col_ind])
    if disp_flag:
        print(f"Overlap: {-cost} of total {len(ref_labels)}.")

    return output_labels + 1, col_ind


@DeprecationWarning
def CBIG_nanmean(x, dim=0):
    """Compute the mean while excluding NaN entries. In python, you can use np.nanmean instead

    Args:
        x (array): data array
        dim (int, optional): along dimension dim. Defaults to 0.

    Returns:
        m (array): mean excluding NaN
    """
    if dim == 0 and x.shape[0] == 1:
        dim = np.nonzero(x.shape > 1)[0]

    # find nan and set it to zero, then sum
    nanind = np.isnan(x)
    x[nanind] = 0
    xsum = np.sum(x, dim)

    # count not nan entries
    count = x.shape[dim] - np.sum(nanind, dim)
    m = xsum / count
    return m


def CBIG_StableAtanh(x, ensure_real=True):
    """Stable version of atanh. It constrains the input within [-1 1] and output within [atanh(-1+eps) atanh(1-eps)]
    Here eps means the smallest accuracy of float

    Args:
        x (array): any shape
        ensure_real (bool, optional): if your data contains no complex number, it will speed up. Defaults to True.

    Returns:
        like x: the array after arctanh
    """
    x[x > (1 - np.finfo(float).eps)] = 1 - np.finfo(float).eps
    x[x < (-1 + np.finfo(float).eps)] = -1 + np.finfo(float).eps
    x = np.arctanh(x)
    # x[np.logical_and(np.isinf(x), x > 0)] = np.arctanh(1-np.finfo(float).eps)
    # x[np.logical_and(np.isinf(x), x < 0)] = np.arctanh(-1+np.finfo(float).eps)
    if not ensure_real:
        x[np.logical_and(np.logical_not(np.isreal(x)),
                         np.real(x) > 0)] = np.arctanh(1 - np.finfo(float).eps)
        x[np.logical_and(np.logical_not(np.isreal(x)),
                         np.real(x)
                         < 0)] = np.arctanh(-1 + np.finfo(float).eps)
    return x


def test_main():
    ref = np.array([2, 1, 3, 2, 1])
    nee = np.array([3, 1, 2, 3, 1])
    print(CBIG_HungarianClusterMatch(ref, nee, True))


if __name__ == "__main__":
    test_main()
