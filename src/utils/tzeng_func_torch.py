import torch


def parameterize_myelin_rsfc(myelin, rsfc_gradient, param_coef):
    """
    From the 10 parameterization parameters to get the 205 parameters
    :param myelin: [N, 1]
    :param rsfc_gradient: [N, 1]
    :param param_coef: [3p + 1, param_sets], where p is the number of parameterization variables
    :return: parameters for 2014 Deco Model [3N+1, param_sets]
    """
    w_EE = param_coef[
        0] + param_coef[1] * myelin + param_coef[2] * rsfc_gradient
    w_EI = param_coef[
        3] + param_coef[4] * myelin + param_coef[5] * rsfc_gradient
    G = param_coef[6]
    sigma = param_coef[
        7] + param_coef[8] * myelin + param_coef[9] * rsfc_gradient
    return torch.vstack((w_EE, w_EI, G, sigma))


def tzeng_KS_distance(s_cdf, t_cdf=None):
    """Compute KS distance within a CDF set or between two CDF sets

    Args:
        s_cdf (tensor): [s_labels, bins]
        t_cdf (tensor, optional): [t_labels, bins]. If None, will compute within s_cdf

    Returns:
        tensor: distance matrix, [s_labels, t_labels]
    """
    s_cdf = s_cdf / s_cdf[:, -1].unsqueeze(1)
    if t_cdf is None:
        t_cdf = s_cdf
    else:
        t_cdf = t_cdf / t_cdf[:, -1].unsqueeze(1)
    distance_matrix = torch.zeros(s_cdf.shape[0], t_cdf.shape[0])
    for k in range(distance_matrix.shape[0]):
        ks_tmp = torch.abs(t_cdf - s_cdf[k].unsqueeze(0))  # [xxx, bins]
        ks_tmp = torch.amax(ks_tmp, dim=1)  # [xxx]
        distance_matrix[k] = ks_tmp
    return distance_matrix


def test_main():
    s = torch.rand(3, 4)
    t = torch.rand(4, 4)
    s_cdf = torch.cumsum(s, dim=1)
    t_cdf = torch.cumsum(t, dim=1)
    distance_mat = tzeng_KS_distance(s_cdf, t_cdf)
    print(distance_mat)


if __name__ == "__main__":
    test_main()
