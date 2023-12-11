import torch
# import CBIG_func


def CBIG_corr(s_series, t_series=None):
    """Compute the Pearson correlation for s_series (self_correlation) or between s and t.
    [IMPORTANT] To remain the same as Matlab version, features are in dim=0 (intrinsically vertical in matrix) and labels are in dim=1

    Args:
        s_series (tensor): [features, labels]
        t_series (tensor, optional): [features, t_labels]. Defaults to None.

    Returns:
        array: [labels, labels] or [labels, t_labels]
    """
    s_series = s_series - torch.mean(s_series, dim=0)
    s_series = s_series / torch.linalg.norm(s_series, dim=0)
    if t_series is None:
        return torch.matmul(torch.transpose(s_series, 0, 1), s_series)
    else:
        t_series = t_series - torch.mean(t_series, dim=0)
        t_series = t_series / torch.linalg.norm(t_series, dim=0)
        return torch.matmul(torch.transpose(s_series, 0, 1), t_series)
    
    
def adj_matrix_to_tensor(adj_mat, normalize=True, add_self_loop=True):
    """ Input 2D symmetric adjacency matrix, return tensors for constructing DGL graphs.
    [IMPORTANT] The adjacency matrix's entries must be larger or equal to 0

    Args:
        adj_mat (array or tensor): [N, N], sometimes may be [N, M]
        normalize (bool, optional): If true, constrain the values in adj_mat to 0-1. Defaults to True.
        add_self_loop (bool, optional): If true, add self loop to the graph (fill the diagonal line with 1). Defaults to True.

    Returns:
        tensors: src_tensor (the starting nodes' tensor), dst_tensor, edge_tensor
    """
    adj_mat = torch.as_tensor(adj_mat)
    if normalize:
        adj_mat = adj_mat / torch.max(adj_mat)
    if add_self_loop:
        adj_mat.fill_diagonal_(1)
    src_list = []
    dst_list = []
    edge_weight = []
    for i in range(adj_mat.shape[0]):
        for j in range(i, adj_mat.shape[1]):
            if adj_mat[i, j] > 0:
                src_list.append(i)
                dst_list.append(j)
                edge_weight.append(adj_mat[i][j])
    src_tensor = torch.as_tensor(src_list)
    dst_tensor = torch.as_tensor(dst_list)
    edge_tensor = torch.as_tensor(edge_weight)
    # shape: [xxx,]
    return src_tensor, dst_tensor, edge_tensor


def test_main():
    x = torch.randn(3, 4)
    y = CBIG_corr(x)
    z = torch.corrcoef(x.T)
    print(y)
    print(z)
    
    
if __name__ == "__main__":
    test_main()
