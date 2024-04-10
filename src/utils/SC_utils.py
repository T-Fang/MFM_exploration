import os
import torch
from src.basic.constants import SPLIT_NAMES, NUM_GROUP_IN_SPLIT, PROJECT_DIR
from pandas import DataFrame, Series
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_path_to_group(split_name, group_index):
    group_index = str(group_index)
    path_to_pMFM_input = os.path.join(PROJECT_DIR,
                                      'dataset_generation/input_to_pMFM/')
    path_to_group = os.path.join(path_to_pMFM_input, split_name, group_index)
    return path_to_group


def get_path_to_group_SC(split_name, group_index, use_SC_with_diag=False):
    path_to_group = get_path_to_group(split_name, group_index)
    file_name = 'group_level_SC_with_diag.csv' if use_SC_with_diag else 'group_level_SC.csv'
    path_to_group_SC = os.path.join(path_to_group, file_name)
    return path_to_group_SC


def load_group_SC(split_name, group_index, use_SC_with_diag=False):
    SC_path = get_path_to_group_SC(split_name, group_index, use_SC_with_diag)
    SC = pd.read_csv(SC_path, header=None)
    return df_to_tensor(SC)


def get_all_group_SC():
    all_group_SC = []
    for split_name in SPLIT_NAMES:
        all_group_SC.extend(get_SC_in_split(split_name))
    return all_group_SC


def get_SC_in_split(split_name):
    SCs_in_split = []
    for group_index in range(1, NUM_GROUP_IN_SPLIT[split_name] + 1):
        print(f'Loading SC for {split_name} {group_index}')
        group_SC = load_group_SC(split_name, group_index)
        SCs_in_split.append(group_SC)
    return SCs_in_split


def df_to_tensor(df: DataFrame, device=None):
    np_df = matrix_to_np(df)
    df_tensor = torch.tensor(np_df, dtype=torch.float32, device=device)
    return df_tensor


def matrix_to_np(matrix):
    if isinstance(matrix, DataFrame):
        matrix = matrix.to_numpy()
    elif isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    elif isinstance(matrix, Series):
        matrix = matrix.to_numpy()
    return matrix


def get_triu_np_vector(matrix):
    np_matrix = matrix_to_np(matrix)
    n = np_matrix.shape[0]
    triu_vector = np_matrix[np.triu_indices(n, 1)]
    return triu_vector


def get_triu_torch_vector(matrix: torch.Tensor):
    n = matrix.shape[0]
    return matrix[np.triu_indices(n, 1)]


def batched_SC_to_triu_vector(batched_SC: torch.Tensor):
    return torch.stack([get_triu_torch_vector(SC) for SC in batched_SC])


def corr_between_matrices(matrices):
    matrices_triu_vectors = [get_triu_np_vector(matrix) for matrix in matrices]
    return np.corrcoef(matrices_triu_vectors)


def show_heatmap(matrix, filename=None):
    sns.heatmap(matrix)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename, dpi=400)
        plt.imshow(plt.imread(filename))


def show_corr_between_all_SCs(
        filename=f'{PROJECT_DIR}reports/figures/corr_between_all_SCs.png'):
    SCs = get_all_group_SC()
    corr_matrix = corr_between_matrices(SCs)
    np.savetxt('./corr_between_all_SCs.csv', corr_matrix, delimiter=',')
    show_heatmap(corr_matrix, filename=filename)


def group_SC_matrices(sc_mats):
    """group SC matrices like that in matlab. Neglect 'nan' entries.
        Those entries with lower than half sc_mats > 0 will be set to 0.
        Scale the final SC by log function. Set the diagonal to be 0.

    Args:
        sc_mats (np.array): [n_mats, n_roi, n_roi]

    Returns:
        grouped_sc (np.array): the group average of the input sc_mats. [n_roi, n_roi]
    """
    n_mats = sc_mats.shape[0]
    n_roi = sc_mats.shape[1]
    grouped_sc = np.zeros((n_roi, n_roi))

    for i in range(n_roi):
        for j in range(n_roi):

            if i == j:  # if i == j, grouped_sc[i, j] = 0;
                continue

            count_non_zero = 0  # To count the number of non_zero
            sum_temp = 0
            for mat_i in range(n_mats):  # For each SC matrix
                if np.isnan(sc_mats[mat_i, i, j]):
                    continue

                if sc_mats[mat_i, i, j] != 0:
                    count_non_zero += 1
                    sum_temp += sc_mats[mat_i, i, j]
            # End for mat_i
            # Only when there are over half of the SC[i, j] are not zero
            if count_non_zero >= 0.5 * n_mats:
                grouped_sc[i, j] = np.log(sum_temp / count_non_zero)

    return grouped_sc
