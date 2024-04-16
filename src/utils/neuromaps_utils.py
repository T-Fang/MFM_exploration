import os
import pandas as pd
import numpy as np
from pca import pca
from matplotlib import pyplot as plt
import torch

from src.basic.constants import NEUROMAPS_SRC_DIR, NUM_DESIKAN_ROI, NUM_ROI, IGNORED_DESIKAN_ROI_ZERO_BASED, \
    DESIKAN_NEUROMAPS_DIR, FIGURES_DIR, DEFAULT_DTYPE
from src.utils.init_utils import get_device
from src.utils.analysis_utils import visualize_stats, concat_n_images
from src.utils.file_utils import convert_to_numpy


##################################################
# Convert Neuromaps to the Desikan-Killiany atlas
##################################################
def get_mapping_labels(source_space: str, resolution: str) -> np.ndarray:
    """
    Get the mapping labels from the given source space and resolution to the Desikan-Killiany atlas.
    If the labels are already generated, we can directly read it.
    Otherwise, we can generate the labels from lh and rh labels.

    Args:
        source_space: The source space of data to be mapped.
        resolution: The resolution of the `source_space`.

    Returns:
        The mapping labels from the given source space and resolution to the Desikan-Killiany atlas.
    """
    labels_file_path = os.path.join(
        NEUROMAPS_SRC_DIR, f"{source_space}_{resolution}_to_Desikan.csv")
    if os.path.exists(labels_file_path):
        labels = pd.read_csv(labels_file_path, header=None).values.flatten()
        return labels

    lh_labels = pd.read_csv(os.path.join(
        NEUROMAPS_SRC_DIR, f"{source_space}_{resolution}_lh_to_Desikan.csv"),
                            header=None).values.flatten()  # noqa: E126

    rh_labels = pd.read_csv(os.path.join(
        NEUROMAPS_SRC_DIR, f"{source_space}_{resolution}_rh_to_Desikan.csv"),
                            header=None).values.flatten()  # noqa: E126

    labels = np.concatenate([lh_labels, rh_labels + 36], axis=0)
    pd.DataFrame(labels).to_csv(labels_file_path, header=False, index=False)

    print(
        f"Mapping labels from {source_space}_{resolution} to Desikan-Killiany atlas generated and saved to {labels_file_path}"
    )
    return labels


def convert_to_desikan(data: np.ndarray,
                       mapping_labels: np.ndarray,
                       save_file_path: str | None = None) -> np.ndarray:
    """
    Convert the neuro map to Desikan-Killiany atlas.

    Args:
        data: The data to be converted.
        mapping_labels: The mapping labels from the source space to the Desikan-Killiany atlas.

    Returns:
        The converted data in the Desikan-Killiany atlas.
    """
    assert data.shape == mapping_labels.shape
    sum_in_ROI = np.zeros((NUM_DESIKAN_ROI + 1, ))
    count_in_ROI = np.zeros((NUM_DESIKAN_ROI + 1, ))
    for i, label in enumerate(mapping_labels):
        if data[i] == 0:
            continue
        sum_in_ROI[label] += data[i]
        count_in_ROI[label] += 1

    converted_data = sum_in_ROI[1:] / count_in_ROI[1:]
    converted_data = np.delete(converted_data, IGNORED_DESIKAN_ROI_ZERO_BASED)
    np.nan_to_num(converted_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    assert converted_data.shape[0] == NUM_ROI
    if save_file_path is not None:
        pd.DataFrame(converted_data).to_csv(save_file_path,
                                            header=False,
                                            index=False)
    return converted_data


##################################################
# PCA related
##################################################
def find_top_feat(data: pd.DataFrame,
                  max_exp_var_loss: float = 0.01) -> list[str]:
    """
    Iteratively remove the least informative neuro-map (removing which
        will result in the smallest reduction in the variance explained by the PC1)

    Stop when the PC1 explained variance drops for more than `max_exp_var_loss` when either of the remaining features is removed.

    Args:
        data: The data to be analyzed.
        max_exp_var_loss: The allowed maximum explained variance loss.

    Returns:
        The top features that can explain the variance of the data.
    """
    data = data.copy()

    model = pca(n_components=1, normalize=True, verbose=2)
    model.fit_transform(data, col_labels=data.columns)
    exp_var = model.results['variance_ratio'][0]

    top_features = data.columns.tolist()
    while True:
        exp_var_loss = []
        for feature in top_features:
            model = pca(n_components=1, normalize=True, verbose=2)
            model.fit_transform(data.drop(columns=feature),
                                col_labels=data.columns)
            exp_var_loss.append(exp_var - model.results['variance_ratio'][0])

        if min(exp_var_loss) > max_exp_var_loss:
            break

        least_informative_feature = top_features[exp_var_loss.index(
            min(exp_var_loss))]
        print(
            f"Removing {least_informative_feature} will result in the smallest reduction in the variance explained by PC1: {min(exp_var_loss)}"
        )
        data = data.drop(columns=least_informative_feature
                         )  # Remove the column from the data
        top_features.remove(least_informative_feature
                            )  # Remove the feature from the top_features list
        exp_var = exp_var - min(exp_var_loss)

    print(f"Number of top features: {len(top_features)}")
    print(f"Top features: {top_features}")
    print(
        f"Variance explained by PC1 when we keep these top features: {exp_var}"
    )
    return top_features


def get_mean_of_neuromaps_under(dir: str,
                                save_to_csv: str = True) -> np.ndarray:
    """
    Get the mean of all the neuromaps under the given directory.

    Args:
        dir: The directory containing the neuromaps.

    Returns:
        The mean of all the neuromaps under the given directory.
    """

    all_maps = []
    for file in sorted(os.listdir(dir)):
        if file.endswith(".csv"):
            all_maps.append(
                pd.read_csv(os.path.join(dir, file),
                            header=None).values.flatten())
    # convert nan in any map to 0
    all_maps = np.stack(all_maps, axis=1)
    np.nan_to_num(all_maps, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    mean_map = np.mean(all_maps, axis=1)

    if save_to_csv:
        mean_map_path = os.path.join(dir, 'mean_map_data.csv')
        all_maps_path = os.path.join(dir, 'all_maps_data.csv')
        pd.DataFrame(mean_map).to_csv(mean_map_path, header=False, index=False)
        pd.DataFrame(all_maps).to_csv(all_maps_path, header=False, index=False)

    return mean_map


##################################################
# Use PCs and mean of neuromaps to parameterize wEE and wEI
##################################################


def get_concat_matrix(dir: str = DESIKAN_NEUROMAPS_DIR,
                      PCs: int | list[int] = 3,
                      use_mean_map=False,
                      use_unit_vector=True):
    """
    Get the concatenated matrix formed by PCs and mean of neuromaps.
    We will parameterize wEE and wEI using the mean of neuromaps and the first few PCs along with a bias term.

    Return:
        The concatenated matrix of shape (N, p), where N is the num of ROIs and p is the num of PCs + 2
    """
    one_array = np.ones(68)
    if use_mean_map:
        mean_map = pd.read_csv(os.path.join(dir, 'mean_map.csv'),
                               header=None).values.flatten()
        concat_matrix = [one_array, mean_map]
    else:
        concat_matrix = [one_array]

    if isinstance(PCs, int):
        PCs = list(range(1, PCs + 1))

    for i in PCs:
        PC = pd.read_csv(os.path.join(dir, f'pc{i}.csv'),
                         header=None).values.flatten()
        if use_unit_vector:
            PC = PC / np.linalg.norm(PC)
        concat_matrix.append(PC)
    return torch.as_tensor(np.stack(
        concat_matrix, axis=1)).to(DEFAULT_DTYPE).to(device=get_device())


def reconstruct(num_of_PCs,
                target_stat,
                target_name,
                visualize_recon=False,
                use_mean_map=False):
    X = get_concat_matrix(PCs=num_of_PCs)
    N = X.shape[0]

    print(f'Reconstructing {target_name}:')
    coef, sum_of_se, _, _ = np.linalg.lstsq(X, target_stat, rcond=None)
    mse = sum_of_se / N
    mse_scalar = mse.item()  # Convert mse to a scalar value
    print(f'MSE: {mse_scalar}')

    # Generate a plot with regression line,
    # as well as the correlation between the target_stat and the projections onto the regression line
    _, ax = plt.subplots()
    projections = X @ coef
    correlation = np.corrcoef(projections.flatten(), target_stat.flatten())[0,
                                                                            1]

    ax.scatter(projections, target_stat)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
    ax.set_xlabel('projections onto the regression line')
    ax.set_ylabel(target_name)
    ax.set_title(
        f'{target_name} vs. projections (used {num_of_PCs} PCs, use_mean_map={use_mean_map})\nMSE={mse_scalar:.4f}, r={correlation:.4f}'
    )
    plt.show()

    recon_res = {
        'coef': coef,
        'mse': mse_scalar,
        'projections': projections,
        'target_stat': target_stat,
        'correlation': correlation,
        'num_of_PCs': num_of_PCs
    }

    if visualize_recon:
        visualize_reconstruction(recon_res, target_name)

    return recon_res


def visualize_reconstruction(recon_res, target_name, use_mean_map=False):
    fig_dir = os.path.join(FIGURES_DIR, 'neuromaps')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    visualize_stats(recon_res['target_stat'], target_name,
                    os.path.join(fig_dir, f"{target_name}_surf_map.png"))
    suffix = f"_using_{recon_res['num_of_PCs']}_PCs"
    if not use_mean_map:
        suffix += "_without_mean_map"

    visualize_stats(
        recon_res['projections'], 'projections',
        os.path.join(fig_dir,
                     f"{target_name}_projections_surf_map{suffix}.png"))

    residuals = convert_to_numpy(recon_res['target_stat']) - convert_to_numpy(
        recon_res['projections'])

    visualize_stats(
        residuals, 'residuals',
        os.path.join(
            fig_dir,
            f"{target_name}_reconstruction_residuals_surf_map{suffix}.png"))

    # Then merge the 3 png images into one png image by laying out this way: target_stat, projection, residual
    concat_n_images([
        os.path.join(fig_dir, f"{target_name}_surf_map.png"),
        os.path.join(fig_dir,
                     f"{target_name}_projections_surf_map{suffix}.png"),
        os.path.join(
            fig_dir,
            f"{target_name}_reconstruction_residuals_surf_map{suffix}.png")
    ], 1, 3, os.path.join(fig_dir,
                          f"{target_name}_reconstruction{suffix}.png"))
