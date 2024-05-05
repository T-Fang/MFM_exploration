import os
import pandas as pd
import numpy as np
from pca import pca
from matplotlib import pyplot as plt
import torch

from src.basic.constants import NEUROMAPS_DATA_DIR, NEUROMAPS_SRC_DIR, NUM_DESIKAN_ROI, NUM_ROI, IGNORED_DESIKAN_ROI_ZERO_BASED, \
    DESIKAN_NEUROMAPS_DIR, FIGURES_DIR, DEFAULT_DTYPE
from src.utils.init_utils import get_device
from src.utils.analysis_utils import plot_scatter, visualize_stats, concat_n_images
from src.utils.file_utils import convert_to_numpy, get_HCPYA_group_myelin, get_HCPYA_group_rsfc_gradient, get_cortex_fs5_label


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
                      use_mean_map=True,
                      use_standardizing=True):
    """
    Get the concatenated matrix formed by PCs and mean of neuromaps.
    We will parameterize wEE and wEI using the mean of neuromaps and the first few PCs along with a bias term.

    Return:
        The concatenated matrix of shape (N, p), where N is the num of ROIs and p is the num of PCs + 2
    """
    one_array = np.ones(68)
    concat_matrix = [one_array]
    if use_mean_map:
        mean_map = pd.read_csv(os.path.join(dir, 'mean_map.csv'),
                               header=None).values.flatten()
        if use_standardizing:
            mean_map = (mean_map - np.mean(mean_map)) / np.std(mean_map)
        concat_matrix.append(mean_map)

    if isinstance(PCs, int):
        PCs = list(range(1, PCs + 1))

    for i in PCs:
        PC = pd.read_csv(os.path.join(dir, f'pc{i}.csv'),
                         header=None).values.flatten()
        if use_standardizing:
            PC = (PC - np.mean(PC)) / np.std(PC)
        concat_matrix.append(PC)
    return torch.as_tensor(np.stack(
        concat_matrix, axis=1)).to(DEFAULT_DTYPE).to(device=get_device())


def reconstruct(num_of_PCs,
                target_stat,
                target_name,
                visualize_recon=False,
                use_mean_map=True):
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
        vis_recon(recon_res, target_name, use_mean_map)

    return recon_res


def vis_recon(recon_res, target_name, use_mean_map=True):
    fig_dir = os.path.join(FIGURES_DIR, 'neuromaps', 'reconstruction')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_target_name = target_name.replace('_', ' ')
    ground_truth_path = os.path.join(fig_dir, f"{target_name}_surf_map.png")
    visualize_stats(recon_res['target_stat'], target_name, ground_truth_path,
                    f'Ground truth {fig_target_name}')
    suffix = f"_using_{recon_res['num_of_PCs']}_PCs"
    if not use_mean_map:
        suffix += "_without_mean_map"

    projection_path = os.path.join(
        fig_dir, f"{target_name}_projections_surf_map{suffix}.png")
    visualize_stats(recon_res['projections'], 'projections', projection_path,
                    f'Projections of {fig_target_name}')

    residuals = convert_to_numpy(recon_res['target_stat']) - convert_to_numpy(
        recon_res['projections'])
    residual_path = os.path.join(
        fig_dir,
        f"{target_name}_reconstruction_residuals_surf_map{suffix}.png")
    visualize_stats(residuals, 'residuals', residual_path,
                    f'Residuals for {fig_target_name} reconstruction')

    merged_img_path = os.path.join(
        fig_dir, f"{target_name}_reconstruction{suffix}.png")
    # Then merge the 3 png images into one png image by laying out this way: target_stat, projection, residual
    concat_n_images([ground_truth_path, projection_path, residual_path], 1, 3,
                    merged_img_path)


def vis_myelin_diff():
    HCP_myelin = get_HCPYA_group_myelin(0, 1029).flatten().numpy()
    pkg_myelin_path = os.path.join(DESIKAN_NEUROMAPS_DIR,
                                   'hcps1200_myelinmap.csv')
    # load myelin in the neuromaps pkgs
    pkg_myelin = pd.read_csv(pkg_myelin_path, header=None).values.flatten()
    save_fig_dir = os.path.join(FIGURES_DIR, 'neuromaps', 'compare_myelin')
    os.makedirs(save_fig_dir, exist_ok=True)

    pkg_myelin_fig_path = pkg_myelin_path.replace('.csv', '.png')
    HCP_myelin_fig_path = os.path.join(save_fig_dir, "myelin_HCPYA.png")
    diff_path = os.path.join(save_fig_dir, "myelin_diff.png")

    scatter_plot_path = os.path.join(save_fig_dir, "myelin_scatter.png")
    merged_img_path = os.path.join(save_fig_dir, "myelin_compare_merged.png")

    res_diff = pkg_myelin - HCP_myelin
    stats_list = [pkg_myelin, HCP_myelin, res_diff]
    stats_names = ['myelin_pkg', 'myelin_HCPYA', 'myelin_diff']
    fig_file_paths = [pkg_myelin_fig_path, HCP_myelin_fig_path, diff_path]
    fig_titles = [
        'Myelin in neuromaps pkg', 'Myelin in HCPYA', 'Myelin difference'
    ]
    visualize_stats(stats_list, stats_names, fig_file_paths, fig_titles)

    # additionally, scatter plot pkg_myelin and HCP_myelin as well as the regression line
    plot_scatter(pkg_myelin, HCP_myelin, 'Myelin in neuromaps pkg',
                 'Myelin in HCPYA', 'Myelin comparison', scatter_plot_path,
                 False, True, True)

    concat_n_images([
        pkg_myelin_fig_path, HCP_myelin_fig_path, diff_path, scatter_plot_path
    ], 2, 2, merged_img_path)


def vis_rsfc_gradient_diff():
    HCP_rsfc_gradient = get_HCPYA_group_rsfc_gradient(0,
                                                      1029).flatten().numpy()
    pkg_rsfc_gradient_path = os.path.join(DESIKAN_NEUROMAPS_DIR,
                                          'margulies2016_fcgradient01.csv')
    # load rsfc gradient in the neuromaps pkgs
    pkg_rsfc_gradient = pd.read_csv(pkg_rsfc_gradient_path,
                                    header=None).values.flatten()
    save_fig_dir = os.path.join(FIGURES_DIR, 'neuromaps',
                                'compare_rsfc_gradient')
    os.makedirs(save_fig_dir, exist_ok=True)

    pkg_rsfc_gradient_fig_path = pkg_rsfc_gradient_path.replace('.csv', '.png')
    HCP_rsfc_gradient_fig_path = os.path.join(save_fig_dir,
                                              "rsfc_gradient_HCPYA.png")
    diff_path = os.path.join(save_fig_dir, "rsfc_gradient_diff.png")

    scatter_plot_path = os.path.join(save_fig_dir, "rsfc_gradient_scatter.png")
    merged_img_path = os.path.join(save_fig_dir,
                                   "rsfc_gradient_compare_merged.png")

    res_diff = pkg_rsfc_gradient - HCP_rsfc_gradient

    stats_list = [pkg_rsfc_gradient, HCP_rsfc_gradient, res_diff]
    stats_names = [
        'rsfc_gradient_pkg', 'rsfc_gradient_HCPYA', 'rsfc_gradient_diff'
    ]
    fig_file_paths = [
        pkg_rsfc_gradient_fig_path, HCP_rsfc_gradient_fig_path, diff_path
    ]
    fig_titles = [
        'rsfc gradient in neuromaps pkg', 'rsfc gradient in HCPYA',
        'rsfc gradient difference'
    ]
    visualize_stats(stats_list, stats_names, fig_file_paths, fig_titles)

    # additionally, scatter plot pkg_rsfc_gradient and HCP_rsfc_gradient as well as the regression line
    plot_scatter(pkg_rsfc_gradient, HCP_rsfc_gradient,
                 'rsfc gradient in neuromaps pkg', 'rsfc gradient in HCPYA',
                 'rsfc gradient comparison', scatter_plot_path, False, True,
                 True)

    concat_n_images([
        pkg_rsfc_gradient_fig_path, HCP_rsfc_gradient_fig_path, diff_path,
        scatter_plot_path
    ], 2, 2, merged_img_path)


def vis_PC_myelin_gradient_corr(num_of_PCs=3):
    train_myelin = get_HCPYA_group_myelin(0, 343).flatten().numpy()
    train_rsfc_gradient = get_HCPYA_group_rsfc_gradient(0,
                                                        343).flatten().numpy()
    save_fig_dir = os.path.join(FIGURES_DIR, 'neuromaps',
                                'corr_PC_with_myelin_gradient')
    os.makedirs(save_fig_dir, exist_ok=True)
    train_myelin_fig_path = os.path.join(save_fig_dir, "myelin_train.png")
    train_rsfc_gradient_fig_path = os.path.join(save_fig_dir,
                                                "rsfc_gradient_train.png")
    stats_list = [train_myelin, train_rsfc_gradient]
    stats_names = ['myelin_train', 'rsfc_gradient_train']
    fig_file_paths = [train_myelin_fig_path, train_rsfc_gradient_fig_path]
    fig_titles = ['myelin in train', 'rsfc gradient in train']
    pc_fig_paths = []
    myelin_pc_scatter_paths = []
    rsfc_gradient_pc_scatter_paths = []
    for i in range(num_of_PCs):
        pc_path = os.path.join(DESIKAN_NEUROMAPS_DIR, f'pc{i+1}.csv')
        pc = pd.read_csv(pc_path, header=None).values.flatten()
        stats_list.append(pc)
        stats_names.append(f'pc{i+1}')
        pc_fig_path = pc_path.replace('.csv', '_surf_map.png')
        fig_file_paths.append(pc_fig_path)
        pc_fig_paths.append(pc_fig_path)
        fig_titles.append(f'PC{i+1} of neuromaps')
        # Additionally, scatter plot myelin and rsfc gradient with PC
        myelin_pc_scatter_paths.append(
            os.path.join(save_fig_dir, f"myelin_PC{i+1}_scatter.png"))
        plot_scatter(pc, train_myelin, f'PC{i+1}', 'myelin in train',
                     f'myelin in train vs. PC{i+1}',
                     myelin_pc_scatter_paths[-1], False, True, True)
        rsfc_gradient_pc_scatter_paths.append(
            os.path.join(save_fig_dir, f"rsfc_gradient_PC{i+1}_scatter.png"))
        plot_scatter(pc, train_rsfc_gradient, f'PC{i+1}',
                     'rsfc gradient in train',
                     f'rsfc gradient in train vs. PC{i+1}',
                     rsfc_gradient_pc_scatter_paths[-1], False, True, True)

    visualize_stats(stats_list, stats_names, fig_file_paths, fig_titles)

    image_path_list = [None] + pc_fig_paths + [train_myelin_fig_path] + myelin_pc_scatter_paths + \
                      [train_rsfc_gradient_fig_path] + rsfc_gradient_pc_scatter_paths

    merged_img_path = os.path.join(save_fig_dir,
                                   "corr_PC_with_myelin_gradient.png")

    concat_n_images(image_path_list, 3, num_of_PCs + 1, merged_img_path)


def vis_transform_diff(author,
                       map_name,
                       save_fig_dir='',
                       space='fsaverage',
                       res='10k'):
    space_and_res = f"{space}_{res}"
    neuromaps_pkg_res_dir = os.path.join(NEUROMAPS_DATA_DIR, space_and_res)
    CBIG_res_dir = os.path.join(NEUROMAPS_DATA_DIR, f'CBIG_{space_and_res}')

    file_name = f"{author}_{map_name}"
    # fig_title = f'{author} {map_name}'
    fig_title = f'{map_name}'
    neuromaps_pkg_res_path = os.path.join(neuromaps_pkg_res_dir,
                                          f"{file_name}.csv")
    CBIG_res_path = os.path.join(CBIG_res_dir, f"{file_name}.csv")

    neuromaps_pkg_res = pd.read_csv(neuromaps_pkg_res_path,
                                    header=None).values.flatten()
    CBIG_res = pd.read_csv(CBIG_res_path, header=None).values.flatten()

    if save_fig_dir == '':
        save_fig_dir = os.path.join(FIGURES_DIR, 'neuromaps',
                                    'compare_transform')

    os.makedirs(save_fig_dir, exist_ok=True)

    neuromaps_pkg_res_fig_path = neuromaps_pkg_res_path.replace('.csv', '.png')
    CBIG_res_fig_path = CBIG_res_path.replace('.csv', '.png')
    diff_path = os.path.join(save_fig_dir, f"{file_name}_diff.png")
    scatter_plot_path = os.path.join(save_fig_dir, f"{file_name}_scatter.png")
    merged_img_path = os.path.join(save_fig_dir,
                                   f"{file_name}_compare_transform.png")

    cortex_fs5_label = get_cortex_fs5_label()
    res_diff = neuromaps_pkg_res - CBIG_res
    res_diff[cortex_fs5_label == 1] = 0

    # visualize the neuromaps pkg's transform outcome and CBIG's transform outcome
    stats_list = [neuromaps_pkg_res, CBIG_res, res_diff]
    stats_names = [
        f'{file_name}_neuromaps_pkg', f'{file_name}_CBIG', f'{file_name}_diff'
    ]
    fig_file_paths = [neuromaps_pkg_res_fig_path, CBIG_res_fig_path, diff_path]
    fig_titles = [
        f'{fig_title} (neuromaps pkg)', f'{fig_title} (CBIG)',
        f'{fig_title} difference'
    ]
    visualize_stats(stats_list,
                    stats_names,
                    fig_file_paths,
                    fig_titles,
                    space='fs5')
    neuromaps_pkg_res_cortex = neuromaps_pkg_res[cortex_fs5_label == 2]
    CBIG_res_cortex = CBIG_res[cortex_fs5_label == 2]

    # additionally, scatter plot neuromaps_pkg_res and CBIG_res as well as the regression line
    plot_scatter(neuromaps_pkg_res_cortex, CBIG_res_cortex,
                 f'{fig_title} (neuromaps pkg)', f'{fig_title} (CBIG)',
                 f'{fig_title} compare', scatter_plot_path, False, True, True)

    concat_n_images([
        neuromaps_pkg_res_fig_path, CBIG_res_fig_path, diff_path,
        scatter_plot_path
    ], 2, 2, merged_img_path)
