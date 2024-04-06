import os
import pandas as pd
import numpy as np
from pca import pca

from src.basic.constants import NEUROMAPS_SRC_DIR, NUM_DESIKAN_ROI, NUM_ROI, IGNORED_DESIKAN_ROI_ZERO_BASED


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
                       mapping_labels: np.ndarray) -> np.ndarray:
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
        sum_in_ROI[label] += data[i]
        count_in_ROI[label] += 1

    converted_data = sum_in_ROI[1:] / count_in_ROI[1:]
    converted_data = np.delete(converted_data, IGNORED_DESIKAN_ROI_ZERO_BASED)
    np.nan_to_num(converted_data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    assert converted_data.shape[0] == NUM_ROI
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
