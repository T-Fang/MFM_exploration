import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import cv2
from matplotlib import pyplot as plt
from scipy import stats, io as sio
from itertools import combinations

from src.basic.constants import HCPYA_1029_DATA_DIR, MATLAB_SCRIPT_DIR, NUM_ROI, SUBJECT_ID_RANGE
from src.utils.file_utils import convert_to_mat, get_HCPYA_group_emp_TC, get_HCPYA_group_myelin, get_HCPYA_group_rsfc_gradient, \
    get_best_params_sim_res_path, get_emp_fig_dir, get_fig_dir_in_logs, get_fig_path_in_logs, \
    get_losses_fig_dir, get_run_dir, get_sim_res_dir, get_target_dir, get_vis_param_dir, load_all_val_dicts, \
    load_best_params, load_train_dict, get_sim_res_path, get_values_at_indices
from src.utils.val_utils import get_agg_seeds_range
from statannotations.Annotator import Annotator

############################################################
# Visualization related
############################################################


# * General Visualization Functions
def concat_n_images(image_path_list, rows, cols, save_file_path=None):
    """
    Combines N color images from a list of image paths into a grid of specified rows and columns.
    If the dimensions of the images are not consistent, the images will be rescaled according to the dimensions of the first image.

    Args:
        image_path_list (List[str]): List of image paths.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        save_file_path (str, optional): File path to save the concatenated image. Defaults to None.

    Returns:
        ndarray: The concatenated image as a numpy array.
    """
    if len(image_path_list) != rows * cols:
        raise ValueError("Number of images does not match the grid dimensions")

    # get the first image that is not None
    first_image_path = next(
        (image_path for image_path in image_path_list if image_path), None)
    if first_image_path is None:
        raise ValueError("No valid image paths found")

    first_image = plt.imread(first_image_path)[:, :, :3]
    first_image_height, first_image_width, _ = first_image.shape

    output = []
    for r in range(rows):
        row_images = []
        for c in range(cols):
            img_path = image_path_list[r * cols + c]
            # if img_path is None, fill with white image
            if img_path is None:
                img = np.ones((first_image_height, first_image_width, 3))
            else:
                img = plt.imread(img_path)[:, :, :3]
                img_height, img_width, _ = img.shape
                if img_height != first_image_height or img_width != first_image_width:
                    img = resize_image(img, first_image_height,
                                       first_image_width)
            row_images.append(img)
        output.append(np.hstack(row_images))
    output = np.vstack(output)

    if save_file_path is not None:
        plt.imsave(save_file_path, output)
    return output


def resize_image(image, target_height, target_width):
    """
    Resizes an image to the target height and width.

    Args:
        image (ndarray): The image as a numpy array.
        target_height (int): The target height.
        target_width (int): The target width.

    Returns:
        ndarray: The resized image as a numpy array.
    """
    resized_image = cv2.resize(image, (target_width, target_height))
    return resized_image


def visualize_stats(stats_list,
                    stats_names,
                    fig_file_paths,
                    fig_titles,
                    space='DK68'):
    """
    Visualize the stats using MATLAB script.
    Firstly, load the [68, 1] statistic (one scalar for each ROI).
    Then, run the vis_DK68_param function to visualize the stats.

    Args:
        stats_list (Union[ndarray, List[ndarray]]): The list of statistics data.
        stats_names (Union[str, List[str]]): The list of names of the statistics.
        fig_file_paths (Union[str, List[str]]): The list of file paths to save the visualizations.
        fig_titles (Union[str, List[str]]): The list of titles of the visualizations.
        space (str, optional): The space of the statistics. Can be one of ['DK68', 'fs5'] Defaults to 'DK68'.

    Raises:
        Exception: If MATLAB encounters an error or terminates.
    """
    stats_list = [stats_list
                  ] if not isinstance(stats_list, list) else stats_list
    stats_names = [stats_names
                   ] if not isinstance(stats_names, list) else stats_names
    fig_file_paths = [
        fig_file_paths
    ] if not isinstance(fig_file_paths, list) else fig_file_paths
    fig_titles = [fig_titles
                  ] if not isinstance(fig_titles, list) else fig_titles

    commands = []
    for stats_data, stats_name, fig_file_path, fig_title in zip(
            stats_list, stats_names, fig_file_paths, fig_titles):
        commands.append(
            f"load('{convert_to_mat(stats_data, stats_name)}', '{stats_name}')"
        )
        commands.append(
            f"vis_{space}_param({stats_name}, '{fig_file_path}', '{fig_title}')"
        )

    run_matlab_commands(commands)
    print("Visualizations saved to:", fig_file_paths)


def run_matlab_commands(commands):
    """
    Run a MATLAB command in the MATLAB
    ! Remember to use double quote for the command string,
    ! and single quote to indicate strings/char vectors in MATLAB.
    ! Additionally, there is no need to add ; at the end of each MATLAB command.

    Args:
        command (str or List[str]): The MATLAB command(s) to run.

    Raises:
        Exception: If MATLAB encounters an error or terminates.
    """
    if isinstance(commands, str):
        commands = [commands]
    commands = f"matlab -nodisplay -nosplash -nodesktop -r \"try, addpath(genpath('{MATLAB_SCRIPT_DIR}')); {'; '.join(commands)}; catch ME, disp(ME.message), exit(1), end, exit;\""  # noqa: E501

    print("Running MATLAB command:", commands)
    result = subprocess.run(commands,
                            shell=True,
                            capture_output=True,
                            text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise Exception("Matlab encountered an error or terminated.")
    else:
        print(result.stdout)


def load_sim_res(ds_name: str, target: str, phase: int, trial_idx: int,
                 seed_idx: int | str):
    sim_res_dir = get_sim_res_dir(ds_name, target, phase, trial_idx, seed_idx)
    sim_res_path = get_best_params_sim_res_path(phase, sim_res_dir)

    sim_res = torch.load(sim_res_path, map_location='cpu')
    return sim_res


# * Plot FC and FCD
def plot_sim_fc_fcd(
    ds_name: str,
    target: str,
    phase: int,
    trial_idx: int,
    seed_idx: int | str,
    param_idx: int,
):
    """
    Plot simulated FC and FCD from the given simulation result.

    Args:
        sim_res_path (str): Path to the simulation result.
        param_idx (int): Index of the parameter to plot.
    """
    sim_res = load_sim_res(ds_name, target, phase, trial_idx, seed_idx)
    fc_sim = sim_res['fc_sim'][param_idx]
    fcd_sim = sim_res['fcd_sim'][param_idx]

    fcd_sim_save_path = get_sim_res_path(ds_name, target, phase, trial_idx,
                                         seed_idx, 'sim_fcd.csv')
    fc_sim_save_path = get_sim_res_path(ds_name, target, phase, trial_idx,
                                        seed_idx, 'sim_fc.csv')

    # save to csv
    pd.DataFrame(fc_sim).to_csv(fc_sim_save_path, index=False, header=False)
    pd.DataFrame(fcd_sim).to_csv(fcd_sim_save_path, index=False, header=False)

    # draw heatmaps
    draw_heatmap(
        [fc_sim_save_path, fcd_sim_save_path],
        [f'simulated FC on {phase} set', f'simulated FCD on {phase} set'], [
            fc_sim_save_path.replace('.csv', '.png'),
            fcd_sim_save_path.replace('.csv', '.png')
        ])


def compare_fc(sim_fc_path, emp_fc_path):
    """
    Compare the simulated FC with the empirical FC.
    """
    agreement_fig_path = sim_fc_path.replace('.csv', '_agreement_with_emp.png')
    command = f"compare_FC('{sim_fc_path}', '{emp_fc_path}', '{agreement_fig_path}')"
    run_matlab_commands(command)

    compare_fig_path = sim_fc_path.replace('.csv', '_compare_with_emp.png')
    sim_fc_fig_path = sim_fc_path.replace('.csv', '.png')
    emp_fc_fig_path = emp_fc_path.replace('.csv', '.png')
    concat_n_images([sim_fc_fig_path, emp_fc_fig_path, agreement_fig_path], 1,
                    3, compare_fig_path)


def compare_fcd(sim_fcd_path, emp_fcd_path):
    """
    Compare the simulated FCD with the empirical FCD.
    """
    sim_fcd_fig_path = sim_fcd_path.replace('.csv', '.png')
    emp_fcd_fig_path = emp_fcd_path.replace('.csv', '.png')
    concat_n_images([sim_fcd_fig_path, emp_fcd_fig_path], 1, 2,
                    sim_fcd_path.replace('.csv', '_compare_with_emp.png'))


def compare_fc_fcd(ds_name, target, phase, trial_idx, seed_idx, param_idx):
    plot_sim_fc_fcd(ds_name, target, phase, trial_idx, seed_idx, param_idx)
    sim_fc_path = get_sim_res_path(ds_name, target, phase, trial_idx, seed_idx,
                                   'sim_fc.csv')
    sim_fcd_path = get_sim_res_path(ds_name, target, phase, trial_idx,
                                    seed_idx, 'sim_fcd.csv')
    emp_fc_fig_dir = get_emp_fig_dir(ds_name, target, 'FC')
    emp_fcd_fig_dir = get_emp_fig_dir(ds_name, target, 'FCD')
    range_start, range_end = SUBJECT_ID_RANGE[ds_name][phase]
    emp_fc_path = os.path.join(emp_fc_fig_dir,
                               f'group_emp_fc_{range_start}_{range_end}.csv')
    emp_fcd_path = os.path.join(
        emp_fcd_fig_dir, f'group_emp_fcd_{range_start}_{range_end}.csv')

    compare_fc(sim_fc_path, emp_fc_path)
    compare_fcd(sim_fcd_path, emp_fcd_path)


def draw_heatmap(heatmap_data_paths, titles, save_file_paths):
    """
    Draw heatmaps from the heatmap data files.

    Args:
        heatmap_data_paths (Union[str, List[str]]): The path(s) to the heatmap data csv file(s).
        titles (Union[str, List[str]]): The title(s) of the heatmap(s).
        save_file_paths (Union[str, List[str]]): The path(s) to save the heatmap(s).
    """

    if not isinstance(heatmap_data_paths, list):
        heatmap_data_paths = [heatmap_data_paths]

    if not isinstance(titles, list):
        titles = [titles]

    if not isinstance(save_file_paths, list):
        save_file_paths = [save_file_paths]

    commands = []
    for heatmap_data_path, title, save_file_path in zip(
            heatmap_data_paths, titles, save_file_paths):
        command = f"draw_heatmap('{heatmap_data_path}', '{title}', '{save_file_path}')"
        commands.append(command)

    run_matlab_commands(commands)
    print("Heatmaps saved to:", save_file_paths)


def plot_bold_TC(bold_TC: torch.Tensor | np.ndarray,
                 fig_file_path: str,
                 standardize_bold: bool = True,
                 remove_first_frames: int = 50):
    """
    Plot the time course of the given BOLD signal of shape (68, 1200), which has the shape (num_ROI, num_time_points).
    """
    if standardize_bold:
        bold_TC = (bold_TC - bold_TC.mean(
            axis=1, keepdims=True)) / bold_TC.std(axis=1, keepdims=True)

    original_font_size = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 5))
    for i, bold_TC_ROI in enumerate(bold_TC):
        # color = plt.cm.viridis(i / 68)  # Use a colormap for colors
        # plt.plot(bold_TC[i], label=f"Time Series {i+1}", color=color)
        # plt.plot(bold_TC_ROI, label=f"Time Series {i+1}", linewidth=0.6)
        plt.plot(bold_TC_ROI,
                 label=f"Time Series {i+1}",
                 linewidth=0.2,
                 color='black')
    plt.xlabel('Time points')
    plt.ylabel('BOLD signal')
    plt.title('BOLD time course')
    plt.ylim(-5, 5)  # Limit the x-axis range from 0 to 1200
    plt.xlim(remove_first_frames,
             bold_TC.shape[1])  # Limit the x-axis range from 0 to 1200
    plt.tight_layout()
    plt.savefig(fig_file_path)
    plt.close()

    # Restore the original font size
    plt.rcParams.update({'font.size': original_font_size})

    print("BOLD time course saved to:", fig_file_path)
    return fig_file_path


def plot_sim_bold_TC(ds_name: str,
                     target: str,
                     phase: int,
                     trial_idx: int,
                     seed_idx: int | str,
                     param_idx: int = 0,
                     standardize_bold: bool = True,
                     sim_fig_path: str = None):
    """
    Plot the time course of the simulated BOLD signal, which has the shape (num_ROI, num_time_points).
    """
    sim_res = load_sim_res(ds_name, target, phase, trial_idx, seed_idx)
    bold_TC = sim_res['bold_TC_sim'][:, param_idx]

    if sim_fig_path is None:
        sim_fig_path = get_sim_res_path(ds_name, target, phase, trial_idx,
                                        seed_idx, 'bold_TC_sim.png')

    plot_bold_TC(bold_TC, sim_fig_path, standardize_bold)

    subject_id_range = SUBJECT_ID_RANGE[ds_name][phase]
    emp_fig_path = plot_emp_bold_TC(subject_id_range[0], subject_id_range[1],
                                    standardize_bold)
    concat_n_images([sim_fig_path, emp_fig_path], 2, 1, sim_fig_path)


def plot_emp_bold_TC(range_start: int,
                     range_end: int,
                     standardize_bold: bool = True,
                     fig_file_path: str = None):
    """
    Plot the time course of the simulated BOLD signal, which has the shape (num_ROI, num_time_points).
    """

    group_TC_save_path = os.path.join(
        HCPYA_1029_DATA_DIR, 'TC', 'group_TC',
        f'group_emp_TC_{range_start}_{range_end}.csv')

    # if group_TC_save_path does not exist
    if not os.path.exists(group_TC_save_path):
        bold_TC = get_HCPYA_group_emp_TC(range_start=range_start,
                                         range_end=range_end)
    else:
        bold_TC = pd.read_csv(group_TC_save_path, header=None).values

    if fig_file_path is None:
        fig_file_path = group_TC_save_path.replace('.csv', '.png')

    return plot_bold_TC(bold_TC, fig_file_path, standardize_bold)


def corr_params_with_myelin_gradient(ds_name: str,
                                     target: str,
                                     phase: int,
                                     trial_idx: int,
                                     seed_idx: int | str,
                                     top_k_params: int = 10):
    """
    Correlate each of the best 10 params with the myelin and rsfc gradient, respectively.
    We will get a resulting DataFrame with index 'myelin' and 'rsfc_gradient',
    and columns the index of correlated param in the top 10 best params.

    Finally, plot the DataFrame as a heatmap, with index and columns
    """
    best_params_dict = load_best_params(ds_name, target, phase, trial_idx,
                                        seed_idx)
    best_params = best_params_dict['parameter'][:, :top_k_params]

    split_params = {
        'wEE': best_params[:NUM_ROI],
        'wEI': best_params[NUM_ROI:2 * NUM_ROI],
        'sigma': best_params[2 * NUM_ROI + 1:]
    }

    myelin = get_HCPYA_group_myelin(0, 343).flatten().numpy()
    rsfc_gradient = get_HCPYA_group_rsfc_gradient(0, 343).flatten().numpy()

    for param_name, param in split_params.items():
        corr_with_myelin = [
            stats.pearsonr(myelin, param[:, i])[0] for i in range(top_k_params)
        ]
        corr_with_rsfc_gradient = [
            stats.pearsonr(rsfc_gradient, param[:, i])[0]
            for i in range(top_k_params)
        ]

        # print('myelin:', myelin)
        # print('param:', param)
        # print('rsfc_gradient:', rsfc_gradient)
        # print('corr_with_myelin:', corr_with_myelin)
        # print('corr_with_rsfc_gradient:', corr_with_rsfc_gradient)

        corr_df = pd.DataFrame({
            'myelin': corr_with_myelin,
            'rsfc_gradient': corr_with_rsfc_gradient
        })

        fig_file_path = get_fig_path_in_logs(
            ds_name, target, 'corr_params_with_myelin_gradient', trial_idx,
            seed_idx, f'corr_params_with_myelin_gradient_{param_name}.png')
        plt.figure()
        sns.heatmap(
            corr_df.T, annot=True
        )  # ! To have annotations, either downgrade matplotlib to 3.7 or upgrade seaborn to 0.13
        plt.xlabel("best param vector index")
        plt.title(
            f'Correlation with myelin and rsfc gradient for {param_name}')
        plt.savefig(fig_file_path)
        plt.close()
        print("Correlation with myelin and rsfc gradient saved to:",
              fig_file_path)


def plot_corr_matrix_for_best_params(ds_name: str, target: str, phase: int,
                                     trial_idx: int, seed_idx: int | str):
    """
    Plot the correlation matrix for the best parameters.
    """
    best_params_dict = load_best_params(ds_name, target, phase, trial_idx,
                                        seed_idx)
    best_params = best_params_dict['parameter']
    wEEs = best_params[:NUM_ROI]
    wEIs = best_params[NUM_ROI:2 * NUM_ROI]
    sigmas = best_params[2 * NUM_ROI + 1:]

    for param, param_name in zip([wEEs, wEIs, sigmas, best_params],
                                 ['wEE', 'wEI', 'sigma', 'all_params']):
        corr_matrix = np.corrcoef(param.T)
        plt.figure()
        heatmap = sns.heatmap(
            corr_matrix, annot=True
        )  # ! To have annotations, either downgrade matplotlib to 3.7 or upgrade seaborn to 0.13
        heatmap.set_title(f'Correlation matrix for {param_name}')
        # plt.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        # plt.colorbar()
        # plt.title(f'Correlation matrix for {param_name}')
        fig_file_path = get_fig_path_in_logs(
            ds_name, target, f'corr_matrix_for_best_{phase}_params', trial_idx,
            seed_idx, f'corr_matrix_{param_name}.png')
        plt.savefig(fig_file_path)
        plt.close()
        print(f"Correlation matrix for {param_name} saved to:", fig_file_path)


def boxplot_network_stats(mat_file_path, stats_name, fig_file_path):
    """
    First, cd into MATLAB_SCRIPT_PATH.
    Then, load the [68, 1] statistic (one scalar for each ROI) stored in the `mat_file_path` file.
    Finally, run the vis_DK68_param function to
    generates a box plot depicting the network pattern of the input statistic.
    """

    print(f"Boxplotting {stats_name} from {mat_file_path}...")

    command = [(
        f"matlab -nodisplay -nosplash -nodesktop -r "
        f"\"addpath(genpath('{MATLAB_SCRIPT_DIR}')); "
        f"load('{mat_file_path}', '{stats_name}'); "
        f"yeo7_network_boxplot({stats_name}, '{stats_name.replace('_', ' ')}', '{fig_file_path}'); "
        f"exit;\"")]

    result = subprocess.run(command,
                            shell=True,
                            capture_output=True,
                            text=True)
    print(result.stdout)
    print(result.stderr)

    print(f'Network boxplot saved to {fig_file_path}')


def ttest_1samp_n_plot(list_1,
                       list_2,
                       need_boxplot,
                       need_pvalue=True,
                       labels=['list_1', 'list_2'],
                       save_fig_path=None,
                       fig_title='t_test',
                       xlabel='list',
                       ylabel='y'):
    """
    Perform 1 sample t-test on the difference between `list_1` and `list_2`, and plot

    Args:
        list_1 (ListLike): The first list / array
        list_2 (ListLike): The second list / array
        need_boxplot (boolean): whether to boxplot
        labels (list, optional): boxplot group labels. Defaults to ['list_1', 'list_2'].
        save_fig_path (str, optional): save figure path. Defaults to None.
        fig_title (str, optional): figure title, p-value will be added automatically. Defaults to 't_test'.
        xlabel (str, optional): x-axis label. Defaults to 'list'.
        ylabel (str, optional): y-axis label. Defaults to 'y'.

    Raises:
        Exception: Not specify save_fig_path if need_boxplot=True
    """
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    if need_pvalue:
        diff = list_1 - list_2
        statistics, p_value = stats.ttest_1samp(diff, 0)
        print('Average: ', np.mean(list_1), np.mean(list_2))
        print(f"t-test results: statistics: {statistics}; p-value: {p_value}")

    if need_boxplot:
        if save_fig_path is None:
            raise Exception(
                "If you need boxplot, please specify the figure save path.")
        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([list_1, list_2], labels=labels, showfliers=False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if need_pvalue:
            title = f'{fig_title}, p={p_value:.4f}'
        else:
            title = f'{fig_title}'
        plt.title(title)
        plt.savefig(save_fig_path)
        plt.close()
        print("Boxplot figure saved.")
    return 0


############################################################
# Visualize param vector
############################################################
def vis_best_param_vector(ds_name: str,
                          target: str,
                          phase: int,
                          trial_idx: int,
                          seed_idx: int | str,
                          corr_with_myelin_gradient: bool = True):
    vis_param_dir = get_vis_param_dir(ds_name, target, phase, trial_idx,
                                      seed_idx)

    save_dict = load_best_params(ds_name, target, phase, trial_idx, seed_idx)
    param_vector = save_dict['parameter'][:, 0]
    wEE = param_vector[:NUM_ROI]
    wEI = param_vector[NUM_ROI:2 * NUM_ROI]
    sigma = param_vector[2 * NUM_ROI + 1:]
    param_dict = {'wEE': wEE, 'wEI': wEI, 'sigma': sigma}

    os.makedirs(vis_param_dir, exist_ok=True)

    train_myelin = get_HCPYA_group_myelin(0, 343).flatten().numpy()
    train_rsfc_gradient = get_HCPYA_group_rsfc_gradient(0,
                                                        343).flatten().numpy()

    train_myelin_fig_dir = get_emp_fig_dir('HCPYA', 'all_participants',
                                           'myelin')
    train_myelin_fig_path = os.path.join(train_myelin_fig_dir,
                                         "myelin_train.png")
    train_rsfc_gradient_fig_dir = get_emp_fig_dir('HCPYA', 'all_participants',
                                                  'rsfc_gradient')
    train_rsfc_gradient_fig_path = os.path.join(train_rsfc_gradient_fig_dir,
                                                "rsfc_gradient_train.png")

    stats_list = [train_myelin, train_rsfc_gradient]
    stats_names = ['myelin_train', 'rsfc_gradient_train']
    fig_file_paths = [train_myelin_fig_path, train_rsfc_gradient_fig_path]
    fig_titles = ['myelin in train', 'rsfc gradient in train']
    param_fig_paths = []
    myelin_param_scatter_paths = []
    rsfc_gradient_param_scatter_paths = []

    param_names = ['wEE', 'wEI', 'sigma']
    if torch.all(sigma == sigma[0]):
        # ignore sigma if all sigma are the same
        param_names = ['wEE', 'wEI']

    for param_name in param_names:
        param = param_dict[param_name]
        stats_list.append(param)
        stats_names.append(f'{param_name}')
        param_fig_path = os.path.join(vis_param_dir,
                                      f"{param_name}_surf_map.png")
        fig_file_paths.append(param_fig_path)
        param_fig_paths.append(param_fig_path)
        fig_titles.append(f'{param_name}')

        if corr_with_myelin_gradient:
            # scatter plot myelin and rsfc gradient with the param
            myelin_param_scatter_paths.append(
                os.path.join(vis_param_dir,
                             f"myelin_{param_name}_scatter.png"))
            plot_scatter(param, train_myelin, param_name, 'myelin in train',
                         f'myelin in train vs. {param_name}',
                         myelin_param_scatter_paths[-1], False, True, True)
            rsfc_gradient_param_scatter_paths.append(
                os.path.join(vis_param_dir,
                             f"rsfc_gradient_{param_name}_scatter.png"))
            plot_scatter(param, train_rsfc_gradient, param_name,
                         'rsfc gradient in train',
                         f'rsfc gradient in train vs. {param_name}',
                         rsfc_gradient_param_scatter_paths[-1], False, True,
                         True)

    visualize_stats(stats_list, stats_names, fig_file_paths, fig_titles)

    if corr_with_myelin_gradient:
        image_path_list = [None] + param_fig_paths + [train_myelin_fig_path] + myelin_param_scatter_paths + \
                          [train_rsfc_gradient_fig_path] + rsfc_gradient_param_scatter_paths
        merged_img_path = os.path.join(
            vis_param_dir,
            "vis_best_param_vector_corr_with_myelin_gradient.png")
        concat_n_images(image_path_list, 3,
                        len(param_names) + 1, merged_img_path)
    else:
        image_path_list = param_fig_paths
        merged_img_path = os.path.join(vis_param_dir,
                                       "vis_best_param_vector.png")
        concat_n_images(image_path_list, 1, len(param_names), merged_img_path)


############################################################
# Loss related analysis
############################################################
def unravel_index(indices, shape):
    """
    Converts a tensor of flat indices into a tensor of coordinate vectors.
    (Extracted from https://github.com/pytorch/pytorch/issues/35674)

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs,
                     rounding_mode='trunc') % shape[:-1]


def get_val_top_k(val_dicts, k=10):
    """
    For a given validation dictionary, get the indices of the top k children with the lowest total loss

    val_dicts is a dictionary with the same keys as every saved validation dictionary.
    But the value of each key in the dictionary now has an extra two dimensions (seed, epoch)
    compared to the original dictionary value, which should all be torch tensors.

    Specifically, we will look values for keys ['corr_loss', 'l1_loss', 'ks_loss'], and 'r_E_reg_loss' if present.

    """
    corr_loss = val_dicts['corr_loss']
    l1_loss = val_dicts['l1_loss']
    ks_loss = val_dicts['ks_loss']
    if 'r_E_reg_loss' in val_dicts.keys():
        r_E_reg_loss = val_dicts['r_E_reg_loss']
    else:
        r_E_reg_loss = None

    # sum up all losses for each child at each epoch
    total_loss = corr_loss + l1_loss + ks_loss
    if r_E_reg_loss is not None:
        total_loss += r_E_reg_loss

    # total_loss is of shape (num_of_seeds, num_of_epochs, num_of_chosen_val_params)
    # we want to find the indices of the top k children with the lowest total loss across all seeds and epochs
    # so we first flatten the tensor to (num_of_seeds * num_of_epochs, num_of_chosen_val_params)
    # then we find the indices of the top k children with the lowest total loss
    # finally, we convert the indices back to the original shape
    original_shape = total_loss.shape
    total_loss = total_loss.view(-1)
    k_lowest_total_losses, topk_indices = torch.topk(total_loss,
                                                     k,
                                                     largest=False)
    topk_indices = unravel_index(topk_indices, original_shape)

    return k_lowest_total_losses, topk_indices


def get_train_top_k(saved_dict, k=1):
    """
    For each epoch, get the indices of the top k children with the lowest total loss
    """
    FC_FCD_loss = saved_dict['FC_FCD_loss']
    if 'r_E_reg_loss' in saved_dict.keys():
        r_E_reg_loss = saved_dict['r_E_reg_loss']
    else:
        r_E_reg_loss = None

    # sum up all losses for each child at each epoch
    total_loss = torch.sum(FC_FCD_loss, dim=1)
    if r_E_reg_loss is not None:
        total_loss += r_E_reg_loss

    # Find the index for k children with the lowest total loss
    # use topk instead of sort to speed up

    if k is not None:
        k_lowest_total_losses, topk_indices = torch.topk(total_loss,
                                                         k,
                                                         largest=False)
    else:
        k_lowest_total_losses, topk_indices = torch.sort(total_loss)

    return k_lowest_total_losses, topk_indices


def plot_losses_for_diff_trials(
        ds_name,
        target,
        trial_range,
        trial_names,
        seed_range,
        group_idx,
        epoch_range,
        loss_types=['total_loss', 'corr_loss', 'l1_loss', 'ks_loss']):
    """
    This function plot the different losses for a given validation subject group across different trials.

    For each loss from loss_types, draw a box plot, where box represents a trial,
    and every dot in a box represents 1 of the 10 param vectors with the lowest total loss among all seeds under the setup of the trial.

    This function assumes that the target's directory contains
    pth files at f'validation/trial{trial_idx}/seed{seed_idx}/group{group_idx}/best_param{epoch_idx}.pth'.
    Each of the file contains a dictionary, with keys ['corr_loss', 'l1_loss', 'ks_loss'], and possibly 'r_E_reg_loss',
    and values the corresponding losses in torch tensor of size (num_of_chosen_val_params, )

    Args:
        ds_name (str): The dataset name.
        target (str): The target name. (e.g., 'age_group')
        trial_range (range): The range of trial indices.
        trial_names (list): The display names of the trials.
        seed_range (range): The range of seed indices.
        group_idx (int): The index of the validation subject group.
        epoch_range (range): The range of epoch indices.
        loss_types (list, optional): The loss types to plot.

    Returns:
        None (the plots will be stored in the f'PROJECT_PATH/logs/{ds_name}/{target}/figures/losses/')
    """
    for loss_type in loss_types:

        fig_save_dir = get_losses_fig_dir(ds_name, target)
        fig_save_path = os.path.join(fig_save_dir,
                                     f'group{group_idx}_{loss_type}.png')

        print(
            f'Plotting {loss_type} for {target} group {group_idx} across trials {trial_names}...'
        )
        plt.figure()
        all_trials_losses = []
        for trial_idx in trial_range:
            val_dicts = load_all_val_dicts(ds_name, target, trial_idx,
                                           seed_range, group_idx, epoch_range)
            k_lowest_total_loss, topk_indices = get_val_top_k(val_dicts, k=10)
            if loss_type == 'total_loss':
                losses = k_lowest_total_loss
            else:
                losses = np.array(
                    [val_dicts[loss_type][tuple(i)] for i in topk_indices])
            all_trials_losses.append(losses)
        plt.boxplot(all_trials_losses, labels=trial_names)
        plt.xlabel('Setups')
        plt.ylabel(loss_type)
        plt.savefig(fig_save_path)
        plt.close()


@DeprecationWarning
# ! Since it's difficult to pinpoint each group's losses,
# ! we will use the plot_losses_for_diff_trials function instead
def plot_losses_for_diff_trials_all_groups(
        ds_name,
        target,
        trial_range,
        trial_names,
        loss_types=['total_loss', 'corr_loss', 'l1_loss', 'ks_loss']):
    """
    For each loss from loss_types, draw a box plot, where box represents a trial,
    and every dot in a box represents a group's lowest loss among all seeds under the setup of the trial.

    This function assumes that the target's directory contains
    a pth file at f'trial{trial_idx}/seed_best_among_all/lowest_losses.pth'.
    Each of the file contains a dictionary, with keys ['total_loss', 'corr_loss', 'l1_loss', 'ks_loss'],
    and values the corresponding losses in torch tensor of size (num_of_groups, )

    Args:
        ds_name (str): The dataset name.
        target (str): The target name. (e.g., 'age_group')
        trial_range (range): The range of trial indices.
        trial_names (list): The display names of the trials.

    Returns:
        None (the plots will be stored in the f'PROJECT_PATH/logs/{ds_name}/{target}/figures/losses/')
    """
    for loss_type in loss_types:

        fig_save_dir = get_losses_fig_dir(ds_name, target)
        fig_save_path = os.path.join(fig_save_dir, f'{loss_type}.png')

        plt.figure()
        all_trials_losses = []
        for trial_idx in trial_range:
            losses_file_dir = get_run_dir(ds_name, target, 'test', trial_idx,
                                          '_best_among_all')
            losses_dict = torch.load(os.path.join(losses_file_dir,
                                                  'lowest_losses.pth'),
                                     map_location='cpu')
            losses = losses_dict[loss_type]
            all_trials_losses.append(losses.numpy())
        plt.boxplot(all_trials_losses, labels=trial_names)
        plt.xlabel('Setups')
        plt.ylabel(loss_type)
        plt.savefig(fig_save_path)
        plt.close()


def plot_train_loss(ds_name,
                    target,
                    trial_idx,
                    seed_idx,
                    group_idx,
                    epoch_range,
                    save_fig_path=None,
                    lowest_top_k=10,
                    show_individual_loss=None):

    if save_fig_path is None:
        group_prefix = '' if group_idx is None else f'group{group_idx}_'
        postfix = '' if show_individual_loss is None else f'_{show_individual_loss}'
        postfix += f'_for_top{lowest_top_k}_children' if lowest_top_k is not None else ''
        save_fig_path = get_fig_path_in_logs(
            ds_name, target, 'losses', trial_idx, seed_idx,
            f'{group_prefix}train_losses{postfix}.png')

    corr_list = []
    l1_list = []
    ks_list = []
    r_E_list = []
    for epoch_idx in epoch_range:
        d = load_train_dict(ds_name,
                            target,
                            trial_idx,
                            seed_idx,
                            group_idx,
                            epoch_idx,
                            apply_sort=False)

        corr_losses = d['corr_loss'][:lowest_top_k]
        corr_list.append(torch.mean(corr_losses).item())
        l1_losses = d['l1_loss'][:lowest_top_k]
        l1_list.append(torch.mean(l1_losses).item())
        ks_losses = d['ks_loss'][:lowest_top_k]
        ks_list.append(torch.mean(ks_losses).item())
        if 'r_E_reg_loss' in d.keys():
            r_E_reg_losses = d['r_E_reg_loss'][:lowest_top_k]
            r_E_list.append(torch.mean(r_E_reg_losses).item())

    x = np.array(epoch_range)
    plt.figure()
    if show_individual_loss is None:
        print('Plotting all losses...')
        plt.plot(x, corr_list, 'r-', label='Corr loss')
        plt.plot(x, l1_list, 'b-', label='L1 loss')
        plt.plot(x, ks_list, 'g-', label='KS loss')
        if len(r_E_list) > 0:
            plt.plot(x, r_E_list, 'y-', label='r_E reg loss')
    else:
        print(f'Plotting {show_individual_loss}...')
        if show_individual_loss == 'corr_loss':
            plt.plot(x, corr_list, 'r-', label='Corr loss')
        elif show_individual_loss == 'l1_loss':
            plt.plot(x, l1_list, 'b-', label='L1 loss')
        elif show_individual_loss == 'ks_loss':
            plt.plot(x, ks_list, 'g-', label='KS loss')
        elif show_individual_loss == 'r_E_reg_loss':
            plt.plot(x, r_E_list, 'y-', label='r_E reg loss')
        else:
            raise Exception(
                f'Invalid show_individual_loss {show_individual_loss}')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    group_affix = '' if group_idx is None else f'{group_idx}'
    suffix = f' for top {lowest_top_k} children' if lowest_top_k is not None else ' for all children'

    plt.title(
        f'{target.replace("_", " ")} {group_affix} train losses {suffix}')
    plt.legend()
    plt.savefig(save_fig_path)
    plt.close()
    print('Saved to:', save_fig_path)


# @DeprecationWarning
def plot_test_losses_old(ds_name,
                         target,
                         trial_indices,
                         trial_names,
                         seed_idx,
                         loss_types=[
                             'total_loss', 'corr_loss', 'old_l1_loss',
                             'MAE_l1_loss', 'ks_loss'
                         ],
                         use_old_total_loss: bool = True):
    """
    Plot the test losses of different trails

    For each loss from loss_types, draw a box plot, where each box represents a trial,
    and every dot in a box represents 1 of the 10 param vectors selected for testing under the setup of the trial.

    This function assumes that the target's directory contains
    pth files at f'test/trial{trial_idx}/seed{seed_idx}/best_from_test.pth'.
    Each of the file contains a dictionary, with keys ['corr_loss', 'l1_loss', 'old_l1_loss', 'MAE_l1_loss', 'ks_loss']

    Args:
        ds_name (str): The dataset name.
        target (str): The target name.
        trial_indices (list): The list of trial indices.
        trial_names (list): The list of trial names.
        seed_idx (int): The seed index.
        loss_types (list, optional): The list of loss types to plot.
        use_old_total_loss (bool, optional): Whether to use the old total loss, which is corr_loss+old_l1_loss+ks_loss.
    """

    setup_losses = {'setup_name': []}
    for loss_type in loss_types:
        setup_losses[loss_type] = []

    for trial_idx, trial_name in zip(trial_indices, trial_names):

        save_dict = load_best_params(ds_name, target, 'test', trial_idx,
                                     seed_idx)
        if use_old_total_loss:
            save_dict['total_loss'] = save_dict['corr_loss'] + save_dict[
                'old_l1_loss'] + save_dict['ks_loss']

        setup_losses['setup_name'] += [trial_name] * len(
            save_dict['total_loss'])
        for loss_type in loss_types:
            setup_losses[loss_type] += save_dict[loss_type].tolist()

    setup_losses_df = pd.DataFrame(setup_losses)

    plot_setup_losses(setup_losses_df, get_losses_fig_dir(ds_name, target))


def plot_setup_losses(setup_losses_df: pd.DataFrame, fig_save_dir: str):
    columns = setup_losses_df.columns
    # find columns that ends with '_loss'
    trial_names = setup_losses_df['setup_name'].unique()
    trial_pairs = list(combinations(trial_names, 2))
    loss_columns = [col for col in columns if col.endswith('_loss')]
    for loss_type in loss_columns:
        plt.figure()
        ax = sns.boxplot(data=setup_losses_df,
                         x='setup_name',
                         y=loss_type,
                         order=trial_names,
                         orient='v')
        plt.xlabel('Setups')
        # plt.ylabel(loss_type)
        plt.tight_layout()
        fig_save_path = os.path.join(fig_save_dir, f'test_{loss_type}.png')
        plt.savefig(fig_save_path)
        annotator = Annotator(ax,
                              trial_pairs,
                              data=setup_losses_df,
                              x='setup_name',
                              y=loss_type,
                              order=trial_names)
        annotator.configure(
            test='t-test_ind',
            # text_format='full',
            text_format='star',
            loc='inside',
            comparisons_correction=None,
            # text_offset=8,
            show_test_name=False,
            line_height=0.05)
        annotator.apply_and_annotate()

        plt.tight_layout()
        plt.savefig(fig_save_path.replace('.png', '_with_p_value.png'))

        plt.close()


def get_indices(lst, targets):
    """
    Get the indices of the targets in the list.
    # Example usage:
    my_list = [1, 'apple', 3, 'banana', 2, 'orange']
    target = ['apple', 2]
    result = get_indices(my_list, target)
    print(result)  # [1, 4]
    """
    indices = []
    if not isinstance(lst, list):
        lst = lst.tolist()

    for target in targets:
        if target in lst:
            indices.append(lst.index(target))
    return indices


def get_best_val_param_in_test(test_save_dict, top_k_val_params: int = 1):
    """
    Get the saved test results for the best param vector with the lowest validation loss in the test save dict

    Args:
        test_save_dict (dict): The test save dictionary.
    """
    # find the indices for top k param vectors with the lowest validation loss in the test save dict
    val_indices_in_test = get_indices(test_save_dict['valid_param_indices'],
                                      range(top_k_val_params))
    best_val_param_dict = get_values_at_indices(test_save_dict,
                                                val_indices_in_test)
    return best_val_param_dict


def plot_test_losses(ds_name: str,
                     target: str,
                     trial_indices,
                     trial_names: list[str],
                     agg_seeds_num: int,
                     loss_types=[
                         'total_loss', 'corr_loss', 'old_l1_loss',
                         'MAE_l1_loss', 'ks_loss'
                     ],
                     use_old_total_loss: bool = True):
    """
    Plot the test losses of different trails

    For each loss from loss_types, draw a box plot, where each box represents a trial,
    and every dot in a box represents 1 of the 10 param vectors selected for testing under the setup of the trial.

    This function assumes that the target's directory contains
    pth files at f'test/trial{trial_idx}/seed{seed_idx}/best_from_test.pth'.
    Each of the file contains a dictionary, with keys ['corr_loss', 'l1_loss', 'old_l1_loss', 'MAE_l1_loss', 'ks_loss']

    Args:
        ds_name (str): The dataset name.
        target (str): The target name.
        trial_indices (list): The list of trial indices.
        trial_names (list): The list of trial names.
        seed_idx (int): The seed index.
        loss_types (list, optional): The list of loss types to plot.
        use_old_total_loss (bool, optional): Whether to use the old total loss, which is corr_loss+old_l1_loss+ks_loss.
    """

    setup_losses = {'setup_name': []}
    for loss_type in loss_types:
        setup_losses[loss_type] = []

    for trial_idx, trial_name in zip(trial_indices, trial_names):
        agg_seed_idx = 0
        while True:
            agg_seed_idx += 1
            seeds_range, agg_seed_label = get_agg_seeds_range(
                agg_seeds_num, agg_seed_idx)
            save_dict = load_best_params(ds_name, target, 'test', trial_idx,
                                         agg_seed_label)
            if save_dict is None:
                break

            best_val_param_dict = get_best_val_param_in_test(
                save_dict, top_k_val_params=1)
            if use_old_total_loss:
                best_val_param_dict['total_loss'] = best_val_param_dict['corr_loss'] \
                    + best_val_param_dict['old_l1_loss'] + best_val_param_dict['ks_loss']

            setup_losses['setup_name'].append(trial_name)
            for loss_type in loss_types:
                setup_losses[loss_type].append(
                    best_val_param_dict[loss_type].item())

    setup_losses_df = pd.DataFrame(setup_losses)
    plot_setup_losses(setup_losses_df, get_losses_fig_dir(ds_name, target))


############################################################
# EI related analysis
############################################################


def regional_EI_age_slope(n_roi, ages, regional_EIs):

    # regional_EIs = np.zeros((nbr_num, n_roi))    # ages = np.zeros((nbr_num))
    # convert age in months to age in years
    ages = ages / 12
    slope_arr = np.zeros((n_roi))
    pvalue_arr = np.zeros((n_roi))
    for i in range(n_roi):
        res = stats.linregress(ages,
                               regional_EIs[:, i],
                               alternative='two-sided')
        slope_arr[i] = res.slope
        pvalue_arr[i] = res.pvalue
    pvalue_fdr = stats.false_discovery_control(pvalue_arr)
    significant_num = np.sum(pvalue_fdr < 0.05)
    print(pvalue_arr, pvalue_fdr)
    print(f'Significant regions after FDR: {significant_num} / {n_roi}')
    return slope_arr, pvalue_arr, pvalue_fdr


def regional_EI_diff_cohen_d(EI_matrix_high, EI_matrix_low):
    """
    Compute the effect size (cohen's d) of E/I ratio difference, specifically:
    For each ROI, compute Cohen's d for the E/I ratio difference between the two groups (low/high-performance)
    Cohen's d's formula for 1-sample is (mean of the sample)/(sample standard deviation of the sample)
    """
    EI_ratio_diff = EI_matrix_low - EI_matrix_high
    EI_ratio_diff_mean = np.mean(EI_ratio_diff, axis=0)
    EI_ratio_diff_std = np.std(
        EI_ratio_diff, axis=0, ddof=1
    )  # * ddof=1 for sample std (after Bessel's correction), equivalent to MATLAB's default std
    cohen_ds = EI_ratio_diff_mean / EI_ratio_diff_std
    cohen_ds = np.reshape(cohen_ds, (cohen_ds.shape[0], 1))  # Reshape to 68x1
    return cohen_ds


############################################################
# r_E related analysis
############################################################


def boxplot_val_r_E_for_diff_trials(ds_name,
                                    target,
                                    trial_range,
                                    trial_names,
                                    seed_range,
                                    group_idx,
                                    epoch_range,
                                    save_fig_path=None):
    """
    Choose the top 10 sets of validation parameters with the lowest loss from all seeds
    plot a box plot of different trials, where each dot represents a parameter's rE averaged across time and across ROIs
    """
    if save_fig_path is None:

        save_fig_dir = os.path.join(get_target_dir(ds_name, target), 'figures',
                                    'val_r_E_for_diff_trials_boxplot')
        if not os.path.exists(save_fig_dir):
            os.makedirs(save_fig_dir)
        save_fig_path = os.path.join(
            save_fig_dir, f'val_r_E_of_group{group_idx}_boxplot.png')

    print(
        f'Plotting r_E for {target} group {group_idx} across trials {trial_names}...'
    )
    plt.figure()
    all_trials_r_E_ave = []
    for trial_idx in trial_range:
        val_dicts = load_all_val_dicts(ds_name, target, trial_idx, seed_range,
                                       group_idx, epoch_range)
        k_lowest_total_loss, topk_indices = get_val_top_k(val_dicts, k=10)
        r_E_ave_across_time_n_ROI = np.array([
            torch.mean(val_dicts['r_E_for_valid_params'][tuple(i)])
            for i in topk_indices
        ])
        all_trials_r_E_ave.append(r_E_ave_across_time_n_ROI)
    plt.boxplot(all_trials_r_E_ave, labels=trial_names)
    plt.xlabel('Setups')
    plt.ylabel('mean r_E')
    plt.title(f'{target.replace("_", " ")} {group_idx} mean r_E')
    plt.savefig(save_fig_path)
    plt.close()


def boxplot_train_r_E(ds_name,
                      target,
                      trial_idx,
                      seed_idx,
                      group_idx,
                      epoch_range,
                      top_k=None,
                      plot_outlier_r_E=False,
                      save_fig_path=None):
    """
    This function will boxplot the r_E for each child at each ROI at each epoch
    """
    all_epochs_r_E = []
    for epoch_idx in epoch_range:
        saved_dict = load_train_dict(ds_name,
                                     target,
                                     trial_idx,
                                     seed_idx,
                                     group_idx,
                                     epoch_idx,
                                     apply_sort=False)
        # extract the r_E for each of the top_k children at each ROI at each epoch from the 'r_E_for_valid_params' tensor
        ROI_r_E_at_epoch = saved_dict['r_E_for_valid_params']
        if top_k is not None:
            ROI_r_E_at_epoch = ROI_r_E_at_epoch[:, :top_k]

        if plot_outlier_r_E:
            ROI_r_E_deviation_at_epoch = ROI_r_E_at_epoch - 3
            r_E_at_epoch = torch.max(torch.abs(ROI_r_E_deviation_at_epoch),
                                     dim=0).values
        else:
            r_E_at_epoch = torch.mean(ROI_r_E_at_epoch, dim=0)

        all_epochs_r_E.append(r_E_at_epoch.numpy())

    if save_fig_path is None:
        suffix = '_outlier' if plot_outlier_r_E else ''
        group_affix = '' if group_idx is None else f'_of_group{group_idx}'
        if top_k is not None:
            suffix += f'_for_top{top_k}_children'
        save_fig_path = get_fig_path_in_logs(
            ds_name, target, 'train_r_E_boxplot', trial_idx, seed_idx,
            f'r_E{group_affix}_boxplot{suffix}.png')

    plt.figure()
    plt.boxplot(all_epochs_r_E, labels=epoch_range)
    plt.xlabel('Epochs')
    if plot_outlier_r_E:
        plt.ylabel('|r_E - 3|')
    else:
        plt.ylabel('r_E')
    suffix = ' r_E (outlier)' if plot_outlier_r_E else ' mean r_E'
    suffix += f' for top {top_k} children' if top_k is not None else ' for all children'
    group_affix = '' if group_idx is None else f'{group_idx}'
    plt.title(f'{target.replace("_", " ")} {group_affix}{suffix}')
    plt.savefig(save_fig_path)
    plt.close()
    print('Saved to:', save_fig_path)


def export_train_r_E(ds_name,
                     target,
                     trial_idx,
                     seed_idx,
                     group_idx,
                     epoch_idx,
                     save_mat_path=None):
    saved_dict = load_train_dict(ds_name,
                                 target,
                                 trial_idx,
                                 seed_idx,
                                 group_idx,
                                 epoch_idx,
                                 apply_sort=False)

    # extract the r_E at each ROI at each epoch for the best child from the 'r_E_for_valid_params' tensor
    r_E_at_epoch = saved_dict[
        'r_E_for_valid_params'][:, :1]  # ! r_E should be a column vector

    # save the r_E_at_epoch as mat
    group_affix = '' if group_idx is None else f'_of_group{group_idx}'
    if save_mat_path is None:
        save_mat_path = get_fig_path_in_logs(
            ds_name, target, 'train_r_E_surf_map', trial_idx, seed_idx,
            f'r_E{group_affix}_at_epoch{epoch_idx}.mat')
    sio.savemat(save_mat_path, {'r_E': r_E_at_epoch})
    return save_mat_path


def visualize_train_r_E(ds_name,
                        target,
                        trial_idx,
                        seed_idx,
                        group_idx,
                        epoch_idx,
                        fig_title='train r_E'):
    save_mat_path = export_train_r_E(ds_name, target, trial_idx, seed_idx,
                                     group_idx, epoch_idx)
    visualize_stats(save_mat_path, 'r_E',
                    save_mat_path.replace('.mat', '_surf_map.png'), fig_title)


def visualize_train_r_E_for_multi_epochs(ds_name, target, trial_idx, seed_idx,
                                         group_idx, n_epochs, rows, cols):
    """
    Visualize the r_E at each ROI at each epoch for the best child for multiple epochs,
    According to rows and cols, we decide how many surf maps to show: n_surf_maps.
    Then, get n_surf_maps epoch that are equally spaced in range(n_epochs).
        Note that we always include the first and the last epoch.
    Based on the epochs chosen, we export the r_E to mat files using export_train_r_E.
    Then, we can call visualize_stats that takes in list of mat_file_paths, list containing 'r_E', and a list of fig_file_paths.
    Finally, combine these n_surf_maps images into one image using concat_n_images.
    """

    n_surf_maps = rows * cols
    epochs = np.floor(np.linspace(0, n_epochs - 1, n_surf_maps)).astype(int)
    mat_file_paths = []
    stats_names = []
    fig_file_paths = []
    fig_titles = []
    for epoch_idx in epochs:
        mat_file_paths.append(
            export_train_r_E(ds_name, target, trial_idx, seed_idx, group_idx,
                             epoch_idx))
        stats_names.append('r_E')
        fig_file_paths.append(mat_file_paths[-1].replace(
            '.mat', '_surf_map.png'))
        fig_titles.append(f'mean rE across time at epoch {epoch_idx}')

    visualize_stats(mat_file_paths, stats_names, fig_file_paths, fig_titles)

    group_affix = '' if group_idx is None else f'_of_group{group_idx}'
    concat_n_images(
        fig_file_paths, rows, cols,
        get_fig_path_in_logs(ds_name, target, 'train_r_E_surf_map', trial_idx,
                             seed_idx, f'r_E{group_affix}_surf_map.png'))


def merge_train_dict_across_seed(ds_name,
                                 target,
                                 trial_idx,
                                 seed_range,
                                 group_idx,
                                 epoch_range,
                                 save_merged_dict=False):
    """
    Merge values for each key of the dictionaries across seeds and epoch
    """
    merged_dict = {}
    for epoch_idx in epoch_range:
        for seed_idx in seed_range:
            saved_dict = load_train_dict(ds_name, target, trial_idx, seed_idx,
                                         group_idx, epoch_idx)
            for key in saved_dict.keys():
                if key not in merged_dict.keys():
                    merged_dict[key] = []
                merged_dict[key].append(saved_dict[key])
    for key in merged_dict.keys():
        # check the dim of the torch tensor, if the dim is only 1, can concat along the 0th dim,
        # if the dim is 2, need to concat along the 1st dim
        dim = len(merged_dict[key][0].shape)
        if dim == 1:
            merged_dict[key] = torch.cat(merged_dict[key], dim=0)
        elif dim == 2:
            if key == 'FC_FCD_loss':
                merged_dict[key] = torch.cat(merged_dict[key], dim=0)
            else:
                merged_dict[key] = torch.cat(merged_dict[key], dim=1)
        else:
            raise Exception(f'Invalid tensor dim {dim}')

    if save_merged_dict:
        # save to pth file at the trial directory
        save_dir = get_run_dir(ds_name, target, 'train', trial_idx, '_merged')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'group{group_idx}_merged.pth')
        torch.save(merged_dict, save_path)
        print(f'Saved to {save_path}')
    return merged_dict


def plot_scatter(x: np.ndarray,
                 y: np.ndarray,
                 xlabel,
                 ylabel,
                 title,
                 fig_file_path,
                 show_mean=True,
                 show_corr=True,
                 show_reg_line=True):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=14)

    # Add mean values as labels
    if show_mean:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        plt.text(x_mean,
                 y_mean,
                 f'Mean: ({x_mean:.3f}, {y_mean:.3f})',
                 color='red')

    # Add correlation coefficient in the title
    if show_corr:
        corr = np.corrcoef(x, y)[0, 1]
        plt.title(f'{title} (r={corr:.3f})', fontsize=14)

    # Add regression line
    if show_reg_line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, slope * x + intercept, 'r', label='fitted line')
        # plt.legend()

    plt.savefig(fig_file_path)
    plt.close()


def scatter_loss_vs_r_E(ds_name,
                        target,
                        trial_idx,
                        seed_range,
                        group_idx,
                        epoch_range,
                        r_E_range=[float('-inf'), float('inf')],
                        total_loss_range=[float('-inf'),
                                          float('inf')],
                        plot_param_ranges=True):
    merge_dict = merge_train_dict_across_seed(ds_name, target, trial_idx,
                                              seed_range, group_idx,
                                              epoch_range)
    r_E_for_valid_params = merge_dict['r_E_for_valid_params']
    mean_r_E = torch.mean(r_E_for_valid_params, dim=0)
    FC_FCD_loss = merge_dict['FC_FCD_loss']
    total_loss = torch.sum(FC_FCD_loss, dim=1)

    # Filter parameters based on r_E_range and total_loss_range
    valid_indices = np.logical_and.reduce(
        (mean_r_E >= r_E_range[0], mean_r_E <= r_E_range[1], total_loss
         >= total_loss_range[0], total_loss <= total_loss_range[1]))
    mean_r_E = mean_r_E[valid_indices]
    loss_dict = {}
    loss_dict['total'] = total_loss[valid_indices]
    FC_FCD_loss = FC_FCD_loss[valid_indices]
    loss_dict['corr'] = FC_FCD_loss[:, 0]
    loss_dict['l1'] = FC_FCD_loss[:, 1]
    loss_dict['ks'] = FC_FCD_loss[:, 2]

    fig_dir = get_fig_dir_in_logs(ds_name, target, 'scatter_loss_vs_r_E',
                                  trial_idx, "_all")

    group_idx_str = "" if group_idx is None else f"{group_idx}"
    target_str = target.replace("_", " ")
    title_postfix = f' ({np.count_nonzero(valid_indices)} param vectors)'
    fig_file_postfix = f'_loss_vs_r_E_for_group{group_idx}.png' if group_idx is not None else '_loss_vs_r_E.png'
    if len(epoch_range) == 1:
        fig_file_postfix = fig_file_postfix.replace(
            '.png', f'_epoch{epoch_range[0]}.png')
    if r_E_range != [float('-inf'), float('inf')]:
        fig_file_postfix = fig_file_postfix.replace('.png',
                                                    '_with_r_E_range.png')
    if total_loss_range != [float('-inf'), float('inf')]:
        fig_file_postfix = fig_file_postfix.replace(
            '.png', '_with_total_loss_range.png')

    def scatter_helper(cost_type):
        plot_scatter(
            mean_r_E, loss_dict[cost_type], 'mean r_E', f'{cost_type} loss',
            f'{target_str} {group_idx_str} {cost_type} loss vs r_E {title_postfix}',
            os.path.join(fig_dir,
                         f'{cost_type}{fig_file_postfix}'), True, False, False)

    scatter_helper('total')
    scatter_helper('corr')
    scatter_helper('l1')
    scatter_helper('ks')

    if plot_param_ranges:
        param_vectors = merge_dict[
            'parameter']  # [3 * N + 1, num_of_param_vectors]
        param_vectors = param_vectors[:, valid_indices]
        wEE = param_vectors[0:NUM_ROI, :].flatten()
        wEI = param_vectors[NUM_ROI:2 * NUM_ROI, :].flatten()
        G = param_vectors[2 * NUM_ROI, :]
        sigma_times_1000 = param_vectors[2 * NUM_ROI + 1:, :].flatten() * 1000

        # box plot the parameter ranges
        plt.figure()
        plt.boxplot([wEE, wEI, G, sigma_times_1000],
                    labels=['wEE', 'wEI', 'G', 'sigma*1000'])
        plt.title(
            f'{target_str} {group_idx_str} sampled parameter ranges {title_postfix}'
        )
        plt.ylim(-0.5, 10.5)  # Set the y-axis limits
        plt.savefig(
            os.path.join(
                fig_dir,
                f'param_ranges{fig_file_postfix.replace("_loss_vs_r_E", "")}'))
        plt.close()
