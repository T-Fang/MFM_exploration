import os
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, io as sio
import torch

from src.basic.constants import MATLAB_SCRIPT_PATH
from src.utils.file_utils import get_fig_file_path, get_losses_fig_dir, get_run_path, load_train_dict

############################################################
# Visualization related
############################################################


def visualize_stats(mat_file_path, stats_name, fig_file_path):
    """
    First, cd into MATLAB_SCRIPT_PATH.
    Then, load the [68, 1] statistic (one scalar for each ROI) stored in the `mat_file_path` file.
    Finally, run the visualize_parameter_desikan_fslr function to visualize the stats.
    """

    print(f"Visualizing {stats_name} from {mat_file_path}...")

    command = [
        (f"cd {MATLAB_SCRIPT_PATH}; "
         f"matlab -nodisplay -nosplash -nodesktop -r "
         f"\"load('{mat_file_path}', '{stats_name}'); "
         f"visualize_parameter_desikan_fslr({stats_name}, '{fig_file_path}'); "
         f"exit;\"")
    ]

    result = subprocess.run(command,
                            shell=True,
                            capture_output=True,
                            text=True)
    print(result.stdout)
    print(result.stderr)

    print(f'Visualization saved to {fig_file_path}')


def boxplot_network_stats(mat_file_path, stats_name, fig_file_path):
    """
    First, cd into MATLAB_SCRIPT_PATH.
    Then, load the [68, 1] statistic (one scalar for each ROI) stored in the `mat_file_path` file.
    Finally, run the visualize_parameter_desikan_fslr function to
    generates a box plot depicting the network pattern of the input statistic.
    """

    print(f"Boxplotting {stats_name} from {mat_file_path}...")

    command = [(
        f"cd {MATLAB_SCRIPT_PATH}; "
        f"matlab -nodisplay -nosplash -nodesktop -r "
        f"\"load('{mat_file_path}', '{stats_name}'); "
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
# Loss related analysis
############################################################


def get_train_top_k_indices(saved_dict, top_k=1):
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

    # sort the total losses and find the index for the child with the lowest total loss
    sorted_index = torch.argsort(total_loss)
    best_child_idx = sorted_index[:top_k]

    return best_child_idx


def plot_losses_for_diff_trials(
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
        data = []
        for trial_idx in trial_range:
            losses_file_dir = get_run_path(ds_name, target, 'test', trial_idx,
                                           '_best_among_all')
            losses_dict = torch.load(os.path.join(losses_file_dir,
                                                  'lowest_losses.pth'),
                                     map_location='cpu')
            losses = losses_dict[loss_type]
            data.append(losses.numpy())
        plt.boxplot(data, labels=trial_names)
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
        postfix = '' if show_individual_loss is None else f'_{show_individual_loss}'
        save_fig_path = get_fig_file_path(
            ds_name, target, 'losses', trial_idx, seed_idx,
            f'group{group_idx}_train_losses{postfix}.png')

    corr_list = []
    l1_list = []
    ks_list = []
    r_E_list = []
    for epoch_idx in epoch_range:
        d = load_train_dict(ds_name, target, trial_idx, seed_idx, group_idx,
                            epoch_idx)
        lowest_top_k_indices = get_train_top_k_indices(d, top_k=lowest_top_k)
        lowest_top_k_indices = lowest_top_k_indices.numpy()
        corr_losses = d['FC_FCD_loss'][:, 0][lowest_top_k_indices]
        corr_list.append(torch.mean(corr_losses).item())
        l1_losses = d['FC_FCD_loss'][:, 1][lowest_top_k_indices]
        l1_list.append(torch.mean(l1_losses).item())
        ks_losses = d['FC_FCD_loss'][:, 2][lowest_top_k_indices]
        ks_list.append(torch.mean(ks_losses).item())
        if 'r_E_reg_loss' in d.keys():
            r_E_reg_losses = d['r_E_reg_loss'][lowest_top_k_indices]
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
    plt.title(f'{target.replace("_", " ")} {group_idx} train losses')
    plt.legend()
    plt.savefig(save_fig_path)
    plt.close()
    print('Saved.')


############################################################
# EI related analysis
############################################################


def regional_EI_age_slope(n_roi, ages, regional_EIs):

    # regional_EIs = np.zeros((nbr_num, n_roi))    # ages = np.zeros((nbr_num))

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


def export_train_r_E(ds_name,
                     target,
                     trial_idx,
                     seed_idx,
                     group_idx,
                     epoch_idx,
                     save_mat_path=None):
    saved_dict = load_train_dict(ds_name, target, trial_idx, seed_idx,
                                 group_idx, epoch_idx)
    top_k_indices = get_train_top_k_indices(saved_dict, top_k=1)
    r_E_for_valid_params = saved_dict['r_E_for_valid_params']

    # extract the r_E for each child at each ROI at each epoch from the 'r_E_for_valid_params' tensor
    r_E_at_epoch = r_E_for_valid_params[:, top_k_indices]

    # save the r_E_at_epoch as mat
    if save_mat_path is None:
        save_mat_path = get_fig_file_path(
            ds_name, target, 'train_r_E', trial_idx, seed_idx,
            f'r_E_of_group{group_idx}_at_epoch{epoch_idx}')
    sio.savemat(save_mat_path, {'r_E': r_E_at_epoch})


def visualize_train_r_E(ds_name, target, trial_idx, seed_idx, group_idx,
                        epoch_idx):
    save_mat_path = get_fig_file_path(
        ds_name, target, 'train_r_E', trial_idx, seed_idx,
        f'r_E_of_group{group_idx}_at_epoch{epoch_idx}.mat')
    export_train_r_E(ds_name, target, trial_idx, seed_idx, group_idx,
                     epoch_idx, save_mat_path)
    visualize_stats(save_mat_path, 'r_E',
                    save_mat_path.replace('.mat', '_surf_map.png'))
