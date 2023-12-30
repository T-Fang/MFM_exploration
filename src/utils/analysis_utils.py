import os
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import torch

from src.basic.constants import LOG_PATH, MATLAB_SCRIPT_PATH
############################################################
# Path related
############################################################


def get_target_path(ds_name, target):
    target_path = os.path.join(LOG_PATH, ds_name, target)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    return target_path


def get_run_path(ds_name, target, mode, trial_idx, seed_idx):
    run_path = os.path.join(get_target_path(ds_name, target), mode,
                            f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    return run_path


def get_group_path(ds_name, target, mode, trial_idx, seed_idx, group_idx):
    run_path = os.path.join(get_target_path(ds_name,
                                            target), mode, f'trial{trial_idx}',
                            f'seed{seed_idx}', f'group{group_idx}')
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    return run_path


def get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx):
    fig_dir = os.path.join(get_target_path(ds_name, target), 'figures',
                           fig_type, f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir


def get_losses_fig_dir(ds_name, target):
    losses_fig_dir = os.path.join(get_target_path(ds_name, target), 'figures',
                                  'losses')
    if not os.path.exists(losses_fig_dir):
        os.makedirs(losses_fig_dir)
    return losses_fig_dir


def get_fig_file_path(ds_name, target, fig_type, trial_idx, seed_idx,
                      fig_name):
    fig_dir = get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx)
    return os.path.join(fig_dir, fig_name)


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
                    save_fig_path=None):
    param_save_dir = get_group_path(ds_name, target, 'train', trial_idx,
                                    seed_idx, group_idx)
    if save_fig_path is None:
        save_fig_path = get_fig_file_path(
            ds_name, target, 'losses', trial_idx, seed_idx,
            f'group{group_idx}_train_losses.png')

    corr_list = []
    l1_list = []
    ks_list = []
    r_E_list = []
    for i in epoch_range:
        d = torch.load(os.path.join(param_save_dir,
                                    f'param_save_epoch{i}.pth'))
        pred_all_losses = d['FC_FCD_loss']

        corr_list.append(torch.mean(pred_all_losses[:, 0]).item())
        l1_list.append(torch.mean(pred_all_losses[:, 1]).item())
        ks_list.append(torch.mean(pred_all_losses[:, 2]).item())

        if 'r_E_reg_loss' in d:
            r_E_reg_loss = d['r_E_reg_loss']
            r_E_list.append(torch.mean(r_E_reg_loss).item())

    x = np.array(epoch_range)
    plt.figure()
    plt.plot(x, corr_list, 'r-', label='Corr loss')
    plt.plot(x, l1_list, 'b-', label='L1 loss')
    plt.plot(x, ks_list, 'g-', label='KS loss')
    if len(r_E_list) > 0:
        plt.plot(x, r_E_list, 'y-', label='r_E reg loss')
    plt.xlabel('iteration')
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
