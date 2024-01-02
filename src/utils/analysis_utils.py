import os
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats, io as sio
import torch

from src.basic.constants import MATLAB_SCRIPT_PATH
from src.utils.file_utils import get_fig_file_path, get_losses_fig_dir, get_run_path, get_target_path, load_all_val_dicts, load_train_dict

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
            losses_file_dir = get_run_path(ds_name, target, 'test', trial_idx,
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
        _, lowest_top_k_indices = get_train_top_k(d, k=lowest_top_k)
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

        save_fig_dir = os.path.join(get_target_path(ds_name,
                                                    target), 'figures',
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
                      plot_outlier_r_E=False,
                      save_fig_path=None):
    """
    This function will boxplot the r_E for each child at each ROI at each epoch
    """
    all_epochs_r_E = []
    for epoch_idx in epoch_range:
        saved_dict = load_train_dict(ds_name, target, trial_idx, seed_idx,
                                     group_idx, epoch_idx)
        _, top_k_indices = get_train_top_k(saved_dict, k=None)
        r_E_for_valid_params = saved_dict['r_E_for_valid_params']

        # extract the r_E for each child at each ROI at each epoch from the 'r_E_for_valid_params' tensor
        ROI_r_E_at_epoch = r_E_for_valid_params[:, top_k_indices]
        if plot_outlier_r_E:
            ROI_r_E_deviation_at_epoch = ROI_r_E_at_epoch - 3
            r_E_at_epoch = torch.max(ROI_r_E_deviation_at_epoch, dim=0).values
        else:
            r_E_at_epoch = torch.mean(ROI_r_E_at_epoch, dim=0)

        all_epochs_r_E.append(r_E_at_epoch.numpy())

    if save_fig_path is None:
        postfix = '_outlier' if plot_outlier_r_E else ''
        save_fig_path = get_fig_file_path(
            ds_name, target, 'train_r_E_boxplot', trial_idx, seed_idx,
            f'r_E_of_group{group_idx}_boxplot{postfix}.png')

    plt.figure()
    plt.boxplot(all_epochs_r_E, labels=epoch_range)
    plt.xlabel('Epochs')
    plt.ylabel('r_E')
    postfix = ' r_E (outlier)' if plot_outlier_r_E else ' mean r_E'
    plt.title(f'{target.replace("_", " ")} {group_idx}{postfix}')
    plt.savefig(save_fig_path)
    plt.close()
    print('Saved.')


def export_train_r_E(ds_name,
                     target,
                     trial_idx,
                     seed_idx,
                     group_idx,
                     epoch_idx,
                     save_mat_path=None):
    saved_dict = load_train_dict(ds_name, target, trial_idx, seed_idx,
                                 group_idx, epoch_idx)
    _, top_k_indices = get_train_top_k(saved_dict, k=1)
    r_E_for_valid_params = saved_dict['r_E_for_valid_params']

    # extract the r_E for each child at each ROI at each epoch from the 'r_E_for_valid_params' tensor
    r_E_at_epoch = r_E_for_valid_params[:, top_k_indices]

    # save the r_E_at_epoch as mat
    if save_mat_path is None:
        save_mat_path = get_fig_file_path(
            ds_name, target, 'train_r_E_surf_map', trial_idx, seed_idx,
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
