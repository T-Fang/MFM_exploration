import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio

from src.basic.constants import PREV_PHASE, DEFAULT_DTYPE, LOG_DIR, MATLAB_SCRIPT_DIR, TMP_DIR, HCPYA_1029_DATA_DIR, FIGURES_DIR
from src.utils.SC_utils import group_SC_matrices
from src.utils.FC_utils import fisher_average

############################################################
# Files and directories
############################################################


def get_fig_dir(ds_name, target):
    target_path = os.path.join(FIGURES_DIR, ds_name, target)
    os.makedirs(target_path, exist_ok=True)
    return target_path


def get_emp_fig_dir(ds_name, target, fig_type):
    fig_dir = get_fig_dir(ds_name, target)
    emp_fig_dir = os.path.join(fig_dir, 'empirical', fig_type)
    os.makedirs(emp_fig_dir, exist_ok=True)
    return emp_fig_dir


def get_sim_res_dir(ds_name, target, phase, trial_idx, seed_idx):
    sim_fig_dir = get_fig_dir_in_logs(ds_name, target,
                                      f'best_from_prev_phase_sim_on_{phase}',
                                      trial_idx, seed_idx)

    return sim_fig_dir


def get_vis_param_dir(ds_name, target, phase, trial_idx, seed_idx):
    vis_param_dir = get_fig_dir_in_logs(ds_name, target,
                                        f'best_from_{phase}_vis_param',
                                        trial_idx, seed_idx)

    return vis_param_dir


def get_sim_res_path(ds_name, target, phase, trial_idx, seed_idx, fig_name):
    sim_fig_dir = get_sim_res_dir(ds_name, target, phase, trial_idx, seed_idx)
    return os.path.join(sim_fig_dir, fig_name)


def get_target_dir(ds_name, target):
    target_path = os.path.join(LOG_DIR, ds_name, target)
    os.makedirs(target_path, exist_ok=True)
    return target_path


def get_trial_dir(ds_name, target, trial_idx):
    trial_path = os.path.join(get_target_dir(ds_name, target),
                              f'trial{trial_idx}')
    os.makedirs(trial_path, exist_ok=True)
    return trial_path


def get_run_dir(ds_name, target, phase, trial_idx, seed_idx):
    run_path = os.path.join(get_target_dir(ds_name, target), phase,
                            f'trial{trial_idx}', f'seed{seed_idx}')
    os.makedirs(run_path, exist_ok=True)
    return run_path


def get_group_dir(ds_name, target, phase, trial_idx, seed_idx, group_idx=None):
    # ! For dataset without groups like HCP, only run_dir will be returned
    run_dir = get_run_dir(ds_name, target, phase, trial_idx, seed_idx)
    if group_idx is None:
        return run_dir
    group_dir = os.path.join(run_dir, f'group{group_idx}')
    os.makedirs(group_dir, exist_ok=True)
    return group_dir


def get_fig_dir_in_logs(ds_name, target, fig_type, trial_idx, seed_idx):
    fig_dir = os.path.join(get_target_dir(ds_name, target), 'figures',
                           fig_type)
    if trial_idx is not None:
        fig_dir = os.path.join(fig_dir, f'trial{trial_idx}')

    if seed_idx is not None:
        fig_dir = os.path.join(fig_dir, f'seed{seed_idx}')

    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def get_train_file_path(train_save_dir, epoch_idx):
    return os.path.join(train_save_dir, f'param_save_epoch{epoch_idx}.pth')


def get_best_params_file_path(phase, save_dir):
    return os.path.join(save_dir, f'best_from_{phase}.pth')


def get_best_params_sim_res_path(sim_phase, save_dir):
    return os.path.join(save_dir,
                        f'best_from_prev_phase_sim_on_{sim_phase}.pth')


def get_val_file_path(val_dir, epoch_idx):
    return os.path.join(val_dir, f'best_param{epoch_idx}.pth')


def get_losses_fig_dir(ds_name, target):
    losses_fig_dir = os.path.join(get_target_dir(ds_name, target), 'figures',
                                  'losses')
    if not os.path.exists(losses_fig_dir):
        os.makedirs(losses_fig_dir)
    return losses_fig_dir


def get_fig_path_in_logs(ds_name, target, fig_type, trial_idx, seed_idx,
                         fig_name):
    fig_dir = get_fig_dir_in_logs(ds_name, target, fig_type, trial_idx,
                                  seed_idx)
    return os.path.join(fig_dir, fig_name)


def get_agg_seeds_range(agg_seeds_num, agg_seed_idx):
    """
    Get the range of seeds to aggregate

    Parameters
    ----------
    agg_seeds_num : int
        The number of seeds to aggregate
    seed_idx : int
        The index of the aggregated seed

    """
    seed_start = (agg_seed_idx - 1) * agg_seeds_num + 1
    seed_end = agg_seed_idx * agg_seeds_num
    agg_seed_label = f'_seed{seed_start}_to_seed{seed_end}'
    return range(seed_start, seed_end + 1), agg_seed_label


def get_prev_phase_best_params_path(ds_name,
                                    target,
                                    phase,
                                    trial_idx,
                                    seed_idx,
                                    agg_seeds_num: int = None):

    if agg_seeds_num is not None and phase != 'train':
        seeds_range, seed_idx = get_agg_seeds_range(agg_seeds_num, seed_idx)
    prev_phase = PREV_PHASE[phase]
    prev_phase_save_dir = get_run_dir(ds_name, target, prev_phase, trial_idx,
                                      seed_idx)
    prev_phase_best_params_path = get_best_params_file_path(
        prev_phase, prev_phase_save_dir)

    # if we haven't saved best params from previous phase, get it from all seeds
    if agg_seeds_num is not None and phase != 'train' and not os.path.exists(
            prev_phase_best_params_path):
        prev_phase_all_seeds_save_dir = [
            get_run_dir(ds_name, target, prev_phase, trial_idx, seed_i)
            for seed_i in seeds_range
        ]
        prev_phase_all_seeds_best_params_path = [
            get_best_params_file_path(prev_phase, prev_phase_save_dir)
            for prev_phase_save_dir in prev_phase_all_seeds_save_dir
        ]
        combine_all_param_dicts(
            paths_to_dicts=prev_phase_all_seeds_best_params_path,
            combined_dict_save_path=prev_phase_best_params_path)

    return prev_phase_best_params_path


def get_curr_phase_save_dir(ds_name,
                            target,
                            phase,
                            trial_idx,
                            seed_idx,
                            agg_seeds_num: int = None):
    if agg_seeds_num is not None and phase != 'train':
        seeds_range, seed_idx = get_agg_seeds_range(agg_seeds_num, seed_idx)
    return get_run_dir(ds_name, target, phase, trial_idx, seed_idx)


############################################################
# Load intermediate files
############################################################
def get_best_train_params(train_save_dir,
                          num_epochs: int = 100,
                          top_k_for_each_epoch: int = 1):
    """
    Firstly, load the dict for each epoch under 'train_save_dir'.
    Afterwards, get the top_k_for_each_epoch param vector along with their costs for each train epoch.
    Then combine them into a dict with the same structure.
    Finally, save the dict as 'best_from_train.pth' under 'train_save_dir'
    """

    train_save_files = [
        get_train_file_path(train_save_dir, ep) for ep in range(num_epochs)
    ]

    # save the top few param vectors with the lowest validation loss
    best_from_train_file_path = get_best_params_file_path(
        'train', train_save_dir)
    if top_k_for_each_epoch != 1:
        best_from_train_file_path = best_from_train_file_path.replace(
            '.pth', f'_top{top_k_for_each_epoch}_per_epoch.pth')
    best_from_train = combine_all_param_dicts(
        is_params_sorted=False,
        paths_to_dicts=train_save_files,
        top_k_per_dict=top_k_for_each_epoch,
        combined_dict_save_path=best_from_train_file_path)

    print(
        f"Successfully saved the top {top_k_for_each_epoch} parameters from each train epoch to: {best_from_train_file_path}"
    )

    return best_from_train


def load_best_params(
    ds_name: str,
    target: str,
    phase: int,
    trial_idx: int,
    seed_idx: int | str,
):
    """
    Load the best parameters from the given phase.
    """
    save_dir = get_run_dir(ds_name, target, phase, trial_idx, seed_idx)
    best_params_file_path = get_best_params_file_path(phase, save_dir)
    if not os.path.exists(best_params_file_path):
        return None
    best_params = torch.load(best_params_file_path, map_location='cpu')
    return best_params


def load_train_dict(ds_name,
                    target,
                    trial_idx,
                    seed_idx,
                    group_idx,
                    epoch_idx,
                    apply_sort=False):
    """
    the saved pth file contains a dictionary with keys ['valid_param_indices', 'corr_loss', 'l1_loss', 'ks_loss']
        and possibly 'r_E_reg_loss', 'r_E_for_valid_params'
    The 'valid_param_indices' is a mask of size (num_valid, ) indicating which children of the current epoch are valid
    The 'corr_loss', 'l1_loss', 'ks_loss' are tensors of size (num_valid, ) containing FC corr loss, FC l1 loss, and FCD KS loss for each child
        Note: previously the three losses are combined into 'FC_FCD_loss',
        which is a tensor of size (num_valid, 3) containing FC corr loss, FC l1 loss, and FCD KS loss for each child
    if 'r_E_for_valid_params' is present, it is a tensor of size (num_of_ROI, num_valid) containing r_E for each child at each ROI at each epoch
    if 'r_E_reg_loss' is present, it is a tensor of size (num_valid, ) containing r_E regularization loss for each child

    * Note that we will only return the dictionary after applying the mask valid_param_indices
    """
    train_dir = get_group_dir(ds_name, target, 'train', trial_idx, seed_idx,
                              group_idx)
    # Check if the file exists
    dict_path = os.path.join(train_dir, f'param_save_epoch{epoch_idx}.pth')
    # if not os.path.exists(dict_path):
    #     # TODO: Temporary fix for the case when the file does not exist
    #     while epoch_idx >= 0:
    #         epoch_idx -= 1
    #         dict_path = os.path.join(train_dir,
    #                                  f'param_save_epoch{epoch_idx}.pth')
    #         if os.path.exists(dict_path):
    #             break
    #     # return None

    saved_dict: dict = torch.load(dict_path, map_location='cpu')

    if not apply_sort:
        return saved_dict

    if 'FC_FCD_loss' in saved_dict.keys():
        FC_FCD_loss = saved_dict.pop('FC_FCD_loss', None)
        saved_dict['corr_loss'] = FC_FCD_loss[:, 0]
        saved_dict['l1_loss'] = FC_FCD_loss[:, 1]
        saved_dict['ks_loss'] = FC_FCD_loss[:, 2]

    # If total_loss already exist
    if 'total_loss' in saved_dict.keys():
        total_loss = saved_dict['total_loss']
    else:
        # if total_loss does not exist, we have to calculate it
        total_loss = saved_dict['corr_loss'] + saved_dict[
            'l1_loss'] + saved_dict['ks_loss']
        if 'r_E_reg_loss' in saved_dict.keys():
            total_loss += saved_dict['r_E_reg_loss']
        if 'rFIC_reg_loss' in saved_dict.keys():
            total_loss += saved_dict['rFIC_reg_loss']

    total_loss, sorted_index = torch.sort(total_loss, descending=False)

    saved_dict['total_loss'] = total_loss
    saved_dict['corr_loss'] = saved_dict['corr_loss'][sorted_index]
    saved_dict['l1_loss'] = saved_dict['l1_loss'][sorted_index]
    saved_dict['ks_loss'] = saved_dict['ks_loss'][sorted_index]
    saved_dict['valid_param_indices'] = saved_dict['valid_param_indices'][
        sorted_index]
    if 'r_E_reg_loss' in saved_dict.keys():
        saved_dict['r_E_reg_loss'] = saved_dict['r_E_reg_loss'][sorted_index]

    if 'rFIC_reg_loss' in saved_dict.keys():
        saved_dict['rFIC_reg_loss'] = saved_dict['rFIC_reg_loss'][sorted_index]

    if 'r_E_for_valid_params' in saved_dict.keys():
        saved_dict['r_E_for_valid_params'] = saved_dict[
            'r_E_for_valid_params'][:, sorted_index]

    return saved_dict


def get_values_at_indices(saved_dict: dict, indices=None):
    """
    Get the values at the given indices from the given dictionary.
    """
    if indices is None:
        return saved_dict
    new_dict = {}
    for key, value in saved_dict.items():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        if value.ndim == 0:
            new_dict[key] = value
            continue
        assert value.ndim == 1 or value.ndim == 2, f"The value for {key} is not a 1D or 2D tensor"
        value = value[indices] if value.ndim == 1 else value[:, indices]
        new_dict[key] = value

    return new_dict


def sort_dict_by_total_loss(saved_dict, top_k=None):
    sorted_indices = torch.argsort(saved_dict['total_loss'], descending=False)
    new_save_dict = get_values_at_indices(saved_dict, sorted_indices)
    if top_k is not None:
        new_save_dict = get_values_at_indices(new_save_dict, range(top_k))

    return new_save_dict


def get_first_k_values(values, k=None):
    """
    Get the first k values from the given values tensor.
    The given values can be either a 1D or 2D tensor.
    If k is None, return the original values tensor.
    If values is not a tensor, convert it to a tensor first.
    If values is a 1D tensor, return the first k values as a 2D column vector tensor.
    If values is a 2D tensor, return the first k columns.
    """
    # convert to tensor if values is not a tensor
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if values.ndim == 0:
        return values.unsqueeze(0).unsqueeze(0)

    if values.ndim == 1:
        if k is not None:
            values = values[:k]
        values = values.unsqueeze(0)
    elif values.ndim == 2:
        if k is not None:
            values = values[:, :k]
    else:
        raise Exception("The input `values` is not a 1D or 2D tensor")

    return values


def merge_all_train_dicts(ds_name,
                          target,
                          trial_idx,
                          seed_range,
                          epoch_range=range(100)):
    all_train_dict_paths = []
    for seed_idx in seed_range:
        train_save_dir = get_run_dir(ds_name, target, 'train', trial_idx,
                                     seed_idx)
        for epoch_idx in epoch_range:
            all_train_dict_paths.append(
                get_train_file_path(train_save_dir, epoch_idx))

    merged_train_dict = combine_all_param_dicts(all_train_dict_paths,
                                                is_params_sorted=False)

    return merged_train_dict


def combine_all_param_dicts(paths_to_dicts: list[str],
                            is_params_sorted: bool = True,
                            top_k_per_dict: int = None,
                            top_k_among_all_dicts: int = None,
                            combined_dict_save_path: str = None):
    """
    Combine all param vector dictionaries in the given list of paths to a single dictionary.
    """
    combined_dict = {}
    for path_to_dict in paths_to_dicts:
        if not os.path.exists(path_to_dict):
            raise Exception("save file doesn't exist:", path_to_dict)
        d = torch.load(path_to_dict, map_location='cpu')
        if not is_params_sorted:
            d['parameter'] = d['parameter'][:, d['valid_param_indices']]
        for key, value in d.items():
            if key not in combined_dict:
                combined_dict[key] = []

            value = get_first_k_values(value, k=top_k_per_dict)
            combined_dict[key].append(value)

    for key, value in combined_dict.items():
        combined_dict[key] = torch.cat(value, dim=1).squeeze()

    # get value's shape for every key
    # for k, v in combined_dict.items():
    #     print(f"{k}: {v.shape}")

    combined_dict = sort_dict_by_total_loss(combined_dict,
                                            top_k_among_all_dicts)
    if combined_dict_save_path is not None:
        torch.save(combined_dict, combined_dict_save_path)
        print(f"Combined dict saved to {combined_dict_save_path}")

    return combined_dict


def load_all_val_dicts(ds_name, target, trial_idx, seed_range, group_idx,
                       epoch_range):
    """
    Load all validation dictionaries for the given trial and group.

    This function will still return a dictionary with the same keys.
    But the value of each key in the dictionary now has an extra two dimensions (seed, epoch)
    compared to the original dictionary value, which should all be torch tensors.
    """
    val_dicts = {}
    for seed_idx in seed_range:
        for epoch_idx in epoch_range:
            val_group_dir = get_group_dir(ds_name, target, 'validation',
                                          trial_idx, seed_idx, group_idx)
            val_dict_path = os.path.join(val_group_dir,
                                         f'best_param{epoch_idx}.pth')
            # check if the file exists
            if not os.path.exists(val_dict_path):
                continue

            val_dict = torch.load(val_dict_path, map_location='cpu')

            if len(val_dicts) == 0:
                for key in val_dict.keys():
                    if hasattr(val_dict[key], "shape"):
                        val_dicts[key] = torch.full(
                            (len(seed_range), len(epoch_range)) +
                            val_dict[key].shape, float('nan'))
            for key in val_dict.keys():
                if hasattr(val_dict[key], "shape"):
                    val_dicts[key][seed_idx - 1, epoch_idx] = val_dict[key]

    return val_dicts


def merge_mat_files(mat_file_path_list, output_file_path):
    """
    Merge multiple mat files into one mat file
    """
    mat_dict = {}
    for mat_file_path in mat_file_path_list:
        mat_dict.update(sio.loadmat(mat_file_path))
    sio.savemat(output_file_path, mat_dict)


def convert_to_mat(data, file_name, save_dir=TMP_DIR):
    """
    Convert the data to a mat file at the given dir.

    The given data can be a file_path to (.mat or .csv file),
        pd.DataFrame, np.ndarray, torch.Tensor, or list of values
    """
    save_file_path = os.path.join(save_dir, f'{file_name}.mat')
    if isinstance(data, str):
        if data.endswith('.mat'):
            return data
        elif data.endswith('.csv'):
            data = pd.read_csv(data, header=None)
        else:
            raise ValueError(
                "The given data file must be either a .mat or .csv file")

    data = convert_to_numpy(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    sio.savemat(save_file_path, {file_name: data})

    print(f"{file_name} saved to {save_file_path}")
    return save_file_path


def convert_to_numpy(data):
    """
    Convert the data to a numpy array.

    The given data can be either
        pd.DataFrame, np.ndarray, torch.Tensor, or list of values
    """

    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, np.ndarray):
        pass
    elif isinstance(data, torch.Tensor):
        data = data.to(device=torch.device('cpu')).numpy()
    elif isinstance(data, list):
        data = np.array(data)
    else:
        raise ValueError(
            "The given data must be either a pd.DataFrame, np.ndarray, torch.Tensor, or list of values"
        )

    return data


############################################################
# HCPYA data related
############################################################


def get_HCPYA_group_stats(range_start: int, range_end: int):
    myelin = get_HCPYA_group_myelin(range_start, range_end)
    rsfc_gradient = get_HCPYA_group_rsfc_gradient(range_start, range_end)
    sc_mat = get_HCPYA_group_sc_mat(range_start, range_end)
    sc_euler = sc_mat / torch.max(sc_mat) * 0.02
    emp_fc = get_HCPYA_group_emp_fc(range_start, range_end)
    emp_fcd_cum = get_HCPYA_group_emp_fcd_cdf(range_start, range_end)

    group_stats = {
        'myelin': myelin,
        'rsfc_gradient': rsfc_gradient,
        'sc_mat': sc_mat,
        'sc_euler': sc_euler,
        'emp_fc': emp_fc,
        'emp_fcd_cum': emp_fcd_cum
    }

    return group_stats


def get_HCPYA_group_stats_for_phase(phase):
    if phase == 'train':
        range_start = 0
        range_end = 343
    elif phase == 'val':
        range_start = 343
        range_end = 686
    elif phase == 'test':
        range_start = 686
        range_end = 1029
    else:
        raise ValueError(
            "The given phase must be either 'train', 'val', or 'test")

    return get_HCPYA_group_stats(range_start, range_end)


def get_HCPYA_group_myelin(range_start: int, range_end: int):
    indiv_emp_myelin = sio.loadmat(
        os.path.join(HCPYA_1029_DATA_DIR, 'myelin_DK68_1029.mat'))
    indiv_emp_myelin = indiv_emp_myelin['myelin_DK68_1029']
    myelin = np.nanmean(indiv_emp_myelin[range_start:range_end], axis=0)
    myelin = torch.as_tensor(myelin).to(DEFAULT_DTYPE).unsqueeze(1)
    return myelin


def get_HCPYA_group_rsfc_gradient(range_start: int, range_end: int):
    indiv_emp_rsfc = sio.loadmat(
        os.path.join(HCPYA_1029_DATA_DIR, 'rsfc_DK68_1029.mat'))
    indiv_emp_rsfc = indiv_emp_rsfc['rsfc_DK68_1029']
    rsfc_gradient = np.nanmean(indiv_emp_rsfc[range_start:range_end], axis=0)
    rsfc_gradient = torch.as_tensor(rsfc_gradient).to(DEFAULT_DTYPE).unsqueeze(
        1)
    return rsfc_gradient


def get_HCPYA_group_sc_mat(range_start: int, range_end: int):
    indiv_emp_sc = sio.loadmat(
        os.path.join(HCPYA_1029_DATA_DIR, 'sc_DK68_1029.mat'))
    indiv_emp_sc = indiv_emp_sc['sc_DK68_1029']
    sc_mat = group_SC_matrices(indiv_emp_sc[range_start:range_end])
    sc_mat = torch.as_tensor(sc_mat).to(DEFAULT_DTYPE)
    return sc_mat


def get_HCPYA_group_emp_fc(range_start: int, range_end: int):
    indiv_emp_fc = sio.loadmat(
        os.path.join(HCPYA_1029_DATA_DIR, 'fc_DK68_1029.mat'))
    indiv_emp_fc = indiv_emp_fc['fc_DK68_1029']
    emp_fc = np.array(indiv_emp_fc[range_start:range_end])
    emp_fc = fisher_average(emp_fc)
    emp_fc = torch.as_tensor(emp_fc).to(DEFAULT_DTYPE)
    return emp_fc


def get_HCPYA_group_emp_fcd_cdf(range_start: int, range_end: int):
    indiv_emp_fcd = sio.loadmat(
        os.path.join(HCPYA_1029_DATA_DIR, 'fcd_cdf_DK68_1029.mat'))
    indiv_emp_fcd = indiv_emp_fcd['fcd_cdf_DK68_1029']
    emp_fcd_cum = torch.as_tensor(
        indiv_emp_fcd[range_start:range_end]).to(DEFAULT_DTYPE)
    emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
    emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)
    return emp_fcd_cum


def get_HCPYA_emp_TC(subject_id):
    """
    Get the empirical time course data for the given subject ID in the HCPYA dataset.
    """
    subject_id = int(subject_id)
    TC_dir = os.path.join(HCPYA_1029_DATA_DIR, 'TC')
    TC_file_path = os.path.join(TC_dir, f'{subject_id}_bold_4_runs.mat')
    TC_file_path_after_GSR = os.path.join(TC_dir,
                                          f'{subject_id}_bold_valid_runs.mat')

    if os.path.exists(TC_file_path):
        TC = sio.loadmat(TC_file_path)
    elif os.path.exists(TC_file_path_after_GSR):
        TC = sio.loadmat(TC_file_path_after_GSR)
    else:
        print(f'{subject_id} TC file not found!')
        return None
    TC = TC['bold_TC']
    TC = torch.as_tensor(TC).to(DEFAULT_DTYPE)

    emp_TC = torch.mean(TC, dim=0)
    assert emp_TC.shape[0] == 68 and emp_TC.shape[
        1] == 1200, f"The time course's shape should be (68, 1200), but we get a shape of {emp_TC.shape}"

    print(
        f'Empirical time course (averaged across runs) loaded for subject {subject_id}'
    )
    return emp_TC


def get_HCPYA_group_emp_TC(range_start: int,
                           range_end: int,
                           save_result: bool = True):
    """
    Get the averaged empirical time course data for the given group in the HCPYA dataset.

    Args:
        range_start (int): The starting index of the group.
        range_end (int): The ending index of the group.

    Returns:
        group_TC (torch.Tensor): The averaged empirical time course data for the given group.
    """
    subject_ids_path = os.path.join(HCPYA_1029_DATA_DIR, 'subject_1029.csv')
    subject_ids = pd.read_csv(subject_ids_path, header=None).values.flatten()

    group_TC_save_dir = os.path.join(HCPYA_1029_DATA_DIR, 'TC', 'group_TC')
    os.makedirs(group_TC_save_dir, exist_ok=True)

    group_TC_save_path = os.path.join(
        group_TC_save_dir, f'group_emp_TC_{range_start}_{range_end}.csv')

    # load and return the existing group TC if the group TC was already calculated before
    if os.path.exists(group_TC_save_path):
        group_TC = pd.read_csv(group_TC_save_path, header=None).values
        group_TC = torch.as_tensor(group_TC).to(DEFAULT_DTYPE)
        return group_TC

    all_group_TCs = [
        get_HCPYA_emp_TC(sub_id)
        for sub_id in subject_ids[range_start:range_end]
    ]

    # average across all TC
    group_TC = torch.mean(torch.stack(all_group_TCs), dim=0)
    if save_result:
        pd.DataFrame(group_TC.cpu().numpy()).to_csv(group_TC_save_path,
                                                    index=False,
                                                    header=False)
        print(
            f'Group empirical time course for subjects {range_start} to {range_end} saved to {group_TC_save_path}'
        )
    return group_TC


############################################################
# Get labels
############################################################


def get_cortex_fs5_label():
    return pd.read_csv(os.path.join(MATLAB_SCRIPT_DIR, 'labels',
                                    'cortex_fs5_label.csv'),
                       header=None).values.flatten()
