import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio

from src.basic.constants import DEFAULT_DTYPE, LOG_DIR, TMP_DIR
from src.utils.SC_utils import group_SC_matrices
from src.utils.FC_utils import fisher_average


def get_target_dir(ds_name, target):
    target_path = os.path.join(LOG_DIR, ds_name, target)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    return target_path


def get_run_dir(ds_name, target, mode, trial_idx, seed_idx):
    run_path = os.path.join(get_target_dir(ds_name, target), mode,
                            f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    return run_path


def get_group_dir(ds_name, target, mode, trial_idx, seed_idx, group_idx=None):
    run_dir = get_run_dir(ds_name, target, mode, trial_idx, seed_idx)
    if group_idx is None:
        return run_dir
    group_dir = os.path.join(run_dir, f'group{group_idx}')
    if not os.path.exists(group_dir):
        os.makedirs(group_dir)
    return group_dir


def get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx):
    fig_dir = os.path.join(get_target_dir(ds_name, target), 'figures',
                           fig_type, f'trial{trial_idx}', f'seed{seed_idx}')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    return fig_dir


def get_losses_fig_dir(ds_name, target):
    losses_fig_dir = os.path.join(get_target_dir(ds_name, target), 'figures',
                                  'losses')
    if not os.path.exists(losses_fig_dir):
        os.makedirs(losses_fig_dir)
    return losses_fig_dir


def get_fig_file_path(ds_name, target, fig_type, trial_idx, seed_idx,
                      fig_name):
    fig_dir = get_fig_dir(ds_name, target, fig_type, trial_idx, seed_idx)
    return os.path.join(fig_dir, fig_name)


def load_train_dict(ds_name, target, trial_idx, seed_idx, group_idx,
                    epoch_idx):
    """
    the saved pth file contains a dictionary with keys ['valid_param_list', 'FC_FCD_loss'] and possibly 'r_E_reg_loss', 'r_E_for_valid_params'
    The 'valid_param_list' is a mask of size (num_of_children, ) indicating which children of the current epoch are valid
    The 'FC_FCD_loss' is a tensor of size (num_of_children, 3) containing FC corr loss, FC l1 loss, and FCD KS loss for each child
    if 'r_E_for_valid_params' is present, it is a tensor of size (num_of_ROI, num_of_children) containing r_E for each child at each ROI at each epoch
    if 'r_E_reg_loss' is present, it is a tensor of size (num_of_children, ) containing r_E regularization loss for each child

    * Note that we will only return the dictionary after applying the mask valid_param_list
    """
    group_dir = get_group_dir(ds_name, target, 'train', trial_idx, seed_idx,
                              group_idx)
    # Check if the file exists
    dict_path = os.path.join(group_dir, f'param_save_epoch{epoch_idx}.pth')
    if not os.path.exists(dict_path):
        # TODO: Temporary fix for the case when the file does not exist
        while epoch_idx >= 0:
            epoch_idx -= 1
            dict_path = os.path.join(group_dir,
                                     f'param_save_epoch{epoch_idx}.pth')
            if os.path.exists(dict_path):
                break
        # return None

    saved_dict = torch.load(dict_path, map_location='cpu')

    # we only consider valid params using the valid_param_list mask
    valid_param_list = saved_dict['valid_param_list']
    saved_dict['FC_FCD_loss'] = saved_dict['FC_FCD_loss'][valid_param_list]
    if 'r_E_for_valid_params' in saved_dict.keys():
        saved_dict['r_E_for_valid_params'] = saved_dict[
            'r_E_for_valid_params'][:, valid_param_list]
    if 'r_E_reg_loss' in saved_dict.keys():
        saved_dict['r_E_reg_loss'] = saved_dict['r_E_reg_loss'][
            valid_param_list]
    return saved_dict


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


def get_group_stats(range_start: int, range_end: int):
    myelin = get_group_myelin(range_start, range_end)
    rsfc_gradient = get_group_rsfc_gradient(range_start, range_end)
    sc_mat = get_group_sc_mat(range_start, range_end)
    sc_euler = sc_mat / torch.max(sc_mat) * 0.02
    fc_emp = get_group_fc_emp(range_start, range_end)
    emp_fcd_cum = get_group_emp_fcd_cum(range_start, range_end)

    group_stats = {
        'myelin': myelin,
        'rsfc_gradient': rsfc_gradient,
        'sc_mat': sc_mat,
        'sc_euler': sc_euler,
        'fc_emp': fc_emp,
        'emp_fcd_cum': emp_fcd_cum
    }

    return group_stats


def get_group_myelin(range_start: int, range_end: int):
    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    myelin_indi = sio.loadmat(
        os.path.join(individual_mats_path, 'myelin_roi_1029.mat'))
    myelin_indi = myelin_indi['myelin_roi_1029']
    myelin = np.nanmean(myelin_indi[range_start:range_end], axis=0)
    myelin = torch.as_tensor(myelin).to(DEFAULT_DTYPE).unsqueeze(1)
    return myelin


def get_group_rsfc_gradient(range_start: int, range_end: int):
    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    rsfc_indi = sio.loadmat(
        os.path.join(individual_mats_path, 'rsfc_roi_1029.mat'))
    rsfc_indi = rsfc_indi['rsfc_roi_1029']
    rsfc_gradient = np.nanmean(rsfc_indi[range_start:range_end], axis=0)
    rsfc_gradient = torch.as_tensor(rsfc_gradient).to(DEFAULT_DTYPE).unsqueeze(
        1)
    return rsfc_gradient


def get_group_sc_mat(range_start: int, range_end: int):
    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    sc_indi = sio.loadmat(os.path.join(individual_mats_path,
                                       'sc_roi_1029.mat'))
    sc_indi = sc_indi['sc_roi_1029']
    sc_mat = group_SC_matrices(sc_indi[range_start:range_end])
    sc_mat = torch.as_tensor(sc_mat).to(DEFAULT_DTYPE)
    return sc_mat


def get_group_fc_emp(range_start: int, range_end: int):
    fc_1029 = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fc_roi_1029.mat'
    )
    fc_1029 = fc_1029['fc_roi_1029']
    fc_emp = np.array(fc_1029[range_start:range_end])
    fc_emp = fisher_average(fc_emp)
    fc_emp = torch.as_tensor(fc_emp).to(DEFAULT_DTYPE)
    return fc_emp


def get_group_emp_fcd_cum(range_start: int, range_end: int):
    fcd_1029 = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fcd_cum_1029.mat'
    )
    fcd_1029 = fcd_1029['fcd_cum_1029']
    emp_fcd_cum = torch.as_tensor(
        fcd_1029[range_start:range_end]).to(DEFAULT_DTYPE)
    emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
    emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)
    return emp_fcd_cum
