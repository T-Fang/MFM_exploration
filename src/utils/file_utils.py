import os
import torch
import scipy.io as sio

from src.basic.constants import LOG_DIR


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
