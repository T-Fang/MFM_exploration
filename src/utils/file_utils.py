import os
import torch

from src.basic.constants import LOG_PATH


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
    run_path = os.path.join(
        get_run_path(ds_name, target, mode, trial_idx, seed_idx),
        f'group{group_idx}')
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
    group_dir = get_group_path(ds_name, target, 'train', trial_idx, seed_idx,
                               group_idx)
    saved_dict = torch.load(os.path.join(group_dir,
                                         f'param_save_epoch{epoch_idx}.pth'),
                            map_location='cpu')

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
