"""
This file contains the functions for exporting intermediate files for following analyses.
"""
import os
import shutil
import torch
import numpy as np
import pandas as pd

from src.basic.constants import NUM_ROI
from src.utils.analysis_utils import get_run_path


def all_groups_EI_to_csv(ds_name,
                         num_groups,
                         target,
                         trial_idx,
                         seed_idx,
                         save_csv_path=None):
    # TODO: export along with losses figures as well
    ei_dir = get_run_path(ds_name, target, 'EI_ratio', trial_idx, seed_idx)
    if save_csv_path is None:
        save_csv_path = os.path.join(ei_dir, 'all_EI_ratios.csv')

    EI_matrix = np.zeros((NUM_ROI, num_groups))
    for i in range(num_groups):
        group_idx = i + 1
        EI_ratios = torch.load(os.path.join(ei_dir, f'group{group_idx}.pth'),
                               map_location='cpu')
        EI_ratios = torch.squeeze(EI_ratios['ei_ratio']).numpy()
        EI_matrix[:, i] = EI_ratios

    # save to csv
    df = pd.DataFrame(EI_matrix)

    print(f'Saving to {save_csv_path}...')
    df.to_csv(save_csv_path, index=False, header=False)

    print("Saved successfully.")


def get_seed_indices_with_lowest_loss(ds_name, target, trial_idx, seed_range,
                                      group_range):
    best_seed_indices = []

    for group_idx in group_range:
        best_seed_idx = 0
        lowest_loss = np.inf

        # get the seed idx with lowest loss
        for seed_idx in seed_range:
            test_dir = get_run_path(ds_name, target, 'test', trial_idx,
                                    seed_idx)
            test_param_dict = torch.load(os.path.join(test_dir,
                                                      f'group{group_idx}',
                                                      'val_results.pth'),
                                         map_location='cpu')
            loss = test_param_dict['val_total_loss'].item()
            if loss < lowest_loss:
                lowest_loss = loss
                best_seed_idx = seed_idx

        best_seed_indices.append(best_seed_idx)

    # save seed_indices_with_lowest_loss as csv
    df = pd.DataFrame(best_seed_indices)
    save_dir = get_run_path(ds_name, target, 'test', trial_idx,
                            '_best_among_all')
    save_csv_path = os.path.join(save_dir, 'best_seed_indices.csv')

    print(f'Saving to {save_csv_path}...')
    df.to_csv(save_csv_path, index=False, header=False)
    print("Saved successfully.")

    return best_seed_indices


def export_EI_from_param_with_lowest_loss_among_seeds(ds_name, target,
                                                      trial_idx, seed_range,
                                                      group_range):
    best_seed_indices = get_seed_indices_with_lowest_loss(
        ds_name, target, trial_idx, seed_range, group_range)
    save_EI_dir = get_run_path(ds_name, target, 'EI_ratio', trial_idx,
                               '_best_among_all')
    for i, best_seed_idx in enumerate(best_seed_indices):
        group_idx = group_range[i]
        # find the EI ratio file and save to the 'seed_best_among_all' directory
        ei_dir = get_run_path(ds_name, target, 'EI_ratio', trial_idx,
                              best_seed_idx)
        ei_file = os.path.join(ei_dir, f'group{group_idx}.pth')
        shutil.copy(ei_file, save_EI_dir)


def export_lowest_losses_among_seeds(ds_name, target, trial_idx, seed_range,
                                     group_range):
    """
    For the given trial, extract the lowest losses among all seeds for each group,
    and stack the losses horizontally (column wise) for each type of loss

    The file stored for each trial, seed, and group in the test directory is a dictionary
    containing keys ['parameter', 'val_total_loss', 'sorted_index', 'corr_loss', 'l1_loss', 'ks_loss'].
    Hence, in other words, we want to concatenate the values of the keys 'val_total_loss', 'corr_loss', 'l1_loss', 'ks_loss'
    """
    best_seed_indices = get_seed_indices_with_lowest_loss(
        ds_name, target, trial_idx, seed_range, group_range)
    save_dir = get_run_path(ds_name, target, 'test', trial_idx,
                            '_best_among_all')
    lowest_losses_among_seeds = torch.zeros((len(group_range), 4))
    for i, best_seed_idx in enumerate(best_seed_indices):
        group_idx = group_range[i]
        test_dir = get_run_path(ds_name, target, 'test', trial_idx,
                                best_seed_idx)
        saved_dict = torch.load(os.path.join(test_dir, f'group{group_idx}',
                                             'val_results.pth'),
                                map_location='cpu')
        lowest_losses_among_seeds[i] = torch.tensor([
            saved_dict['val_total_loss'], saved_dict['corr_loss'],
            saved_dict['l1_loss'], saved_dict['ks_loss']
        ])

    # split seed_indices_with_lowest_loss into separate losses and form a dictionary and finally save as pth
    losses_dict = {
        'total_loss': lowest_losses_among_seeds[:, 0],
        'corr_loss': lowest_losses_among_seeds[:, 1],
        'l1_loss': lowest_losses_among_seeds[:, 2],
        'ks_loss': lowest_losses_among_seeds[:, 3]
    }
    torch.save(losses_dict, os.path.join(save_dir, 'lowest_losses.pth'))
