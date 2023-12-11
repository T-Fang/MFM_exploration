import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
import tzeng_func
import CBIG_func


def roi_fc_2_network_fc(network_dlabel, roi_dlabel, sublist, roi_fc_dir, save_roi2network_dir):

    # If you need mapping_dict that is yeo7 to desikan, load it yourself in '/home/tzeng/storage/Matlab/Utils/general_mats/yeo7_to_desikan68.mat'. It is a dict that can be directly used here.
    
    network_num = np.amax(network_dlabel)
    roi_num = np.amax(roi_dlabel)
    _, mapping_dict = tzeng_func.tzeng_map_network_roi(network_dlabel, roi_dlabel)  # By overlapping. Deprecated in many case.
    rois = np.zeros((roi_num,), dtype=np.int32)
    network_len_list = [0]
    count = 0
    for network in range(1, network_num + 1):
        key_network = f'network{network}'
        roi_nbrs = np.squeeze(mapping_dict[key_network])
        rois[count:count+len(roi_nbrs)] = roi_nbrs
        count += len(roi_nbrs)
        network_len_list.append(count)
    rois = rois - 1  # To form python index

    for sub_nbr in range(0, len(sublist)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = sublist[sub_nbr]

        roi_fc_path = os.path.join(roi_fc_dir, f'{subject_id}.mat')
        if not os.path.exists(roi_fc_path):
            continue
        roi_fc = sio.loadmat(roi_fc_path)
        roi_fc = roi_fc['fc']  # [runs, 68, 68]

        fc = roi_fc

        reorganize_fc = fc[:, rois, :]
        reorganize_fc = reorganize_fc[:, :, rois]
        network_fc = np.zeros((fc.shape[0], network_num, network_num))
        for i in range(len(network_len_list) - 1):
            for j in range(len(network_len_list) - 1):
                network_fc[:, i, j] = np.mean(reorganize_fc[:, network_len_list[i]:network_len_list[i+1], network_len_list[j]:network_len_list[j+1]], axis=(1, 2))
        sio.savemat(os.path.join(save_roi2network_dir, f'{subject_id}.mat'), {'fc': network_fc})


def roi_fc_2_network_fc_desikan(sublist, roi_fc_dir, save_roi2network_dir):

    # If you need mapping_dict that is yeo7 to desikan, load it yourself in '/home/tzeng/storage/Matlab/Utils/general_mats/yeo7_to_desikan68.mat'. It is a dict that can be directly used here.
    
    network_num = 7
    roi_num = 68
    # _, mapping_dict = tzeng_func.tzeng_map_network_roi(network_dlabel, roi_dlabel)
    mapping_dict = sio.loadmat('/home/tzeng/storage/Matlab/Utils/general_mats/yeo7_to_desikan68.mat')
    rois = np.zeros((roi_num,), dtype=np.int32)
    network_len_list = [0]
    count = 0
    for network in range(1, network_num + 1):
        key_network = f'network{network}'
        roi_nbrs = np.squeeze(mapping_dict[key_network])
        rois[count:count+len(roi_nbrs)] = roi_nbrs
        count += len(roi_nbrs)
        network_len_list.append(count)
    rois = rois - 1  # To form python index

    for sub_nbr in range(0, len(sublist)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = sublist[sub_nbr]

        roi_fc_path = os.path.join(roi_fc_dir, f'{subject_id}.mat')
        if not os.path.exists(roi_fc_path):
            continue
        roi_fc = sio.loadmat(roi_fc_path)
        roi_fc = roi_fc['fc']  # [runs, 68, 68]

        fc = roi_fc

        reorganize_fc = fc[:, rois, :]
        reorganize_fc = reorganize_fc[:, :, rois]
        network_fc = np.zeros((fc.shape[0], network_num, network_num))
        for i in range(len(network_len_list) - 1):
            for j in range(len(network_len_list) - 1):
                network_fc[:, i, j] = np.mean(reorganize_fc[:, network_len_list[i]:network_len_list[i+1], network_len_list[j]:network_len_list[j+1]], axis=(1, 2))
        sio.savemat(os.path.join(save_roi2network_dir, f'{subject_id}.mat'), {'fc': network_fc})


def check_within_between_fc(id_age_file, roi2network_dir, save_fig_dir, fig_postfix):
    within_fc_list = []
    between_fc_list = []
    whole_fc_list = []
    age_list = []

    within_mask = None
    for sub_nbr in range(len(id_age_file)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = id_age_file.iloc[sub_nbr, 0]

        fc_path = os.path.join(roi2network_dir, f'{subject_id}.mat')
        if not os.path.exists(fc_path):
            continue
        
        age = id_age_file.iloc[sub_nbr, 1] / 12
        age_list.append(age)
        fc = sio.loadmat(fc_path)
        fc = fc['fc']
        fc = np.nanmean(fc, axis=0)  # [7, 7] average across runs
        if within_mask is None:
            within_mask = np.eye(fc.shape[0], dtype=bool)
        within_fc = np.mean(fc[within_mask])
        between_fc = np.mean(fc[~within_mask])
        within_fc_list.append(within_fc)
        between_fc_list.append(between_fc)
        whole_fc_list.append(np.mean(fc))
    tzeng_func.tzeng_scatter_with_regress_line(age_list, within_fc_list, os.path.join(save_fig_dir, f'within_fc{fig_postfix}.png'), 'Within-Network FC', 'Age/y', 'Ave FC value')
    tzeng_func.tzeng_scatter_with_regress_line(age_list, between_fc_list, os.path.join(save_fig_dir, f'between_fc{fig_postfix}.png'), 'Between-Network FC', 'Age/y', 'Ave FC value')
    tzeng_func.tzeng_scatter_with_regress_line(age_list, whole_fc_list, os.path.join(save_fig_dir, f'whole_fc{fig_postfix}.png'), 'Whole-brain FC', 'Age/y', 'Ave FC value')


def roi_fc_vs_age(roi_fc_dir, id_age_df, roi_num):
    # every entry of FC (ROIs x ROIs) vs age

    age_list = []
    fc_all_subjects = np.zeros((len(id_age_df), roi_num, roi_num))
    count_sub = 0
    for sub_nbr in range(len(id_age_df)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = id_age_df.iloc[sub_nbr, 0]

        fc_path = os.path.join(roi_fc_dir, f'{subject_id}.mat')
        if not os.path.exists(fc_path):
            continue
        
        age = id_age_df.iloc[sub_nbr, 1]
        age_list.append(age)

        fc = sio.loadmat(fc_path)
        fc = fc['fc']
        fc = np.nanmean(fc, axis=0)  # [ROIs, ROIs] average across runs

        fc_all_subjects[count_sub] = fc
        count_sub += 1
    
    fc_all_subjects = fc_all_subjects[:count_sub]
    age_list = np.array(age_list)[:, np.newaxis]

    fc_mean = np.mean(fc_all_subjects, axis=0)

    fc_development = np.zeros((roi_num, roi_num))
    for i in range(roi_num):
        fc_line = fc_all_subjects[:, i, :]  # [sub_num, roi_num]
        corr_line = CBIG_func.CBIG_corr(fc_line, age_list)
        fc_development[i, :] = np.squeeze(corr_line)
    
    fc_development[np.isnan(fc_development)] = 0
    
    return fc_mean, fc_development


def plot_equal_line(x_list, y_list, save_fig_path, fig_title='Compare x y', xlabel='x', ylabel='y'):
    x_list = np.squeeze(np.array(x_list))
    y_list = np.squeeze(np.array(y_list))

    plt.figure()
    max_value = max(np.amax(x_list), np.amax(y_list))
    min_value = min(np.amin(x_list), np.amin(y_list))
    value_range = max_value - min_value

    x_equal_line = np.arange(min_value - 0.1 * value_range, max_value + 0.1 * value_range, 0.01 * value_range)
    plt.plot(x_equal_line, x_equal_line)
    plt.scatter(x_list, y_list)
    plt.title(fig_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_fig_path)
    plt.close()
