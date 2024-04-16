import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import usage_functions
import sys

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
from tzeng_func_torch import parameterize_myelin_rsfc
import tzeng_func
import CBIG_func

# Note for roi_fc_to_network_fc type functions, please see Matlab HCP_Dev/scripts/data_process/HCD_roi2network_fc.m. It is finetuned by Xiaoxuan. The functions presented here assign ROI to network just by overlapping.


def EI_ratio_group_parameter(group_nbr, trial_nbr):
    dataset_name = 'HCP_Dev'

    group_nbr = int(group_nbr)

    parent_dir = '/home/tzeng/storage/Python/MFMApplication/HCD/group'
    param_path = os.path.join(
        parent_dir, f'test/trial{trial_nbr}/group{group_nbr}/val_results.pth')
    if not os.path.exists(param_path):
        print("This group doesn't have validation results.")
        return 1

    best_param_10 = torch.load(param_path)['param_10']

    group_mats = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    myelin = torch.as_tensor(group_mats['myelin_group_1029'])
    rsfc_gradient = torch.as_tensor(group_mats['rsfc_group_1029'])
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration
    EI_save_path = os.path.join(
        parent_dir, f'EI_ratio/trial{trial_nbr}/group{group_nbr}.pth')

    d = torch.load(EI_save_path)
    param_10 = d['param_10']
    parameter = parameterize_myelin_rsfc(myelin, rsfc_gradient, param_10)
    wee = parameter[0:68]
    wei = parameter[68:136]
    # print(f'{torch.mean(wei).item():.2f}', end=' ')
    print(f'{torch.mean(wee / wei).item():.2f}', end=' ')
    return 0


def map_desikan_to_yeo7():
    map_dir = '/home/shaoshi.z/storage/MFM/desikan_map/network'
    networks = os.listdir(map_dir)
    mapping_dict = {}
    for network in networks:
        key_network = f'network{network[0]}'
        rois = os.listdir(os.path.join(map_dir, network))
        roi_nbrs = np.array([int(rois[i][3:5]) for i in range(len(rois))])
        for i in range(len(roi_nbrs)):
            if roi_nbrs[i] in [1, 5, 37, 41]:
                roi_nbrs[i] = 0
            elif roi_nbrs[i] > 41:
                roi_nbrs[i] -= 4
            elif roi_nbrs[i] > 37:
                roi_nbrs[i] -= 3
            elif roi_nbrs[i] > 5:
                roi_nbrs[i] -= 2
            elif roi_nbrs[i] > 1:
                roi_nbrs[i] -= 1
        mapping_dict[key_network] = roi_nbrs[roi_nbrs != 0]
    print(mapping_dict)
    sio.savemat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/yeo7_to_desikan68.mat',
        mapping_dict)
    print('Saved.')


@DeprecationWarning
def roi_fc_to_network_fc():
    # Use Matlab script instead. This is by overlapping
    network_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yeo17_fslr32.mat')
    network_dlabel = network_dlabel['dlabel']
    roi_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
    roi_dlabel = roi_dlabel['dlabel']

    sublist = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_sublist_all.txt',
        header=None,
        index_col=False)
    sublist = np.squeeze(np.array(sublist))

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo7_rfMRI_ICAFIX'
    save_roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_roi2network'
    usage_functions.roi_fc_2_network_fc(network_dlabel, roi_dlabel, sublist,
                                        roi_fc_dir, save_roi2network_dir)


def roi_fc_to_network_fc_desikan():
    sublist = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_sublist_all.txt',
        header=None,
        index_col=False)
    sublist = np.squeeze(np.array(sublist))

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Desikan_rfMRI_fslr_aCompCor_bp'
    save_roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Desikan_Yeo7_roi2network_fslr_aCompCor_bp'
    usage_functions.roi_fc_2_network_fc_desikan(sublist, roi_fc_dir,
                                                save_roi2network_dir)


def roi_fc_to_network_fc_group_desikan():

    network_num = 7
    roi_num = 68

    group_num = 29

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group'
    save_roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/tmp/Desikan_Yeo7_roi2network'
    save_reorganize_roi_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo7_reorganize'

    mapping_dict = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/yeo7_to_desikan68.mat')
    rois = np.zeros((roi_num, ), dtype=np.int32)
    network_len_list = [0]
    count = 0
    for network in range(1, network_num + 1):
        key_network = f'network{network}'
        roi_nbrs = np.squeeze(mapping_dict[key_network])
        rois[count:count + len(roi_nbrs)] = roi_nbrs
        count += len(roi_nbrs)
        network_len_list.append(count)
    rois = rois - 1  # To form python index

    for group_nbr in range(1, group_num + 1):

        roi_fc_path = os.path.join(roi_fc_dir,
                                   f'{group_nbr}/train/fc_fcd_cdf.mat')
        if not os.path.exists(roi_fc_path):
            continue
        roi_fc = sio.loadmat(roi_fc_path)
        roi_fc = roi_fc['fc']  # [68, 68]

        fc = roi_fc

        reorganize_fc = fc[rois, :]
        reorganize_fc = reorganize_fc[:, rois]
        sio.savemat(
            os.path.join(save_reorganize_roi_dir, f'group{group_nbr}.mat'),
            {'fc': reorganize_fc})
        '''network_fc = np.zeros((network_num, network_num))
        for i in range(len(network_len_list) - 1):
            for j in range(len(network_len_list) - 1):
                network_fc[i, j] = np.mean(reorganize_fc[network_len_list[i]:network_len_list[i+1], network_len_list[j]:network_len_list[j+1]])
        sio.savemat(os.path.join(save_roi2network_dir, f'group{group_nbr}.mat'), {'fc': network_fc})'''


@DeprecationWarning
def roi_fc_to_network_fc_group_Yan():
    # Still by overlapping
    network_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yeo17_fslr32.mat')
    network_dlabel = network_dlabel['dlabel']
    roi_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
    roi_dlabel = roi_dlabel['dlabel']

    save_reorganize_roi_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_reorganize_age_group'

    network_num = np.amax(network_dlabel)
    roi_num = np.amax(roi_dlabel)
    _, mapping_dict = tzeng_func.tzeng_map_network_roi(network_dlabel,
                                                       roi_dlabel)
    rois = np.zeros((roi_num, ), dtype=np.int32)
    network_len_list = [0]
    count = 0
    for network in range(1, network_num + 1):
        key_network = f'network{network}'
        roi_nbrs = np.squeeze(mapping_dict[key_network])
        rois[count:count + len(roi_nbrs)] = roi_nbrs
        count += len(roi_nbrs)
        network_len_list.append(count)
    rois = rois - 1  # To form python index

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo7_rfMRI_ICAFIX'
    for group_nbr in range(2, 30):
        sub_demo = pd.read_csv(
            f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/subjects_demo.txt',
            sep='\t',
            header=None,
            index_col=False)
        fc_cat = None
        for i in range(len(sub_demo)):
            subject_id = sub_demo.iloc[i, 0]
            roi_fc_path = os.path.join(roi_fc_dir, f'{subject_id}.mat')
            if not os.path.exists(roi_fc_path):
                continue
            roi_fc = sio.loadmat(roi_fc_path)
            roi_fc = roi_fc['fc']  # [runs, rois, rois]
            if fc_cat is None:
                fc_cat = roi_fc
            else:
                fc_cat = np.concatenate((fc_cat, roi_fc), axis=0)

        fc = np.mean(fc_cat, axis=0)
        reorganize_fc = fc[rois, :]
        reorganize_fc = reorganize_fc[:, rois]
        sio.savemat(
            os.path.join(save_reorganize_roi_dir, f'group{group_nbr}.mat'),
            {'fc': reorganize_fc})


def check_within_between_fc():
    roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan100_Yeo17_roi2network_MNI_wbs_bp'
    id_age_file = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)
    save_fig_dir = '/home/ftian/storage/projects/MFM_exploration/reports/figures/HCD'
    fig_postfix = '_MNI_wbs_bp_Yan100_Yeo17'

    usage_functions.check_within_between_fc(id_age_file, roi2network_dir,
                                            save_fig_dir, fig_postfix)


def check_within_between_fc_group_desikan():
    group_num = 29

    roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/tmp/Desikan_Yeo7_roi2network'
    save_fig_dir = '/home/ftian/storage/projects/MFM_exploration/reports/figures/HCD'
    fig_postfix = '_group_MR_Desikan_Yeo7'

    within_fc_list = []
    between_fc_list = []
    whole_fc_list = []
    age_list = []

    within_mask = None
    for group_nbr in range(1, group_num + 1):

        fc_path = os.path.join(roi2network_dir, f'group{group_nbr}.mat')
        if not os.path.exists(fc_path):
            continue
        id_age_file = pd.read_csv(
            f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/train/train_subjects_demo.txt',
            sep='\t',
            header=None,
            index_col=False)
        age = np.mean(id_age_file.iloc[:, 1])
        age_list.append(age)
        fc = sio.loadmat(fc_path)
        fc = fc['fc']  # [7, 7] average across runs
        if within_mask is None:
            within_mask = np.eye(fc.shape[0], dtype=bool)
        within_fc = np.mean(fc[within_mask])
        between_fc = np.mean(fc[~within_mask])
        within_fc_list.append(within_fc)
        between_fc_list.append(between_fc)
        whole_fc_list.append(np.mean(fc))
    tzeng_func.tzeng_scatter_with_regress_line(
        age_list, within_fc_list,
        os.path.join(save_fig_dir, f'within_fc{fig_postfix}.png'))
    tzeng_func.tzeng_scatter_with_regress_line(
        age_list, between_fc_list,
        os.path.join(save_fig_dir, f'between_fc{fig_postfix}.png'))
    tzeng_func.tzeng_scatter_with_regress_line(
        age_list, whole_fc_list,
        os.path.join(save_fig_dir, f'whole_fc{fig_postfix}.png'))


def age_match_fc_with_PNC():
    id_age_PNC = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    id_age_HCD = pd.read_csv(
        '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/HCD_demo_valid_aftMR.txt',
        sep='\t',
        header=0,
        index_col=False)

    count_pnc = 0
    count_hcd = 0
    count_pnc_list = []
    count_hcd_list = []

    start_age = 98
    end_age = 264
    for age in range(start_age, end_age, 6):
        # PNC
        for i_pnc in range(count_pnc, len(id_age_PNC)):
            if id_age_PNC.iloc[i_pnc, 1] < age:
                continue
            elif id_age_PNC.iloc[i_pnc, 1] >= age:
                count_pnc = i_pnc
                count_pnc_list.append(count_pnc)
                break
        # HCD
        for i_hcd in range(count_hcd, len(id_age_HCD)):
            if id_age_HCD.iloc[i_hcd, 1] < age:
                continue
            elif id_age_HCD.iloc[i_hcd, 1] >= age:
                count_hcd = i_hcd
                count_hcd_list.append(count_hcd)
                break
    print("Count PNC list: ", count_pnc_list)
    print("Count HCD list: ", count_hcd_list)

    fc_pnc_list = []
    age_pnc_list = []
    for i in range(len(count_pnc_list) - 1):
        start_ind = count_pnc_list[i]
        end_ind = count_pnc_list[i + 1]
        fc_group = np.zeros((end_ind - start_ind, 68, 68))
        sub_count = 0
        for sub_i in range(start_ind, end_ind):
            subject_id = id_age_PNC.iloc[sub_i, 0]
            fc_sub = pd.read_csv(
                f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv',
                header=None,
                index_col=False)
            fc_sub = np.array(fc_sub)
            fc_group[sub_count] = fc_sub
            sub_count += 1
        fc_group = np.mean(fc_group, axis=0)
        fc_pnc_list.append(fc_group)
        age_pnc_list.append(np.mean(id_age_PNC.iloc[start_ind:end_ind, 1]))

    fc_hcd_list = []
    age_hcd_list = []
    for i in range(len(count_hcd_list) - 1):
        start_ind = count_hcd_list[i]
        end_ind = count_hcd_list[i + 1]
        fc_group = np.zeros((end_ind - start_ind, 68, 68))
        sub_count = 0
        for sub_i in range(start_ind, end_ind):
            subject_id = id_age_HCD.iloc[sub_i, 0]
            fc_sub = sio.loadmat(
                f'/mnt/isilon/CSC1/Yeolab/Users/tzeng/Matlab/HCP_Dev/matfiles/FC_FCD_rest/{subject_id}_V1_MR.mat'
            )
            fc_sub = np.mean(fc_sub['fc'], axis=0)
            fc_group[sub_count] = fc_sub
            sub_count += 1
        fc_group = np.nanmean(fc_group, axis=0)
        fc_hcd_list.append(fc_group)
        age_hcd_list.append(np.mean(id_age_HCD.iloc[start_ind:end_ind, 1]))

    print("Age PNC list: ", age_pnc_list)
    print("Age HCD list: ", age_hcd_list)
    fc_pnc_list = np.array(fc_pnc_list)
    fc_hcd_list = np.array(fc_hcd_list)
    np.save(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_pnc.npy',
        fc_pnc_list)
    np.save(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_hcd.npy',
        fc_hcd_list)
    print("Done.")
    # fc_PNC = pd.read_csv(f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv', header=None, index_col=False)


def plot_age_match_fc():
    fc_pnc = np.load(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_pnc.npy'
    )
    fc_hcd = np.load(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_hcd.npy'
    )
    print("Shape: ", fc_pnc.shape, fc_hcd.shape)
    fc_mask = np.triu(np.ones((68, 68), dtype=bool), 1)
    fc_pnc = fc_pnc[:, fc_mask]
    fc_hcd = fc_hcd[:, fc_mask]
    fc_corr = CBIG_func.CBIG_corr(fc_pnc.T, fc_hcd.T)
    print(fc_corr[np.eye(27, dtype=bool)])
    plt.figure()
    plt.imshow(fc_corr)
    plt.colorbar()
    plt.xlabel('HCD')
    plt.ylabel('PNC')
    plt.savefig(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_corr.png'
    )
    plt.close()


@DeprecationWarning
def plot_reorganize_fc():
    network_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yeo17_fslr32.mat')
    network_dlabel = network_dlabel['dlabel']
    roi_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
    roi_dlabel = roi_dlabel['dlabel']

    network_num = np.amax(network_dlabel)
    roi_num = np.amax(roi_dlabel)
    _, mapping_dict = tzeng_func.tzeng_map_network_roi(network_dlabel,
                                                       roi_dlabel)

    network_len_list = [0]
    count = 0
    for network in range(1, network_num + 1):
        key_network = f'network{network}'
        roi_nbrs = np.squeeze(mapping_dict[key_network])
        count += len(roi_nbrs)
        network_len_list.append(count)
    network_len_list = np.array(network_len_list)

    fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_reorganize_age_group'
    figure_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_reorganize_figures'

    fc_diff = []
    for group_nbr in range(1, 25):
        fc_path_1 = os.path.join(fc_dir, f'group{group_nbr}.mat')
        fc_path_2 = os.path.join(fc_dir, f'group{group_nbr + 5}.mat')
        if not os.path.exists(fc_path_1) or not os.path.exists(fc_path_2):
            continue
        fc_1 = sio.loadmat(fc_path_1)
        fc_1 = fc_1['fc']
        fc_2 = sio.loadmat(fc_path_2)
        fc_2 = fc_2['fc']
        fc_diff.append(fc_2 - fc_1)
    fc_diff = np.array(fc_diff)
    fc_diff = np.mean(fc_diff, axis=0)

    network_fc = np.zeros((network_num, network_num))
    for i in range(len(network_len_list) - 1):
        for j in range(len(network_len_list) - 1):
            network_fc[i, j] = np.mean(
                fc_diff[network_len_list[i]:network_len_list[i + 1],
                        network_len_list[j]:network_len_list[j + 1]])
    # sio.savemat(os.path.join('/home/tzeng/storage/Python/MFMApplication/usage_files/files/network17.mat'), {'fc': network_fc})

    d = sio.loadmat(
        '/home/tzeng/storage/CBIG_private/utilities/matlab/figure_utilities/corr_mat_colorscale.mat'
    )
    cmap = d['rbmap2']
    cmap = np.hstack((cmap, np.ones((cmap.shape[0], 1))))
    cmap = ListedColormap(cmap)

    plt.figure()
    plt.imshow(network_fc, cmap=cmap)
    plt.colorbar()
    plt.title('Average FC difference - 17 networks')
    plt.xlabel('Networks')
    plt.ylabel('Networks')
    plt.savefig(
        '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_reorganize_figures/fc_diff_network.png'
    )
    plt.close()


def roi_fc_vs_age():
    # every entry of FC (ROIs x ROIs) vs age
    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan100_Yeo17_rfMRI_MNI_wbs_bp'
    id_age_file = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)

    roi_num = 100
    save_postfix = '_Yan100_Yeo17_MNI_wbs_bp'

    age_list = []
    fc_all_subjects = np.zeros((len(id_age_file), roi_num, roi_num))
    count_sub = 0
    for sub_nbr in range(len(id_age_file)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = id_age_file.iloc[sub_nbr, 0]

        fc_path = os.path.join(roi_fc_dir, f'{subject_id}.mat')
        if not os.path.exists(fc_path):
            continue

        age = id_age_file.iloc[sub_nbr, 1]
        age_list.append(age)

        fc = sio.loadmat(fc_path)
        fc = fc['fc']
        fc = np.nanmean(fc, axis=0)  # [ROIs, ROIs] average across runs

        fc_all_subjects[count_sub] = fc
        count_sub += 1

    fc_all_subjects = fc_all_subjects[:count_sub]
    age_list = np.array(age_list)[:, np.newaxis]

    sio.savemat(
        f"/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/ForPlot/fc_mean{save_postfix}.mat",
        {'fc': np.mean(fc_all_subjects, axis=0)})
    print("Saved mean.")

    fc_development = np.zeros((roi_num, roi_num))
    for i in range(roi_num):
        fc_line = fc_all_subjects[:, i, :]  # [sub_num, roi_num]
        corr_line = CBIG_func.CBIG_corr(fc_line, age_list)
        fc_development[i, :] = np.squeeze(corr_line)

    sio.savemat(
        f"/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/ForPlot/fc_development{save_postfix}.mat",
        {'fc': fc_development})
    print("Saved.")


def roi2network_fc_vs_age():
    roi2network_fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_roi2network_fslr_aCompCor_bp'
    id_age_file = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)

    network_num = 17
    save_postfix = '_Yan400_Yeo17_fslr_aCompCor_bp'

    age_list = []
    fc_all_subjects = np.zeros((len(id_age_file), network_num, network_num))
    count_sub = 0
    for sub_nbr in range(len(id_age_file)):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = id_age_file.iloc[sub_nbr, 0]

        fc_path = os.path.join(roi2network_fc_dir, f'{subject_id}.mat')
        if not os.path.exists(fc_path):
            continue

        age = id_age_file.iloc[sub_nbr, 1]
        age_list.append(age)

        fc = sio.loadmat(fc_path)
        fc = fc['fc']
        fc = np.nanmean(fc, axis=0)  # [ROIs, ROIs] average across runs

        fc_all_subjects[count_sub] = fc
        count_sub += 1

    fc_all_subjects = fc_all_subjects[:count_sub]
    age_list = np.array(age_list)[:, np.newaxis]

    # sio.savemat(f"/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/roi2network/fc_mean{save_postfix}.mat", {'fc': np.mean(fc_all_subjects, axis=0)})
    # print("Saved mean.")

    fc_development = np.zeros((network_num, network_num))
    for i in range(network_num):
        fc_line = fc_all_subjects[:, i, :]  # [sub_num, roi_num]
        corr_line = CBIG_func.CBIG_corr(fc_line, age_list)
        fc_development[i, :] = np.squeeze(corr_line)

    sio.savemat(
        f"/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/roi2network/develop_mats/fc_development{save_postfix}.mat",
        {'fc': fc_development})
    print("Saved.")


def compare_pipeline():
    parent_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Analysis/roi_develop2network'
    fc_develop_1 = sio.loadmat(
        os.path.join(parent_dir, 'develop_mats', 'fslr_ICAFIX.mat'))
    fc_develop_1 = fc_develop_1['fc']
    fc_develop_2 = sio.loadmat(
        os.path.join(parent_dir, 'develop_mats', 'fslr_aCompCor_bp.mat'))
    fc_develop_2 = fc_develop_2['fc']

    network_num = 17

    within_mask = np.eye(network_num, dtype=bool)
    between_mask = np.triu(np.ones(network_num, dtype=bool), 1)
    whole_mask = np.triu(np.ones(network_num, dtype=bool))

    postfix_1 = 'ICAFIX'
    postfix_2 = 'aCompCor'

    save_fig_path = os.path.join(parent_dir, 'within_plots',
                                 f'within_{postfix_1}_{postfix_2}.png')
    usage_functions.plot_equal_line(fc_develop_1[within_mask],
                                    fc_develop_2[within_mask],
                                    save_fig_path,
                                    fig_title='compare within',
                                    xlabel=f'{postfix_1}',
                                    ylabel=f'{postfix_2}')

    save_fig_path = os.path.join(parent_dir, 'between_plots',
                                 f'between_{postfix_1}_{postfix_2}.png')
    usage_functions.plot_equal_line(fc_develop_1[between_mask],
                                    fc_develop_2[between_mask],
                                    save_fig_path,
                                    fig_title='compare between',
                                    xlabel=f'{postfix_1}',
                                    ylabel=f'{postfix_2}')


def param_vs_age():
    split_nbr = 1
    trial_nbr = 21
    test_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/test/trial{trial_nbr}/split{split_nbr}'
    age_list = []
    wee_list = []
    wei_list = []
    for group_nbr in range(1, 30):
        test_res_path = os.path.join(test_dir, f'group{group_nbr}',
                                     'val_results.pth')
        if not os.path.exists(test_res_path):
            continue
        test_res = torch.load(test_res_path)
        wee = test_res['parameter'][0:68, 0].numpy()
        wei = test_res['parameter'][68:136, 0].numpy()

        age_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/split{split_nbr}/validation/validation_subjects_demo.txt'
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        age_list.append(np.mean(age))

        wee_list.append(np.mean(wee))
        wei_list.append(np.mean(wei))

    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCD/wee_vs_age_group_trial{trial_nbr}_split{split_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wee_list,
                                               save_fig_path,
                                               figure_title='w_EE vs Age',
                                               xlabel='Age',
                                               ylabel='w_EE')
    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCD/wei_vs_age_group_trial{trial_nbr}_split{split_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wei_list,
                                               save_fig_path,
                                               figure_title='w_EI vs Age',
                                               xlabel='Age',
                                               ylabel='w_EI')


def param_distribution():
    split_nbr = 1
    trial_nbr = 21
    train_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/train/trial{trial_nbr}/split{split_nbr}'
    test_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/test/trial{trial_nbr}/split{split_nbr}'
    for group_nbr in range(1, 22):
        train_final_state = os.path.join(train_dir, f'group{group_nbr}',
                                         'final_state.pth')
        test_res_path = os.path.join(test_dir, f'group{group_nbr}',
                                     'val_results.pth')
        if not os.path.exists(test_res_path) or not os.path.exists(
                train_final_state):
            continue
        train_res = torch.load(train_final_state)
        test_res = torch.load(test_res_path)
        wei = test_res['parameter'][68:136, 0]
        wei_search_range = train_res['wEI_search_range']
        record = torch.hstack((wei.unsqueeze(1), wei_search_range)).numpy()
        record = pd.DataFrame(record)
        record.to_csv(
            f'/home/tzeng/storage/Python/MFMApplication/usage_files/files/HCD/param_distribution/{group_nbr}.txt',
            sep='\t',
            header=False,
            index=False)


def mean_fc_profile_vs_age_group():
    split_nbr = 1
    n_group = 21
    fc_list = np.zeros((n_group, 68))
    age_list = np.zeros((n_group, ))
    valid_count = 0
    for group_nbr in range(1, n_group + 1):
        group_mats_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/age_group_Harvard/{group_nbr}'
        fc_fcd_file = sio.loadmat(
            os.path.join(group_mats_path, 'fc_fcd_cdf_aCompCor.mat'))
        emp_fc = np.array(fc_fcd_file['fc'])
        fc_list[valid_count] = np.mean(emp_fc, axis=1)

        age_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/age_group_Harvard/{group_nbr}/subjects_demo.txt'
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        age_list[valid_count] = np.mean(age)

        valid_count += 1
    '''print(np.mean(fc_list, axis=0))
    fc_list = pd.DataFrame(fc_list)
    fc_list.to_csv('/home/tzeng/storage/Python/MFMApplication/usage_files/files/HCD/mean_fc_profiles_groups.txt', sep='\t', header=False, index=False)'''

    corr, p_value = CBIG_func.CBIG_corr(fc_list,
                                        age_list.astype(np.float64).reshape(
                                            (-1, 1)),
                                        need_pvalue=True)
    df = pd.DataFrame(corr, columns=['corr'])
    df['p_value'] = p_value
    df.to_csv(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/HCD/Harvard_mean_fc_profiles_vs_age_group.txt',
        sep='\t',
        header=False,
        index=False)


def mean_fc_profile_vs_age_individual():
    id_age_file = pd.read_csv(
        '/home/tzeng/storage/Matlab/HCP_Dev/txtfiles/HCD_Harvard_demo.txt',
        sep='\t',
        header=0,
        index_col=False)
    fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Desikan_rfMRI_fslr_aCompCor_bp'

    n_sub = len(id_age_file)
    fc_list = np.zeros((n_sub, 68))
    age_list = np.zeros((n_sub, ))
    valid_count = 0

    for sub_nbr in range(n_sub):
        if sub_nbr % 50 == 0:
            print(sub_nbr)
        subject_id = id_age_file.iloc[sub_nbr, 0]

        fc_path = os.path.join(fc_dir, f'{subject_id}.mat')
        if not os.path.exists(fc_path):
            continue
        fc_sub = sio.loadmat(fc_path)
        fc_sub = np.mean(fc_sub['fc'], axis=0)
        fc_list[valid_count] = np.mean(fc_sub, axis=1)
        age_list[valid_count] = id_age_file.iloc[sub_nbr, 1]
        valid_count += 1

    fc_list = fc_list[:valid_count]
    age_list = age_list[:valid_count]
    corr, p_value = CBIG_func.CBIG_corr(fc_list,
                                        age_list.astype(np.float64).reshape(
                                            (-1, 1)),
                                        need_pvalue=True)
    df = pd.DataFrame(corr, columns=['corr'])
    df['p_value'] = p_value
    df.to_csv(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/HCD/Harvard_mean_fc_profiles_vs_age_indi.txt',
        sep='\t',
        index=False)


def site_information():
    ndar_info = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/HCPDevelopmentRec/ndar_subject01.txt',
        sep='\t',
        header=0,
        index_col=False)
    fc_dir = '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Yan400_Yeo17_rfMRI_fslr_aCompCor_bp'
    n_sub = 652
    n_roi = 400
    site_name = 'WashU'
    # 'Harvard', 'UMinn', 'UCLA', 'WashU'
    fc_all_subjects = np.zeros((n_sub, n_roi, n_roi))
    age_all_subjects = np.zeros((n_sub, 1))
    valid_count = 0
    for i in range(n_sub):
        site = ndar_info.loc[i + 1, 'site']
        if site != site_name:
            continue
        subject_id = ndar_info.loc[i + 1, 'src_subject_id']
        fc_path = os.path.join(fc_dir, f'{subject_id}_V1_MR.mat')
        if not os.path.exists(fc_path):
            print(subject_id)
            continue
        fc = sio.loadmat(fc_path)
        fc = fc['fc']
        age = ndar_info.loc[i + 1, 'interview_age']
        fc_all_subjects[valid_count] = np.nanmean(fc, axis=0)
        age_all_subjects[valid_count, 0] = age
        valid_count += 1
    fc_all_subjects = fc_all_subjects[:valid_count]
    age_all_subjects = age_all_subjects[:valid_count]
    print(valid_count)

    fc_mean = np.mean(fc_all_subjects, axis=0)

    fc_development = np.zeros((n_roi, n_roi))
    for i in range(n_roi):
        fc_line = fc_all_subjects[:, i, :]  # [sub_num, roi_num]
        corr_line = CBIG_func.CBIG_corr(fc_line, age_all_subjects)
        fc_development[i, :] = np.squeeze(corr_line)

    fc_development[np.isnan(fc_development)] = 0

    sio.savemat(
        f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Analysis/site_difference/Yan400_Yeo17_fc_mean_fslr_aCompCor_{site_name}.mat',
        {'fc': fc_mean})
    sio.savemat(
        f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/within_between_FC/Analysis/site_difference/Yan400_Yeo17_fc_dev_fslr_aCompCor_{site_name}.mat',
        {'fc': fc_development})


def sc_fc_corr():
    mask_68 = np.triu(np.ones((68, 68), dtype=bool), 1)
    group_mats = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    sc_mat = np.array(group_mats['sc_group_1029'])
    sc_mat = sc_mat[mask_68][:, np.newaxis]
    n_group = 21
    fc_mats = np.zeros((sc_mat.shape[0], n_group))
    for group_nbr in range(1, n_group + 1):
        # fc_mat = sio.loadmat(f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/age_group_Harvard/{group_nbr}/fc_fcd_cdf_aCompCor.mat')
        fc_mat = torch.load(
            f'/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/simulate/fc/trial7/split1/group{group_nbr}.pth'
        )
        fc_mat = fc_mat['fc'].numpy()
        fc_mats[:, group_nbr - 1] = fc_mat[mask_68]
    corr_mat = CBIG_func.CBIG_corr(sc_mat, fc_mats)
    print(corr_mat)


def sim_fc_emp_fc():
    mask_68 = np.triu(np.ones((68, 68), dtype=bool), 1)
    save_fig_dir = '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCD/FC_aComp_ICA'
    n_group = 11
    for group_nbr in range(1, n_group + 1):
        fc_1 = sio.loadmat(
            f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/split1/train/fc_fcd_cdf.mat'
        )
        fc_1 = fc_1['fc']
        fc_1 = fc_1[mask_68]
        fc_2 = sio.loadmat(
            f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/split1/train/fc_fcd_cdf_aCompCor.mat'
        )
        fc_2 = fc_2['fc']
        fc_2 = fc_2[mask_68]

        save_fig_path = os.path.join(save_fig_dir, f'{group_nbr}.png')
        corr = CBIG_func.CBIG_corr(fc_1[:, np.newaxis], fc_2[:, np.newaxis])
        usage_functions.plot_equal_line(fc_1,
                                        fc_2,
                                        save_fig_path,
                                        fig_title=f'corr={corr[0, 0]:.4f}',
                                        xlabel='ICA FC',
                                        ylabel='aCompCor FC')


if __name__ == "__main__":
    # param_vs_age()
    # param_distribution()
    # mean_fc_profile_vs_age_individual()
    sim_fc_emp_fc()
