import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import usage_functions
import sys

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
from tzeng_func_torch import parameterize_myelin_rsfc
import tzeng_func
import CBIG_func


def compare_ica_acompcor():
    ica_list = []
    acomp_list = []

    for group_nbr in range(1, 30):
        group_mats_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}'
        fc_fcd_acomp = sio.loadmat(
            os.path.join(group_mats_path, 'train', 'fc_fcd_cdf_aCompCor.mat'))
        fc_fcd_ica = sio.loadmat(
            os.path.join(group_mats_path, 'train', 'fc_fcd_cdf.mat'))
        fc_acomp = fc_fcd_acomp['fc']
        fc_ica = fc_fcd_ica['fc']
        ica_list.append(np.mean(fc_ica))
        acomp_list.append(np.mean(fc_acomp))
    print("ICA: ", np.mean(np.array(ica_list)))
    print("aComp: ", np.mean(np.array(acomp_list)))
    # ICA:  0.24217337223702787 aComp:  0.4973324396341084

    fc_dataframe = pd.DataFrame(ica_list, columns=['ICAFIX_FC'])
    fc_dataframe['aCompCor_FC'] = acomp_list
    fc_dataframe.to_csv(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_mean_ica_acomp.txt',
        sep='\t',
        index=False)
    print("Saved.")


@DeprecationWarning
def roi_fc_to_network_fc():
    network_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yeo17_fslr32.mat')
    network_dlabel = network_dlabel['dlabel']
    roi_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
    roi_dlabel = roi_dlabel['dlabel']

    sublist = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_sublist_all.txt',
        header=None,
        index_col=False)
    sublist = np.squeeze(np.array(sublist))

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/within_between_FC/Yan400_Yeo7_rfMRI_ICAFIX'
    save_roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/within_between_FC/Yan400_Yeo17_roi2network'
    usage_functions.roi_fc_2_network_fc(network_dlabel, roi_dlabel, sublist,
                                        roi_fc_dir, save_roi2network_dir)


def check_within_between_fc():
    roi2network_dir = '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/within_between_FC/Yan400_Yeo17_roi2network_fslr_aCompCor'
    id_age_file = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)
    save_fig_dir = '/home/ftian/storage/projects/MFM_exploration/reports/figures/HCA'
    fig_postfix = '_Yan400_Yeo17_fslr_aCompCor'

    usage_functions.check_within_between_fc(id_age_file, roi2network_dir,
                                            save_fig_dir, fig_postfix)


def roi_fc_vs_age():
    roi_fc_dir = '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/within_between_FC/Yan400_Yeo17_fslr_aCompCor'
    id_age_df = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)
    roi_num = 400

    save_dir = '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/within_between_FC/rois_mean_develop'
    save_postfix = '_Yan400_Yeo17_fslr_aCompCor'

    fc_mean, fc_develop = usage_functions.roi_fc_vs_age(
        roi_fc_dir, id_age_df, roi_num)

    sio.savemat(os.path.join(save_dir, f'fc_mean{save_postfix}.mat'),
                {'fc': fc_mean})
    sio.savemat(os.path.join(save_dir, f'fc_develop{save_postfix}.mat'),
                {'fc': fc_develop})
    print("Saved.")


def param_distribution():
    test_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/test/trial5'
    x_list = []
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
        x = np.ones_like(wee) * group_nbr

        wee_list.append(wee)
        wei_list.append(wei)
        x_list.append(x)

    wee_list = np.array(wee_list).reshape((-1, ))
    wei_list = np.array(wei_list).reshape((-1, ))
    x_list = np.array(x_list).reshape((-1, ))

    plt.figure()
    plt.scatter(x_list, wee_list)
    plt.xlabel('group nbr')
    plt.ylabel('param')
    plt.savefig(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCA_trial5_wee.png'
    )
    plt.close()

    plt.figure()
    plt.scatter(x_list, wei_list)
    plt.xlabel('group nbr')
    plt.ylabel('param')
    plt.savefig(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCA_trial5_wei.png'
    )
    plt.close()


def param_vs_age():
    trial_nbr = 3
    test_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/test/trial{trial_nbr}'
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

        age_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}/validation/validation_subjects_demo.txt'
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        age_list.append(np.mean(age))

        wee_list.append(np.mean(wee))
        wei_list.append(np.mean(wei))

    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCA/wee_vs_age_trial{trial_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wee_list,
                                               save_fig_path,
                                               figure_title='w_EE vs Age',
                                               xlabel='Age',
                                               ylabel='w_EE')
    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCA/wei_vs_age_trial{trial_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wei_list,
                                               save_fig_path,
                                               figure_title='w_EI vs Age',
                                               xlabel='Age',
                                               ylabel='w_EI')


if __name__ == "__main__":
    param_vs_age()
