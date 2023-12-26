import scipy.io as sio
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import sys

sys.path.append('/home/tzeng/storage/Python/MFMDeepLearning')
import analysis_functions

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
import tzeng_func


def thickness_trend():
    thicknesses = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/other/corrthickness_all_subjects.mat'
    )
    thicknesses = np.squeeze(thicknesses['thickness_subjects'])
    subject_list = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_sublist_all.txt',
        sep='\t',
        header=None,
        index_col=False)
    subject_demo = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)

    age_list = []
    for i in range(len(subject_list)):
        subject_id = subject_list.loc[i, 0]
        for j in range(len(subject_demo)):
            if subject_demo.loc[j, 'src_subject_id'] == subject_id[0:-6]:
                age_list.append(subject_demo.loc[j, 'interview_age'])
                break
    '''plt.figure()
    plt.scatter(age_list, thicknesses)
    plt.savefig('/home/tzeng/storage/Python/MFMApplication/usage_figures/corrthickness_age_HCD.png')
    plt.close()'''
    return age_list, thicknesses


def thickness_trend_HCP():
    thicknesses = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCP_Dev/matfiles/other/corrthickness_HCP.mat'
    )
    thicknesses = np.squeeze(thicknesses['thickness_subjects'])
    subject_list = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/subject_ids.mat')
    subject_list = subject_list['subject_ids']
    subject_demo = pd.read_csv(
        '/mnt/isilon/CSC1/Yeolab/Data/HCP/S1200/scripts/subject_measures/unrestricted_jingweili_12_7_2017_21_0_16_NEO_A_corrected.csv',
        sep=',',
        header=0,
        index_col=False)

    age_list = []
    thickness_list = []
    for i in range(len(subject_list)):
        subject_id = subject_list[i, 0]
        if thicknesses[i] == 0:
            continue
        thickness_list.append(thicknesses[i])
        for j in range(len(subject_demo)):
            if int(subject_demo.loc[j, 'Subject']) == subject_id:
                age_list.append(int(subject_demo.loc[j, 'Age'][0:2]))
                break
    '''plt.figure()
    plt.scatter(age_list, thickness_list)
    plt.savefig('/home/tzeng/storage/Python/MFMApplication/usage_figures/corrthickness_age_HCP.png')
    plt.close()'''
    return age_list, thickness_list


def thickness_trend_HCA():
    thicknesses = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCP_Aging/matfiles/other/corrthickness_all_subjects.mat'
    )
    thicknesses = np.squeeze(thicknesses['thickness_subjects'])
    subject_list = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_sublist_all.txt',
        sep='\t',
        header=None,
        index_col=False)
    subject_demo = pd.read_csv(
        '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_demographics.txt',
        sep='\t',
        header=0,
        index_col=False)

    age_list = []
    for i in range(len(subject_list)):
        subject_id = subject_list.loc[i, 0]
        for j in range(len(subject_demo)):
            if subject_demo.loc[j, 'src_subject_id'] == subject_id[0:-6]:
                age_list.append(subject_demo.loc[j, 'interview_age'])
                break
    '''plt.figure()
    plt.scatter(age_list, thicknesses)
    plt.savefig('/home/tzeng/storage/Python/MFMApplication/usage_figures/corrthickness_age_HCA.png')
    plt.close()'''
    return age_list, thicknesses


def cat_thickness_lifespan():
    age_HCD, thick_HCD = thickness_trend()
    age_HCP, thick_HCP = thickness_trend_HCP()
    age_HCA, thick_HCA = thickness_trend_HCA()

    age_HCD = np.array(age_HCD)
    thick_HCD = np.array(thick_HCD)
    age_HCP = (np.array(age_HCP) + 2) * 12
    thick_HCP = np.array(thick_HCP)
    age_HCA = np.array(age_HCA)
    thick_HCA = np.array(thick_HCA)

    age_all = np.concatenate((age_HCD, age_HCP, age_HCA), axis=0)
    thick_all = np.concatenate((thick_HCD, thick_HCP, thick_HCA), axis=0)
    plt.figure()
    plt.scatter(age_all, thick_all, s=10)
    plt.xlabel('age/month')
    plt.ylabel('corrthickness')
    plt.title('Cortical Thickness across HCP lifespan')
    plt.savefig(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/figures/thick_lifespan.png'
    )
    plt.close()


def fc_mean_datasets():
    # Mean FC values: PNC:  0.35208016838105693 HCD:  0.37484359063030814

    # For PNC
    '''pnc_fc = []
    pnc_age = []
    for group_nbr in range(1, 30):
        group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}'
        fc_emp = np.array(pd.read_csv(os.path.join(group_mats_path, f'FC_train.csv'), header=None, index_col=False))
        pnc_fc.append(np.mean(fc_emp))

        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        pnc_age.append(np.mean(age))

    save_fig_path = '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/PNC/fc_mean_group_development_fit.png'
    tzeng_func.tzeng_scatter_with_regress_line(pnc_age, pnc_fc, save_fig_path, figure_title='PNC mean FC vs age', xlabel='age', ylabel='FC')'''

    # For HCD
    hcd_fc = []
    hcd_age = []
    for group_nbr in range(1, 22):
        group_mats_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/age_group_Harvard/{group_nbr}'
        fc_fcd_file = sio.loadmat(
            os.path.join(group_mats_path, 'fc_fcd_cdf_aCompCor.mat'))
        fc_emp = np.array(fc_fcd_file['fc'])
        hcd_fc.append(np.mean(fc_emp))

        age_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/age_group_Harvard/{group_nbr}/subjects_demo.txt'
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        hcd_age.append(np.mean(age))

    save_fig_path = '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCD/aCompCor_Harvard_fc_mean_group_development_fit.png'
    tzeng_func.tzeng_scatter_with_regress_line(
        hcd_age,
        hcd_fc,
        save_fig_path,
        figure_title='HCP-D mean FC vs age',
        xlabel='age',
        ylabel='FC')

    # For HCA
    '''hca_fc = []
    hca_age = []
    for group_nbr in range(1, 30):
        group_mats_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}'
        fc_fcd_acomp = sio.loadmat(os.path.join(group_mats_path, 'train', 'fc_fcd_cdf.mat'))
        fc_acomp = fc_fcd_acomp['fc']
        hca_fc.append(np.mean(fc_acomp))

        age_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}/validation/validation_subjects_demo.txt'
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        hca_age.append(np.mean(age))
    
    save_fig_path = '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCA/ICAFIX_fc_mean_group_development_fit.png'
    tzeng_func.tzeng_scatter_with_regress_line(hca_age, hca_fc, save_fig_path, figure_title='HCP-A mean FC vs age', xlabel='age', ylabel='FC')'''

    # print("PNC: ", np.mean(np.array(pnc_fc)))
    print("HCD: ", np.mean(np.array(hcd_fc)))
    # print("HCA: ", np.mean(np.array(hca_fc)))
    '''
    PNC:  0.35208016838105693
    HCD:  0.37484359063030814 GSR: 0.09639266857954885
    HCA:  0.4973324396341084

    ICAFIX
    HCD:  0.28709723134887805
    HCA:  0.24217337223702787
    '''
    '''fc_dataframe = pd.DataFrame(pnc_fc, columns=['PNC_FC'])
    fc_dataframe['HCD_FC'] = hcd_fc
    fc_dataframe.to_csv('/home/tzeng/storage/Python/MFMApplication/usage_files/files/fc_mean_datasets.txt', sep='\t', index=False)
    print("Saved.")'''


def test_results_compare():
    # pnc_test_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group/test/trial4'

    test_dir_1 = '/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/test/trial13/split1'
    total_loss_1 = []
    corr_loss_1 = []
    l1_loss_1 = []
    ks_loss_1 = []
    for group_nbr in range(1, 30):
        test_res_path = os.path.join(test_dir_1, f'group{group_nbr}',
                                     'val_results.pth')
        if not os.path.exists(test_res_path):
            continue
        test_res = torch.load(test_res_path)
        total_loss_1.append(test_res['val_total_loss'].item())
        corr_loss_1.append(test_res['corr_loss'].item())
        l1_loss_1.append(test_res['l1_loss'].item())
        ks_loss_1.append(test_res['ks_loss'].item())

    test_dir_2 = '/home/ftian/storage/projects/MFM_exploration/logs/HCD/group/test/trial14/split1'
    total_loss_2 = []
    corr_loss_2 = []
    l1_loss_2 = []
    ks_loss_2 = []
    for group_nbr in range(1, 30):
        test_res_path = os.path.join(test_dir_2, f'group{group_nbr}',
                                     'val_results.pth')
        if not os.path.exists(test_res_path):
            continue
        test_res = torch.load(test_res_path)
        total_loss_2.append(test_res['val_total_loss'].item())
        corr_loss_2.append(test_res['corr_loss'].item())
        l1_loss_2.append(test_res['l1_loss'].item())
        ks_loss_2.append(test_res['ks_loss'].item())

    # print("PNC: ", np.mean(pnc_total_loss), np.mean(pnc_corr_loss))
    print("Test results 1: ", np.mean(total_loss_1), np.mean(corr_loss_1),
          np.mean(l1_loss_1), np.mean(ks_loss_1))
    print("Test results 2: ", np.mean(total_loss_2), np.mean(corr_loss_2),
          np.mean(l1_loss_2), np.mean(ks_loss_2))
    print(total_loss_1)
    print(total_loss_2)
    '''PNC:  0.5893890644127213 0.44811536732532303
HCD:  0.7954383642345592 0.6545771808308892
HCA:  0.5441086261797499 0.43440086874327527'''


if __name__ == "__main__":
    fc_mean_datasets()
