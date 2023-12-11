import torch
import sys
import scipy.io as sio
import scipy.stats as stats
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
sys.path.append('/home/tzeng/storage/Python/MFMDeepLearning')
import analysis_functions


def plot_pred_loss(nbr=None):
    # sub_nbr = 5
    if nbr is None:
        nbr = 3
    target = 'group'
    prefix = 'group' if target == 'group' else 'sub'
    split_nbr = 1
    trial_nbr = 22
    epochs = 50
    param_save_dir = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/{target}/train/trial{trial_nbr}/split{split_nbr}/{prefix}{nbr}'
    figure_save_dir = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/{target}/figures/pred_loss/trial{trial_nbr}'
    if not os.path.exists(figure_save_dir):
        os.makedirs(figure_save_dir)
    figure_save_path = os.path.join(figure_save_dir, f'split{split_nbr}_{prefix}{nbr}.png')
    analysis_functions.plot_pred_loss(epochs=epochs, param_save_dir=param_save_dir, figure_path=figure_save_path)
    return


def check_dl_epochs():
    trial_nbr = 1
    nbr = 1
    parent_dir = '/home/tzeng/storage/Python/MFMApplication/HCDParams/group'
    train_results_dir = os.path.join(parent_dir, f'train/trial{trial_nbr}/group{nbr}')
    val_results_dir = os.path.join(parent_dir, f'val_train_param/trial{trial_nbr}/group{nbr}')
    save_dir = os.path.join(parent_dir, 'figures')
    epochs = 67
    distinct_name = f'trial{trial_nbr}_{nbr}'
    analysis_functions.check_dl_version_test_results_epochs(train_results_dir, val_results_dir, save_dir, epochs, distinct_name, need_plot_overlay_dl_euler=True, need_save_record=True, need_plot_ground_euler=False, ground_dir=None)
    return 0


def age_distribution():
    age_file = '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/postprocessing/HCD_demographics.txt'
    age_file = pd.read_csv(age_file, sep='\t', header=0, index_col=False)

    ages = age_file.iloc[:, 1]
    ages = np.squeeze(np.array(ages) / 12)
    min_age = np.amin(ages)
    max_age = np.amax(ages)
    plt.figure()
    plt.hist(ages, bins=np.arange(int(min_age), int(max_age) + 1, 1))
    plt.title('HCP-D age distribution')
    plt.xlabel('Age/year')
    plt.ylabel('number')
    plt.savefig('/home/tzeng/storage/Python/MFMApplication/figures/HCD/age/HCD_age_dist.png')
    plt.close()
    print("Done.")


def site_age_distribution():
    ndar_info = pd.read_csv('/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Development/HCPDevelopmentRec/ndar_subject01.txt', sep='\t', header=0, index_col=False)
    n_sub = 652
    site_list = ['Harvard', 'UMinn', 'UCLA', 'WashU']
    count = 0
    for site_name in site_list:
        count += 1
        age_list = []
        for i in range(n_sub):
            site = ndar_info.loc[i + 1, 'site']
            if site != site_name:
                continue
            age = ndar_info.loc[i + 1, 'interview_age']
            age_list.append(int(age) / 12)
        print("Subject number: ", len(age_list))
        age_list = np.array(age_list)
        min_age = np.amin(age_list)
        max_age = np.amax(age_list)
        
        plt.hist(age_list, bins=np.arange(8, int(max_age) + 1, 1))
        plt.title(f'Age distribution - {site_name}, {len(age_list)} subjects')
        plt.xlabel('Age/year')
        plt.ylabel('number')
        plt.savefig(f'/home/tzeng/storage/Python/MFMApplication/figures/HCD/age/{site_name}_age_dist.png')
        plt.close()
        print(f'{site_name} mean: ', np.mean(age_list))
        print(f'{site_name} std: ', np.std(age_list))
        print("Done.")


def plot_EI_ratio_group():
    split_nbr = 1
    trial_nbr = 7
    # trial_nbr_2 = 3
    EI_dir = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/group/EI_ratio/trial{trial_nbr}/split{split_nbr}'
    # EI_dir_2 = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/group/EI_ratio/trial{trial_nbr_2}/split{split_nbr}'
    save_fig_dir = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/group/figures/EI/trial{trial_nbr}'
    if not os.path.exists(save_fig_dir):
        os.makedirs(save_fig_dir)
    save_fig_path = os.path.join(save_fig_dir, f'split{split_nbr}_mean_development_fit.png')

    EI_ave_list = []
    age_list = []
    for group_nbr in range(1, 30):
        age_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/split{split_nbr}/validation/validation_subjects_demo.txt'
        EI_path = os.path.join(EI_dir, f'group{group_nbr}.pth')
        # EI_path_2 = os.path.join(EI_dir_2, f'group{group_nbr}.pth')
        if not os.path.exists(EI_path):
            continue
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        ei_tmp = torch.load(EI_path)
        # ei_tmp_2 = torch.load(EI_path_2)
        EI_ave_list.append(torch.mean(ei_tmp['ei_ratio']).item())
        age_list.append(np.mean(age))

    analysis_functions.plot_EI_ratio(EI_ave_list, age_list, save_fig_path)


def regional_EI():
    split_nbr = 1
    trial_nbr = 19
    EI_dir = f'/home/tzeng/storage/Python/MFMApplication/Params/HCDParams/group/EI_ratio/trial{trial_nbr}/split{split_nbr}'
    save_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/pMFM/EI_ratio/group_regional_age_trial{trial_nbr}.mat'

    EI_regional_list = []
    age_list = []
    n_group = 21
    for group_nbr in range(1, n_group + 1):
        age_path = f'/home/tzeng/storage/Matlab/HCP_Dev/matfiles/age_group/{group_nbr}/split{split_nbr}/validation/validation_subjects_demo.txt'
        EI_path = os.path.join(EI_dir, f'group{group_nbr}.pth')
        if not os.path.exists(EI_path):
            continue
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        ei_tmp = torch.load(EI_path)
        EI_regional_list.append(ei_tmp['ei_ratio'].squeeze().numpy())
        age_list.append(np.mean(age))
    
    EI_regional_list = np.array(EI_regional_list)
    age_list = np.array(age_list)
    reg = LinearRegression()
    reg.fit(age_list[:, np.newaxis], EI_regional_list)
    sio.savemat(save_path, {'EI_ROI': reg.coef_[:, np.newaxis]})
    print("Saved regional EI development.")
    return 0


if __name__ == "__main__":
    '''for i in [1, 4, 10, 16, 21]:
        plot_pred_loss(i)'''
    plot_pred_loss(1)
    # plot_EI_ratio_group()
    # regional_EI()
    # site_age_distribution()
