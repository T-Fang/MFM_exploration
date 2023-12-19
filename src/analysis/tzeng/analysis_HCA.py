import torch
import scipy.io as sio
# import scipy.stats as stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from src.analysis import analysis_functions
# from src.utils import CBIG_func


def plot_pred_loss(nbr=None):
    # sub_nbr = 5
    if nbr is None:
        nbr = 2
    target = 'group'
    prefix = 'group' if target == 'group' else 'sub'
    trial_nbr = 5
    param_save_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/{target}/train/trial{trial_nbr}/{prefix}{nbr}'
    figure_save_path = (
        f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/{target}/figures/'
        f'pred_loss/trial{trial_nbr}_{prefix}{nbr}.png')
    analysis_functions.plot_pred_loss(epochs=50,
                                      param_save_dir=param_save_dir,
                                      figure_path=figure_save_path)
    return


def check_dl_epochs():
    trial_nbr = 1
    nbr = 1
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group'
    train_results_dir = os.path.join(parent_dir,
                                     f'train/trial{trial_nbr}/group{nbr}')
    val_results_dir = os.path.join(
        parent_dir, f'val_train_param/trial{trial_nbr}/group{nbr}')
    save_dir = os.path.join(parent_dir, 'figures')
    epochs = 50
    distinct_name = f'trial{trial_nbr}_{nbr}'
    analysis_functions.check_dl_version_test_results_epochs(
        train_results_dir,
        val_results_dir,
        save_dir,
        epochs,
        distinct_name,
        need_plot_overlay_dl_euler=True,
        need_save_record=True,
        need_plot_ground_euler=False,
        ground_dir=None)
    return 0


def find_fail_trial():
    trial_nbr = 4
    train_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/train/trial{trial_nbr}'
    for nbr in range(1, 30):
        train_final_state = os.path.join(train_dir,
                                         f'group{nbr}/param_save_epoch0.pth')
        if not os.path.exists(train_final_state):
            print(f"'{nbr}'", end=' ')
    print('\nDone.')


def age_distribution():
    age_file = '/mnt/nas/CSC7/Yeolab/Data/HCP/HCP_Aging/postprocessing/HCA_demographics.txt'
    age_file = pd.read_csv(age_file, sep='\t', header=0, index_col=False)

    ages = age_file.iloc[:, 1]
    ages = np.squeeze(np.array(ages) / 12)
    min_age = np.amin(ages)
    max_age = np.amax(ages)
    plt.figure()
    plt.hist(ages, bins=np.arange(int(min_age), int(max_age) + 1, 1))
    plt.title('HCP-A age distribution')
    plt.xlabel('Age/year')
    plt.ylabel('number')
    plt.savefig(
        '/home/ftian/storage/projects/MFM_exploration/reports/figures/dataset_related/HCA_age_dist.png'
    )
    plt.close()
    print("Done.")


def plot_EI_ratio_group():
    trial_nbr = 5
    EI_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/EI_ratio/trial{trial_nbr}'
    save_fig_path = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/figures/EI/trial{trial_nbr}_development_fit.png'

    EI_ave_list = []
    age_list = []
    for group_nbr in range(1, 30):
        age_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}/validation/validation_subjects_demo.txt'
        EI_path = os.path.join(EI_dir, f'group{group_nbr}.pth')
        if not os.path.exists(EI_path):
            continue
        age = pd.read_csv(age_path, sep='\t', header=None, index_col=False)
        age = age.loc[:, 1]
        ei_tmp = torch.load(EI_path)
        EI_ave_list.append(torch.mean(ei_tmp['ei_ratio']).item())
        age_list.append(np.mean(age))

    analysis_functions.plot_EI_ratio(EI_ave_list, age_list, save_fig_path)


def regional_EI():
    trial_nbr = 5
    EI_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCAParams/group/EI_ratio/trial{trial_nbr}'
    save_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/pMFM/EI_ratio/group_regional_vs_age_trial{trial_nbr}.mat'

    EI_regional_list = []
    age_list = []
    for group_nbr in range(1, 30):
        age_path = f'/home/tzeng/storage/Matlab/HCP_Aging/matfiles/age_group/{group_nbr}/validation/validation_subjects_demo.txt'
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
    '''
    corr = CBIG_func.CBIG_corr(age_list[:, np.newaxis], EI_regional_list)
    print(corr)
    '''

    reg = LinearRegression()
    reg.fit(age_list[:, np.newaxis], EI_regional_list)
    sio.savemat(save_path, {'EI_ROI': reg.coef_[:, np.newaxis]})
    print("Saved regional EI development.")
    '''
    # Slope t-test
    slope_list = []
    pvalue_list = []
    for i in range(68):
        results = stats.linregress(age_list, EI_regional_list[:, i])
        slope_list.append(results.slope)
        pvalue_list.append(results.pvalue)
    print(slope_list)
    print(pvalue_list)
    '''

    return 0


if __name__ == "__main__":
    regional_EI()
