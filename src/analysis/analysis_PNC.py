import torch
import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '/home/ftian/storage/projects/MFM_exploration')
from src.analysis import analysis_functions
from src.utils.export_utils import (  # noqa
    all_groups_EI_to_csv, export_EI_from_param_with_lowest_loss_among_seeds,
    export_lowest_losses_among_seeds)
from src.utils import tzeng_func
from src.utils.analysis_utils import (  # noqa
    boxplot_network_stats, boxplot_train_r_E, get_run_path, get_fig_file_path,
    plot_losses_for_diff_trials, plot_train_loss, visualize_stats,
    ttest_1samp_n_plot, regional_EI_age_slope, regional_EI_diff_cohen_d,
    plot_losses_for_diff_trials_all_groups, visualize_train_r_E)
from src.basic.constants import NUM_GROUPS_PNC_AGE, NUM_GROUPS_PNC_COGNITION, NUM_ROI

NUM_GROUPS = {
    'age_group': NUM_GROUPS_PNC_AGE,
    'overall_acc_group/high': NUM_GROUPS_PNC_COGNITION,
    'overall_acc_group/low': NUM_GROUPS_PNC_COGNITION
}


def plot_pred_loss():
    param_save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/high/train/trial3/seed1/group14'
    figure_save_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/figures/pred_loss/t3se1g14.png'
    analysis_functions.plot_pred_loss(epochs=50,
                                      param_save_dir=param_save_dir,
                                      figure_path=figure_save_path)


def compare_train_loss(nbr):
    dir_1 = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/FinetuneData/test/trial1/sub{nbr}'
    dir_2 = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/FinetuneData/test/trial1/sub{nbr}'
    epoch_nbr_1 = 99
    epoch_nbr_2 = 119

    path_1 = os.path.join(dir_1, f'param_save_epoch{epoch_nbr_1}.pth')
    path_2 = os.path.join(dir_2, f'param_save_epoch{epoch_nbr_2}.pth')
    if not os.path.exists(path_1) or not os.path.exists(path_2):
        raise Exception("Check path first.")
    d_1 = torch.load(path_1, map_location='cpu')
    d_2 = torch.load(path_2, map_location='cpu')
    '''print("Corr loss: ", torch.mean(d_1['corr_loss']), torch.mean(d_2['corr_loss']))
    print("L1 loss: ", torch.mean(d_1['L1_loss']), torch.mean(d_2['L1_loss']))
    print("KS loss: ", torch.mean(d_1['ks_loss']), torch.mean(d_2['ks_loss']))'''

    mean_total_loss_1 = torch.mean(d_1['corr_loss'] + d_1['L1_loss'] +
                                   d_1['ks_loss'])
    mean_total_loss_2 = torch.mean(d_2['corr_loss'] + d_2['L1_loss'] +
                                   d_2['ks_loss'])
    print("Total loss: ", mean_total_loss_1, mean_total_loss_2)
    if mean_total_loss_1 < mean_total_loss_2:
        return 1
    return 0


def compare_val_loss():
    count = 0
    my_list = []
    shaoshi_list = []
    for group_nbr in range(1, NUM_GROUPS_PNC_AGE + 1):
        shaoshi = np.squeeze(
            np.array(
                pd.read_csv(
                    f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/test/{group_nbr}/test_all.csv',
                    header=None,
                    index_col=False)))
        shaoshi_loss = shaoshi[9]
        '''
        shaoshi = torch.load(
            f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/group/test_shaoshi/group{group_nbr}/val_results.pth'
        )
        shaoshi_loss = shaoshi['corr_loss'][0].item() + shaoshi['l1_loss'][0].item() + shaoshi['ks_loss'][0].item()
        '''
        shaoshi_list.append(shaoshi_loss)

        me = torch.load(
            f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/group/test_before/group{group_nbr}/val_results.pth',
            map_location='cpu')
        my_loss = me['corr_loss'][0].item() + me['l1_loss'][0].item(
        ) + me['ks_loss'][0].item()
        my_list.append(my_loss)
        if my_loss < shaoshi_loss:
            count += 1
        print(f'{group_nbr}: ', my_loss, shaoshi_loss)
    print(count)
    statistics, p_value = stats.ttest_ind(np.array(my_list),
                                          np.array(shaoshi_list))
    print(f"T-test results: statistics: {statistics}; p-value: {p_value}")
    need_boxplot = True
    if need_boxplot:
        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([my_list, shaoshi_list],
                    labels=['Hybrid_val_loss', 'Shaoshi_val_loss'],
                    showfliers=False)
        plt.ylim()
        plt.ylabel('Loss')
        plt.title(f'Compare_Hybrid_vs_Shaoshi, p={p_value:.4f}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/group',
                'Compare_hybrid_shaoshi_no_finetune.png'))
        plt.close()
        print("Age figure saved.")


def parameter_surface_map(param_10_path, myelin, rsfc_gradient, save_map_path):
    d = torch.load(param_10_path, map_location='cpu')
    param_10 = d['param_10']
    w_EE = param_10[0] + param_10[1] * myelin + param_10[
        2] * rsfc_gradient  # [NUM_ROI, param_sets (100)]
    w_EI = param_10[3] + param_10[4] * myelin + param_10[5] * rsfc_gradient
    G = param_10[6]  # noqa
    sigma = param_10[7] + param_10[8] * myelin + param_10[9] * rsfc_gradient

    for i in range(1):

        sio.savemat(os.path.join(save_map_path, f'ee_{i}.mat'),
                    {'v': w_EE[:, i].unsqueeze(1).numpy()})
        sio.savemat(os.path.join(save_map_path, f'ei_{i}.mat'),
                    {'v': w_EI[:, i].unsqueeze(1).numpy()})
        sio.savemat(os.path.join(save_map_path, f'sigma_{i}.mat'),
                    {'v': sigma[:, i].unsqueeze(1).numpy()})
    print("Successfully saved.")


def plot_EI_ratio_individual():
    EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/EI_ratio/trial4/seed1'
    save_fig_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/figures/trial4/seed1_pFIC_EI.png'

    age_all = pd.read_csv(
        '/mnt/isilon/CSC1/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    age_all = np.array(age_all)[:, 1]

    valid_count = 0
    valid_sub_list = []
    EI_ave = []
    age_valid = []
    for sub_nbr in range(0, 885):
        EI_path = os.path.join(EI_dir, f'sub{sub_nbr}.pth')
        if os.path.exists(EI_path):
            valid_count += 1
            d = torch.load(EI_path, map_location='cpu')
            ei_tmp = torch.mean(d['ei_ratio'].squeeze()).item()
            # if ei_tmp < 2.25:
            valid_sub_list.append(sub_nbr)
            EI_ave.append(ei_tmp)
            age_valid.append(age_all[sub_nbr])
    print("Valid count: ", valid_count)

    analysis_functions.plot_EI_ratio(EI_ave,
                                     age_valid,
                                     save_fig_path,
                                     ylabel='EI_ratio')


def regional_EI_age_individual():
    EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/EI_ratio_rFIC/trial1/seed1'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/figures/trial1/rFIC_EI_regional.mat'
    nbr_list = np.arange(0, 885, 1)
    n_roi = NUM_ROI
    nbr_num = len(nbr_list)
    EI_regional_list = np.zeros((nbr_num, n_roi))
    age_list = np.zeros((nbr_num))
    age_all = pd.read_csv(
        '/mnt/isilon/CSC1/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    age_all = np.array(age_all)[:, 1]
    count = 0
    for i in range(nbr_num):
        nbr = nbr_list[i]
        # age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{nbr}/validation_subject_age.txt'
        # age = np.array(pd.read_csv(age_path, header=None, index_col=False))

        EI_path = os.path.join(EI_dir, f'sub{nbr}.pth')
        if not os.path.exists(EI_path):
            continue

        ei_tmp = torch.load(EI_path, map_location='cpu')
        EI_regional_list[count] = ei_tmp['ei_ratio'].squeeze().numpy()
        # age_list[i] = np.mean(age)
        age_list[count] = age_all[nbr]
        count += 1

    EI_regional_list = EI_regional_list[:count]
    age_list = age_list[:count]
    print("Valid count: ", count)
    slope_arr = np.zeros((n_roi))
    pvalue_arr = np.zeros((n_roi))
    for i in range(n_roi):
        # slope, intercept, rvalue, pvalue, stderr, intercept_stderr = stats.linregress(EI_regional_list[:, i], age_list, alternative='two-sided')
        # print(slope, pvalue)
        res = stats.linregress(age_list,
                               EI_regional_list[:, i],
                               alternative='two-sided')
        slope_arr[i] = res.slope
        pvalue_arr[i] = res.pvalue
    sio.savemat(
        save_path, {
            'regional_EI_vs_age_slope': slope_arr[:, np.newaxis],
            'p_value': pvalue_arr[:, np.newaxis]
        })
    print("Saved regional EI development.")
    return slope_arr, pvalue_arr


def generate_EI_ratio_ave_across_trials():
    EI_dir_all_trials = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/perturbation/age_group/EI_ratio'
    group_range = np.arange(1, NUM_GROUPS_PNC_AGE + 1)
    trials_range = np.arange(1, 101)
    EI_ratio_tensors = torch.zeros(NUM_ROI, len(group_range))

    # Find valid trials
    valid_trial_list = []
    for trial_nbr in trials_range:
        EI_dir_1_trial = os.path.join(EI_dir_all_trials, f'trial{trial_nbr}',
                                      'seed1')
        valid_flag = 1
        for group_nbr in group_range:
            EI_path = os.path.join(EI_dir_1_trial, f'group{group_nbr}.pth')
            if not os.path.exists(EI_path):
                valid_flag = 0
                break
        if valid_flag == 1:
            valid_trial_list.append(trial_nbr)
    print(f'In total {len(valid_trial_list)} valid trials, {valid_trial_list}')

    for trial_nbr in valid_trial_list:
        EI_dir_1_trial = os.path.join(EI_dir_all_trials, f'trial{trial_nbr}',
                                      'seed1')
        for i in range(len(group_range)):
            group_nbr = group_range[i]
            EI_path = os.path.join(EI_dir_1_trial, f'group{group_nbr}.pth')
            EI_file = torch.load(EI_path, map_location='cpu')
            EI_ratio_tensors[:, i] += torch.squeeze(EI_file['ei_ratio'])
    EI_ratio_tensors /= len(valid_trial_list)
    print("Averaged.")

    for i in range(len(group_range)):
        group_nbr = group_range[i]
        torch.save({'ei_ratio': EI_ratio_tensors[:, i].unsqueeze(1)},
                   os.path.join(EI_dir_all_trials, 'ave',
                                f'group{group_nbr}.pth'))
    print("Done.")
    return


def statistics_EI_age_group_perturbation():
    # Compute correlation and slope for each trial and statistic
    trial_list = [
        10, 13, 14, 15, 18, 22, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38,
        40, 41, 42, 44, 46, 48, 54, 55, 59, 60, 63, 64, 67, 68, 74, 76, 78, 79,
        83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 96, 98, 99, 100
    ]  # Must be valid
    EI_dir_all_trials = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/perturbation/age_group/EI_ratio'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/perturbation/age_group/figures/EI_ratio/statistics_50trials.mat'

    nbr_list = np.arange(1, NUM_GROUPS_PNC_AGE + 1, 1)
    n_roi = NUM_ROI
    nbr_num = len(nbr_list)

    # Get age list
    age_list = []
    for i in range(nbr_num):
        nbr = nbr_list[i]
        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{nbr}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        age_list.append(np.mean(age))
    age_list = np.array(age_list)

    # Get EI ratio
    EI_list = np.zeros((len(trial_list), nbr_num, n_roi))
    corr_mean_arr = np.zeros((len(trial_list)))
    slope_arr = np.zeros((len(trial_list), n_roi))
    pvalue_arr = np.zeros((len(trial_list), n_roi))
    for i in range(len(trial_list)):
        trial_nbr = trial_list[i]
        EI_dir_1_trial = os.path.join(EI_dir_all_trials, f'trial{trial_nbr}',
                                      'seed1')
        for j in range(nbr_num):
            group_nbr = nbr_list[j]
            EI_path = os.path.join(EI_dir_1_trial, f'group{group_nbr}.pth')
            EI_file = torch.load(EI_path, map_location='cpu')
            EI_ratio = torch.squeeze(EI_file['ei_ratio']).numpy()
            EI_list[i, j] = EI_ratio

        corr = np.corrcoef(
            np.mean(EI_list[i, :, :], axis=1).reshape(1, -1),
            age_list.reshape(1, -1))[0, 1]
        corr_mean_arr[i] = corr

        for region_i in range(n_roi):
            res = stats.linregress(age_list,
                                   EI_list[i, :, region_i],
                                   alternative='two-sided')
            slope_arr[i, region_i] = res.slope
            pvalue_arr[i, region_i] = res.pvalue

    sio.savemat(
        save_path, {
            'EI_list': EI_list,
            'corr_mean_arr': corr_mean_arr,
            'slope': slope_arr,
            'pvalue': pvalue_arr
        })
    print("Saved successfully.")


def statistics_EI_overall_acc_group_perturbation():
    # Compute correlation and slope for each trial and statistic
    # trial_list = [1, 3, 10, 13, 14, 15, 18, 22, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 48, 54, 55, 59, 60, 63, 64, 67, 74, 76, 78, 79, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 96, 98, 99]  # Must be valid # noqa
    trial_list = [3]
    EI_dir_high_trials = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/high/EI_ratio'
    EI_dir_low_trials = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/low/EI_ratio'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/figures/EI_ratio/trial2_statistics.mat'

    nbr_list = np.arange(1, NUM_GROUPS_PNC_COGNITION + 1, 1)
    n_roi = NUM_ROI
    nbr_num = len(nbr_list)

    # Get EI ratio
    # EI_diff_list = np.zeros((len(trial_list), nbr_num, n_roi))
    EI_high_list = np.zeros((len(trial_list), nbr_num, n_roi))
    EI_low_list = np.zeros((len(trial_list), nbr_num, n_roi))
    pvalue_arr = np.zeros((len(trial_list)))
    for i in range(len(trial_list)):
        trial_nbr = trial_list[i]
        EI_dir_high_trial = os.path.join(EI_dir_high_trials,
                                         f'trial{trial_nbr}', 'seed1')
        EI_dir_low_trial = os.path.join(EI_dir_low_trials, f'trial{trial_nbr}',
                                        'seed1')
        for j in range(nbr_num):
            group_nbr = nbr_list[j]
            EI_high = torch.load(os.path.join(EI_dir_high_trial,
                                              f'group{group_nbr}.pth'),
                                 map_location='cpu')
            EI_high = torch.squeeze(EI_high['ei_ratio']).numpy()
            EI_low = torch.load(os.path.join(EI_dir_low_trial,
                                             f'group{group_nbr}.pth'),
                                map_location='cpu')
            EI_low = torch.squeeze(EI_low['ei_ratio']).numpy()
            EI_high_list[i, j] = EI_high
            EI_low_list[i, j] = EI_low

        statistics, p_value = stats.ttest_ind(
            np.mean(EI_high_list[i, :, :], axis=1),
            np.mean(EI_low_list[i, :, :], axis=1))
        pvalue_arr[i] = p_value

    sio.savemat(
        save_path, {
            'EI_high_list': EI_high_list,
            'EI_low_list': EI_low_list,
            'pvalue': pvalue_arr
        })
    print("Saved successfully.")


def compare_test_results_plot():
    test_dir_1 = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group_by_45/test/trial1'
    test_dir_2 = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group_by_45/test/trial5'
    best_val_or_test_mean = 'best_val'
    prefix = 'group'
    nbr_range = np.arange(1, 20)
    loss_lists = analysis_functions.compare_test_results_two_lists(
        test_dir_1, test_dir_2, best_val_or_test_mean, prefix, nbr_range)
    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group_by_45/figures/compare_shaoshi_1_5_seed'
    fig_name = f'shaoshi_1seed_vs_shaoshi_5seed_{best_val_or_test_mean}'
    labels = ['1', '5']
    need_plot = True
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        loss_lists[0, :, 0],
        loss_lists[1, :, 0],
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_all_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Total Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        loss_lists[0, :, 1],
        loss_lists[1, :, 1],
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_corr_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Corr Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        loss_lists[0, :, 2],
        loss_lists[1, :, 2],
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_L1_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='L1 Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        loss_lists[0, :, 3],
        loss_lists[1, :, 3],
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_KS_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='KS Loss')


def compare_test_results_many_dirs_plot():

    test_dirs = [
        f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/group_by_45/test/trial{i}'
        for i in range(1, 6)
    ]
    best_val_or_test_mean = 'best_val'
    prefix = 'group'
    nbr_range = np.arange(1, 20)
    loss_lists = analysis_functions.get_test_results_from_many_dirs(
        test_dirs, best_val_or_test_mean, prefix, nbr_range)

    print(np.mean(loss_lists[:, :, 0], axis=1))
    '''
    labels = [f'{i} seed' for i in range(1, 6)]
    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group_by_45/figures/compare_shaoshi_1_5_seed'
    fig_name = f'Shaoshi_1_5_seeds_{best_val_or_test_mean}'

        tzeng_func.tzeng_boxplot(
        loss_lists[:, :, 0],
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_all_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Total Loss')
    tzeng_func.tzeng_boxplot(
        loss_lists[:, :, 1],
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_corr_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Corr Loss')
    tzeng_func.tzeng_boxplot(
        loss_lists[:, :, 2],
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_l1_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='L1 Loss')
    tzeng_func.tzeng_boxplot(
        loss_lists[:, :, 3],
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_ks_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='KS Loss')

    for i in range(loss_lists.shape[0] - 1):
        print(i + 1)
        for j in range(i + 1, loss_lists.shape[0]):
            statistics, p_value = stats.ttest_ind(loss_lists[i, :, 0], loss_lists[j, :, 0])
            print(f"{p_value}", end=' ')
        print('\n')
    '''


############################################################
# Actively Used
############################################################


def corr_mean_EI_vs_age(trial_idx, seed_idx):
    EI_dir = get_run_path('PNC', 'age_group', 'EI_ratio', trial_idx, seed_idx)
    save_fig_path = get_fig_file_path('PNC', 'age_group', 'EI_ratio',
                                      trial_idx, seed_idx,
                                      'corr_mean_EI_vs_age.png')
    EI_ave_list = []
    age_list = []
    for group_nbr in range(1, NUM_GROUPS_PNC_AGE + 1):
        # age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}/validation_subject_age.txt'
        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/age_results/input/{group_nbr}/validation_subject_age.txt'
        EI_path = os.path.join(EI_dir, f'group{group_nbr}.pth')
        if not os.path.exists(EI_path):
            continue
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        ei_tmp = torch.load(EI_path, map_location='cpu')
        EI_ave_list.append(torch.mean(ei_tmp['ei_ratio']).item())
        age_list.append(np.mean(age))
    print(EI_ave_list)

    analysis_functions.plot_EI_ratio(EI_ave_list, age_list, save_fig_path)


def export_regional_EI_vs_age_slope(trial_idx, seed_idx, save_mat_path=None):
    EI_dir = get_run_path('PNC', 'age_group', 'EI_ratio', trial_idx, seed_idx)
    if save_mat_path is None:
        save_mat_path = get_fig_file_path('PNC', 'age_group', 'EI_ratio',
                                          trial_idx, seed_idx,
                                          'regional_EI_vs_age_slope.mat')
    regional_EIs = np.zeros((NUM_GROUPS_PNC_AGE, NUM_ROI))
    ages = np.zeros(NUM_GROUPS_PNC_AGE)
    count = 0
    for i in range(NUM_GROUPS_PNC_AGE):
        group_idx = i + 1
        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_idx}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))

        EI_path = os.path.join(EI_dir, f'group{group_idx}.pth')
        if not os.path.exists(EI_path):
            continue

        ei_tmp = torch.load(EI_path, map_location='cpu')
        regional_EIs[count] = ei_tmp['ei_ratio'].squeeze().numpy()
        ages[i] = np.mean(age)
        count += 1

    regional_EIs = regional_EIs[:count]
    ages = ages[:count]
    print("Valid count: ", count)
    slope_arr, pvalue_arr, pvalue_fdr = regional_EI_age_slope(
        NUM_ROI, ages, regional_EIs)
    sio.savemat(
        save_mat_path, {
            'regional_EI_vs_age_slope': slope_arr[:, np.newaxis],
            'pvalue': pvalue_arr[:, np.newaxis],
            'pvalue_fdr': pvalue_fdr[:, np.newaxis]
        })
    # reg = LinearRegression()
    # reg.fit(age_list[:, np.newaxis], EI_regional_list)
    # sio.savemat(save_path, {'regional_EI_vs_age_slope': reg.coef_})
    print("Saved regional_EI_vs_age_slope.")
    return slope_arr, pvalue_arr, pvalue_fdr


def visualize_regional_EI_vs_age_slope(trial_idx, seed_idx):
    save_mat_path = get_fig_file_path('PNC', 'age_group', 'EI_ratio',
                                      trial_idx, seed_idx,
                                      'regional_EI_vs_age_slope.mat')
    export_regional_EI_vs_age_slope(trial_idx,
                                    seed_idx,
                                    save_mat_path=save_mat_path)

    visualize_stats(save_mat_path, 'regional_EI_vs_age_slope',
                    save_mat_path.replace('.mat', '_surf_map.png'))
    boxplot_network_stats(
        save_mat_path,
        'regional_EI_vs_age_slope',
        save_mat_path.replace('.mat', '_boxplot.png'),
    )


def plot_mean_EI_diff_t_test(trial_idx, seed_idx):
    high_ei_dir = get_run_path('PNC', 'overall_acc_group/high', 'EI_ratio',
                               trial_idx, seed_idx)
    low_ei_dir = get_run_path('PNC', 'overall_acc_group/low', 'EI_ratio',
                              trial_idx, seed_idx)
    save_fig_path = get_fig_file_path('PNC', 'overall_acc_group', 'EI_ratio',
                                      trial_idx, seed_idx,
                                      'mean_EI_diff_t_test.png')
    high_list = []
    low_list = []
    for group_idx in range(1, NUM_GROUPS_PNC_COGNITION + 1):
        high_ei_path = os.path.join(high_ei_dir, f'group{group_idx}.pth')
        low_ei_path = os.path.join(low_ei_dir, f'group{group_idx}.pth')
        high_ei = torch.load(high_ei_path, map_location='cpu')
        low_ei = torch.load(low_ei_path, map_location='cpu')
        high_list.append(torch.mean(high_ei['ei_ratio']).item())
        low_list.append(torch.mean(low_ei['ei_ratio']).item())
    print(high_list)
    print(low_list)
    ttest_1samp_n_plot(
        high_list,
        low_list,
        need_boxplot=True,
        need_pvalue=True,
        labels=['high-performance', 'low-performance'],
        save_fig_path=save_fig_path,
        fig_title='t-test on mean cortical E/I ratio difference',
        xlabel='performance group',
        ylabel='mean cortical E/I ratio')
    return


def export_EI_ratio_diff_effect_size(trial_idx, seed_idx, save_mat_path=None):
    high_ei_dir = get_run_path('PNC', 'overall_acc_group/high', 'EI_ratio',
                               trial_idx, seed_idx)
    low_ei_dir = get_run_path('PNC', 'overall_acc_group/low', 'EI_ratio',
                              trial_idx, seed_idx)
    if save_mat_path is None:
        save_mat_path = get_fig_file_path('PNC', 'overall_acc_group',
                                          'EI_ratio', trial_idx, seed_idx,
                                          'EI_ratio_diff_effect_size.mat')

    EI_matrix_high = np.zeros((NUM_GROUPS_PNC_COGNITION, NUM_ROI))
    EI_matrix_low = np.zeros((NUM_GROUPS_PNC_COGNITION, NUM_ROI))
    for i in range(NUM_GROUPS_PNC_COGNITION):
        group_idx = i + 1
        EI_high = torch.load(os.path.join(high_ei_dir,
                                          f'group{group_idx}.pth'),
                             map_location='cpu')
        EI_high = torch.squeeze(EI_high['ei_ratio']).numpy()
        EI_low = torch.load(os.path.join(low_ei_dir, f'group{group_idx}.pth'),
                            map_location='cpu')
        EI_low = torch.squeeze(EI_low['ei_ratio']).numpy()
        EI_matrix_high[i] = EI_high
        EI_matrix_low[i] = EI_low

    EI_ratio_diff_effect_size = regional_EI_diff_cohen_d(
        EI_matrix_high, EI_matrix_low)
    sio.savemat(save_mat_path,
                {'EI_ratio_diff_effect_size': EI_ratio_diff_effect_size})
    print("Saved successfully.")


def visualize_EI_ratio_diff_effect_size(trial_idx, seed_idx):
    save_mat_path = get_fig_file_path('PNC', 'overall_acc_group', 'EI_ratio',
                                      trial_idx, seed_idx,
                                      'EI_ratio_diff_effect_size.mat')
    export_EI_ratio_diff_effect_size(trial_idx,
                                     seed_idx,
                                     save_mat_path=save_mat_path)

    visualize_stats(save_mat_path, 'EI_ratio_diff_effect_size',
                    save_mat_path.replace('.mat', '_surf_map.png'))
    boxplot_network_stats(
        save_mat_path,
        'EI_ratio_diff_effect_size',
        save_mat_path.replace('.mat', '_boxplot.png'),
    )


def EI_analysis_age_group(trial_idx, seed_idx):
    corr_mean_EI_vs_age(trial_idx, seed_idx)
    visualize_regional_EI_vs_age_slope(trial_idx, seed_idx)


def EI_analysis_overall_acc_group(trial_idx, seed_idx):
    plot_mean_EI_diff_t_test(trial_idx, seed_idx)
    visualize_EI_ratio_diff_effect_size(trial_idx, seed_idx)


def EI_analysis(target, trial_idx, seed_idx):
    if target == 'age_group':
        EI_analysis_age_group(trial_idx, seed_idx)
    elif target == 'overall_acc_group/low':
        # no need to repeat analysis for high since high and low are considered together
        EI_analysis_overall_acc_group(trial_idx, seed_idx)


############################################################
# Various analysis pipeline
############################################################
def analyze_epoch(target, trial_idx, seed_idx, group_idx, epoch_idx):
    visualize_train_r_E('PNC', target, trial_idx, seed_idx, group_idx,
                        epoch_idx)


def analyze_group(target, trial_idx, seed_idx, group_idx):
    # plot_train_loss('PNC',
    #                 target,
    #                 trial_idx,
    #                 seed_idx,
    #                 group_idx,
    #                 epoch_range=range(49))
    # plot_train_loss('PNC',
    #                 target,
    #                 trial_idx,
    #                 seed_idx,
    #                 group_idx,
    #                 epoch_range=range(49),
    #                 show_individual_loss='r_E_reg_loss')
    boxplot_train_r_E('PNC', target, trial_idx, seed_idx, group_idx,
                      [0, 9, 19, 29, 39, 49], True)


def analyze_run(target, trial_idx, seed_idx):
    all_groups_EI_to_csv('PNC', NUM_GROUPS[target], target, trial_idx,
                         seed_idx)


def analyze_trial(target, trial_idx):
    """
    Analyze the trial-level results.
    """
    export_EI_from_param_with_lowest_loss_among_seeds(
        'PNC', target, trial_idx, range(1, 3), range(1,
                                                     NUM_GROUPS[target] + 1))
    all_groups_EI_to_csv('PNC', NUM_GROUPS[target], target, trial_idx,
                         '_best_among_all')
    # export_lowest_losses_among_seeds('PNC', target, trial_idx, range(1, 3),
    #                                  range(1, NUM_GROUPS[target] + 1))
    EI_analysis(target, trial_idx, '_best_among_all')

    # # * if each seed's Tester will compare the current seed's validation results
    # # * with previous seeds' validation results and get the best one, use the following lines
    # last_seed_idx = 2
    # EI_analysis(trial_idx, last_seed_idx)


def analyze_target(target):
    """
    Analyze the target-level results.
    """
    # plot_losses_for_diff_trials_all_groups(
    #     'PNC', target, [0, 1, 4, 3, 6],
    #     ['baseline', 'MAE L1', 'fixed sigma', 'rE reg', 'free rE'])

    # # Plot losses

    GROUP_IDX = 1
    COMMON_KWARGS = {
        'ds_name': 'PNC',
        'target': target,
        'seed_range': range(1, 3),
        'group_idx': GROUP_IDX,
        'epoch_range': range(49)
    }
    # Compare total loss between different setups
    plot_losses_for_diff_trials(
        loss_types=['total_loss'],
        trial_range=[5, 4, 6],
        trial_names=['baseline', 'fixed sigma', 'free rE'],
        **COMMON_KWARGS)

    # Compare corr loss between different setups
    plot_losses_for_diff_trials(
        loss_types=['corr_loss'],
        trial_range=[5, 1, 4, 3, 6],
        trial_names=['baseline', 'MAE L1', 'fixed sigma', 'rE reg', 'free rE'],
        **COMMON_KWARGS)

    # Compare l1 loss between different setups
    plot_losses_for_diff_trials(
        loss_types=['l1_loss'],
        trial_range=[5, 4, 3, 6],
        trial_names=['baseline', 'fixed sigma', 'rE reg', 'free rE'],
        **COMMON_KWARGS)

    # Compare ks loss between different setups
    plot_losses_for_diff_trials(
        loss_types=['ks_loss'],
        trial_range=[5, 1, 4, 3, 6],
        trial_names=['baseline', 'MAE L1', 'fixed sigma', 'rE reg', 'free rE'],
        **COMMON_KWARGS)

    # Compare r_E reg loss between different setups
    plot_losses_for_diff_trials(loss_types=['r_E_reg_loss'],
                                trial_range=[3],
                                trial_names=['rE reg'],
                                **COMMON_KWARGS)


if __name__ == "__main__":
    ALL_TARGETS = [
        'age_group', 'overall_acc_group/high', 'overall_acc_group/low'
    ]
    # Epoch-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in [3, 6]:
    #         for seed_idx in [1]:
    #             for group_idx in [1]:
    #                 for epoch_idx in [0, 9, 19, 29, 39, 49]:
    #                     analyze_epoch(target, trial_idx, seed_idx, group_idx,
    #                                   epoch_idx)

    # Group-level analysis
    for target in ALL_TARGETS:
        for trial_idx in [3, 6]:
            for seed_idx in range(1, 2):
                # for group_idx in range(1, NUM_GROUPS[target] + 1):
                for group_idx in range(1, 2):
                    analyze_group(target, trial_idx, seed_idx, group_idx)

    # Run-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in range(3, 4):
    #         for seed_idx in range(1, 3):
    #             analyze_run(target, trial_idx, seed_idx)

    # Trial-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in range(1, 7):
    #         analyze_trial(target, trial_idx)

    # Target-level analysis
    # for target in ALL_TARGETS:
    #     analyze_target(target)

    # Debugging
    # plot_pred_loss()
    # corr_mean_EI_vs_age(1, 1)
    # visualize_regional_EI_vs_age_slope(1, 1)
    # plot_mean_EI_diff_t_test(1, 1)
    # visualize_EI_ratio_diff_effect_size(1, 1)

    # EI_analysis(1, 1)
