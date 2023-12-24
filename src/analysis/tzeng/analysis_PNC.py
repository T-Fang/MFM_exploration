import torch
import scipy.io as sio
import scipy.stats as stats
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(1, '/home/ftian/storage/projects/MFM_exploration')
from src.analysis.tzeng import analysis_functions
from src.utils import tzeng_func


def plot_pred_loss():
    param_save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/high/train/trial3/seed1/group14'
    figure_save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/figures/pred_loss/t3se1g14.png'
    analysis_functions.plot_pred_loss(epochs=50,
                                      param_save_dir=param_save_dir,
                                      figure_path=figure_save_path)


def compare_train_loss(nbr):
    dir_1 = f'/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/FinetuneData/test/trial1/sub{nbr}'
    dir_2 = f'/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/FinetuneData/test/trial1/sub{nbr}'
    epoch_nbr_1 = 99
    epoch_nbr_2 = 119

    path_1 = os.path.join(dir_1, f'param_save_epoch{epoch_nbr_1}.pth')
    path_2 = os.path.join(dir_2, f'param_save_epoch{epoch_nbr_2}.pth')
    if not os.path.exists(path_1) or not os.path.exists(path_2):
        raise Exception("Check path first.")
    d_1 = torch.load(path_1)
    d_2 = torch.load(path_2)
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
    for group_nbr in range(1, 30):
        shaoshi = np.squeeze(
            np.array(
                pd.read_csv(
                    f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/test/{group_nbr}/test_all.csv',
                    header=None,
                    index_col=False)))
        shaoshi_loss = shaoshi[9]
        '''
        shaoshi = torch.load(
            f'/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group/test_shaoshi/group{group_nbr}/val_results.pth'
        )
        shaoshi_loss = shaoshi['corr_loss'][0].item() + shaoshi['l1_loss'][0].item() + shaoshi['ks_loss'][0].item()
        '''
        shaoshi_list.append(shaoshi_loss)

        me = torch.load(
            f'/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group/test_before/group{group_nbr}/val_results.pth'
        )
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
    d = torch.load(param_10_path)
    param_10 = d['param_10']
    w_EE = param_10[0] + param_10[1] * myelin + param_10[
        2] * rsfc_gradient  # [68, param_sets (100)]
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
    EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/individual/EI_ratio/trial4/seed1'
    save_fig_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/individual/figures/trial4/seed1_pFIC_EI.png'

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
            d = torch.load(EI_path)
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
    EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/individual/EI_ratio_rFIC/trial1/seed1'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/individual/figures/trial1/rFIC_EI_regional.mat'
    nbr_list = np.arange(0, 885, 1)
    n_roi = 68
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

        ei_tmp = torch.load(EI_path)
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
    sio.savemat(save_path, {
        'EI_ROI': slope_arr[:, np.newaxis],
        'p_value': pvalue_arr[:, np.newaxis]
    })
    print("Saved regional EI development.")
    return slope_arr, pvalue_arr


def plot_EI_ratio_age_group():
    EI_dir = '/home/tzeng/storage/Python/MFMApplication/Params/PNCParams/age_group/EI_ratio/trial4/seed1'
    # EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/age_group/EI_ratio/trial1/seed1'
    save_fig_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/age_group/figures/EI/trial1_development_fit.png'

    EI_ave_list = []
    age_list = []
    for group_nbr in range(1, 30):
        # age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}/validation_subject_age.txt'
        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/age_results/input/{group_nbr}/validation_subject_age.txt'
        EI_path = os.path.join(EI_dir, f'group{group_nbr}.pth')
        if not os.path.exists(EI_path):
            continue
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        ei_tmp = torch.load(EI_path)
        EI_ave_list.append(torch.mean(ei_tmp['ei_ratio']).item())
        age_list.append(np.mean(age))
    print(EI_ave_list)

    analysis_functions.plot_EI_ratio(EI_ave_list, age_list, save_fig_path)


def generate_EI_ratio_ave_across_trials():
    EI_dir_all_trials = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/perturbation/age_group/EI_ratio'
    group_range = np.arange(1, 30)
    trials_range = np.arange(1, 101)
    EI_ratio_tensors = torch.zeros(68, len(group_range))

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
            EI_file = torch.load(EI_path)
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


def regional_EI_age_group():
    EI_dir = '/home/tzeng/storage/Python/MFMApplication/Params/PNCParams/age_group/EI_ratio/trial4/seed1'
    # EI_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/age_group/EI_ratio/trial1/seed1'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/age_group/figures/EI/trial1_regional.mat'
    nbr_list = np.arange(1, 30, 1)
    n_roi = 68
    nbr_num = len(nbr_list)
    EI_regional_list = np.zeros((nbr_num, n_roi))
    age_list = np.zeros((nbr_num))
    count = 0
    for i in range(nbr_num):
        nbr = nbr_list[i]
        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{nbr}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))

        EI_path = os.path.join(EI_dir, f'group{nbr}.pth')
        if not os.path.exists(EI_path):
            continue

        ei_tmp = torch.load(EI_path)
        EI_regional_list[count] = ei_tmp['ei_ratio'].squeeze().numpy()
        age_list[i] = np.mean(age)
        count += 1

    EI_regional_list = EI_regional_list[:count]
    age_list = age_list[:count]
    print("Valid count: ", count)
    slope_arr, pvalue_arr, pvalue_fdr = analysis_functions.EI_age_slope_regional(
        n_roi, age_list, EI_regional_list)
    sio.savemat(
        save_path, {
            'EI_ROI': slope_arr[:, np.newaxis],
            'pvalue': pvalue_arr[:, np.newaxis],
            'pvalue_fdr': pvalue_fdr[:, np.newaxis]
        })
    # reg = LinearRegression()
    # reg.fit(age_list[:, np.newaxis], EI_regional_list)
    # sio.savemat(save_path, {'EI_ROI': reg.coef_})
    print("Saved regional EI development.")
    return slope_arr, pvalue_arr, pvalue_fdr


def statistics_EI_age_group_perturbation():
    # Compute correlation and slope for each trial and statistic
    trial_list = [
        10, 13, 14, 15, 18, 22, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38,
        40, 41, 42, 44, 46, 48, 54, 55, 59, 60, 63, 64, 67, 68, 74, 76, 78, 79,
        83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 96, 98, 99, 100
    ]  # Must be valid
    EI_dir_all_trials = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/perturbation/age_group/EI_ratio'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/perturbation/age_group/figures/EI_ratio/statistics_50trials.mat'

    nbr_list = np.arange(1, 30, 1)
    n_roi = 68
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
            EI_file = torch.load(EI_path)
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


def plot_EI_overall_acc_group():
    high_ei_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/high/EI_ratio/trial3/seed3'
    low_ei_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/low/EI_ratio/trial3/seed3'
    save_fig_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/figures/EI_ratio/' \
                    'trial3_seed3_overall_comparison.png'
    save_fig_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/figures/EI_ratio/' \
                    'trial3_seed3_overall_comparison.png'
    high_list = []
    low_list = []
    for nbr in range(1, 15):
        high_ei = os.path.join(high_ei_dir, f'group{nbr}.pth')
        low_ei = os.path.join(low_ei_dir, f'group{nbr}.pth')
        high_ei = torch.load(high_ei)
        low_ei = torch.load(low_ei)
        high_list.append(torch.mean(high_ei['ei_ratio']).item())
        low_list.append(torch.mean(low_ei['ei_ratio']).item())
    print(len(high_list))
    print(high_list)
    print(low_list)
    pvalue = stats.ttest_1samp(np.array(high_list) - np.array(low_list), 0)
    print(pvalue)
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        high_list,
        low_list,
        need_boxplot=True,
        need_pvalue=False,
        labels=['high acc', 'low acc'],
        save_fig_path=save_fig_path,
        fig_title='high_acc_vs_low_acc_EI_ratio',
        xlabel='performance group',
        ylabel='mean cortical E/I ratio')
    return


def overall_acc_group_effect_size():
    EI_dir_high_trial = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/high/EI_ratio/trial3/seed3'
    EI_dir_low_trial = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/low/EI_ratio/trial3/seed3'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/figures/EI_ratio/trial3_seed3_EI.mat'

    nbr_list = np.arange(1, 15, 1)
    n_roi = 68
    nbr_num = len(nbr_list)

    EI_high_list = np.zeros((nbr_num, n_roi))
    EI_low_list = np.zeros((nbr_num, n_roi))
    for i in range(nbr_num):
        group_nbr = nbr_list[i]
        EI_high = torch.load(
            os.path.join(EI_dir_high_trial, f'group{group_nbr}.pth'))
        EI_high = torch.squeeze(EI_high['ei_ratio']).numpy()
        EI_low = torch.load(
            os.path.join(EI_dir_low_trial, f'group{group_nbr}.pth'))
        EI_low = torch.squeeze(EI_low['ei_ratio']).numpy()
        EI_high_list[i] = EI_high
        EI_low_list[i] = EI_low

    sio.savemat(save_path, {
        'EI_high_list': EI_high_list,
        'EI_low_list': EI_low_list
    })
    print("Saved successfully.")


def statistics_EI_overall_acc_group_perturbation():
    # Compute correlation and slope for each trial and statistic
    # trial_list = [1, 3, 10, 13, 14, 15, 18, 22, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 40, 41, 42, 44, 46, 48, 54, 55, 59, 60, 63, 64, 67, 74, 76, 78, 79, 83, 84, 86, 87, 88, 89, 91, 92, 93, 94, 96, 98, 99]  # Must be valid # noqa
    trial_list = [3]
    EI_dir_high_trials = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/high/EI_ratio'
    EI_dir_low_trials = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/low/EI_ratio'
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/overall_acc_group/figures/EI_ratio/trial2_statistics.mat'

    nbr_list = np.arange(1, 15, 1)
    n_roi = 68
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
            EI_high = torch.load(
                os.path.join(EI_dir_high_trial, f'group{group_nbr}.pth'))
            EI_high = torch.squeeze(EI_high['ei_ratio']).numpy()
            EI_low = torch.load(
                os.path.join(EI_dir_low_trial, f'group{group_nbr}.pth'))
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
    test_dir_1 = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/test/trial1'
    test_dir_2 = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/test/trial5'
    best_val_or_test_mean = 'best_val'
    prefix = 'group'
    nbr_range = np.arange(1, 20)
    loss_lists = analysis_functions.compare_test_results_two_lists(
        test_dir_1, test_dir_2, best_val_or_test_mean, prefix, nbr_range)
    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/figures/compare_shaoshi_1_5_seed'
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
        f'/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/test/trial{i}'
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
    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/figures/compare_shaoshi_1_5_seed'
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


if __name__ == "__main__":
    # plot_pred_loss()
    # plot_EI_overall_acc_group()
    # overall_acc_group_effect_size()
    plot_EI_ratio_age_group()
    # regional_EI_age_group()
