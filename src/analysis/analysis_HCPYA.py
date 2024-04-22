import torch
import scipy.io as sio
# import scipy.stats as stats
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.insert(1, '/home/ftian/storage/projects/MFM_exploration')
from src.basic.constants import SUBJECT_ID_RANGE
from src.analysis import analysis_functions
from src.utils import tzeng_func
from src.utils.analysis_utils import boxplot_train_r_E, compare_fc, compare_fcd, plot_train_loss, scatter_loss_vs_r_E, visualize_train_r_E_for_multi_epochs, draw_heatmap, plot_sim_fc_fcd  # noqa
from src.utils.file_utils import get_HCPYA_group_emp_TC, get_HCPYA_group_emp_fc, get_emp_fig_dir, get_sim_fig_path
from src.models.mfm_2014 import MfmModel2014

sub_109 = [
    862, 863, 864, 865, 866, 868, 869, 870, 871, 872, 873, 874, 875, 877, 879,
    881, 883, 885, 888, 889, 890, 891, 892, 895, 896, 897, 900, 901, 902, 904,
    905, 907, 911, 912, 914, 916, 918, 919, 921, 924, 925, 929, 930, 931, 933,
    935, 936, 937, 938, 939, 941, 943, 944, 945, 946, 949, 950, 951, 952, 953,
    954, 955, 956, 960, 962, 964, 966, 967, 968, 969, 970, 971, 972, 973, 974,
    976, 977, 979, 981, 983, 984, 986, 987, 988, 989, 990, 991, 993, 994, 996,
    998, 999, 1001, 1002, 1003, 1004, 1006, 1008, 1014, 1016, 1017, 1019, 1021,
    1022, 1023, 1024, 1025, 1026, 1028
]

DS_NAME = 'HCPYA'
N_EPOCHS = 100


def get_HCPYA_group_emp_fcd(range_start: int, range_end: int):
    group_TC = get_HCPYA_group_emp_TC(range_start, range_end)
    group_TC = group_TC.unsqueeze(1)
    fcd, _ = MfmModel2014.FCD_calculate(group_TC, get_FCD_matrix=True)
    fcd = fcd.squeeze()
    assert fcd.ndim == 2

    return fcd


def plot_HCPYA_group_emp_fc_fcd(range_start: int, range_end: int):
    # get emp fc and emp fcd tensors
    group_emp_fc = get_HCPYA_group_emp_fc(range_start, range_end).cpu().numpy()
    group_emp_fcd = get_HCPYA_group_emp_fcd(range_start,
                                            range_end).cpu().numpy()

    fcd_save_dir = get_emp_fig_dir(DS_NAME, 'all_participants', 'FCD')
    fc_save_dir = get_emp_fig_dir(DS_NAME, 'all_participants', 'FC')
    fcd_save_path = os.path.join(
        fcd_save_dir, f'group_emp_fcd_{range_start}_{range_end}.csv')
    fc_save_path = os.path.join(fc_save_dir,
                                f'group_emp_fc_{range_start}_{range_end}.csv')

    # save to csv
    pd.DataFrame(group_emp_fc).to_csv(fc_save_path, index=False, header=False)
    pd.DataFrame(group_emp_fcd).to_csv(fcd_save_path,
                                       index=False,
                                       header=False)

    # draw heatmaps
    draw_heatmap([fc_save_path, fcd_save_path], [
        f'empirical FC for subjects {range_start} to {range_end}',
        f'empirical FCD for subjects {range_start} to {range_end}'
    ], [
        fc_save_path.replace('.csv', '.png'),
        fcd_save_path.replace('.csv', '.png')
    ])


def plot_pred_loss():
    param_save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029/train/trial1/seed1'
    figure_save_path = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029/figures/pred_loss/trial1_seed1.png'
    analysis_functions.plot_pred_loss(epochs=100,
                                      param_save_dir=param_save_dir,
                                      figure_path=figure_save_path)
    return


def check_dl_epochs():
    trial_nbr = 1
    nbr = 1
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/age_group'
    train_results_dir = os.path.join(parent_dir,
                                     f'train/trial{trial_nbr}/group{nbr}')
    val_results_dir = os.path.join(
        parent_dir, f'val_train_param/trial{trial_nbr}/group{nbr}')
    save_dir = os.path.join(parent_dir, 'figures')
    epochs = 67
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


def age_distribution():
    age_file = '/home/tzeng/storage/Matlab/HCPS1200/txt_files/HCPYA_ages_1029.txt'
    age_file = pd.read_csv(age_file, sep='\t', header=0, index_col=False)

    ages = age_file.iloc[:, 1]
    ages = np.squeeze(np.array(ages))
    min_age = np.amin(ages)
    max_age = np.amax(ages)
    plt.figure()
    plt.hist(ages, bins=np.arange(int(min_age), int(max_age) + 1, 1))
    plt.title('HCP-YA age distribution')
    plt.xlabel('Age/year')
    plt.ylabel('number')
    plt.savefig(
        '/home/ftian/storage/projects/MFM_exploration/reports/figures/HCPYA/HCPYA_age_dist.png'
    )
    plt.close()
    print("Done.")


def plot_EI_ratio_group():
    trial_nbr = 1
    EI_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/age_group/EI_ratio/trial{trial_nbr}'
    save_fig_path = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/age_group/figures/EI/trial{trial_nbr}_development_fit.png'

    EI_ave_list = []
    age_list = []
    for group_nbr in range(1, 35):
        age_path = f'/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/age_group/{group_nbr}/validation/validation_subject_demo.txt'
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
    trial_nbr = 1
    EI_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/age_group/EI_ratio/trial{trial_nbr}'
    save_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/pMFM/EI_ratio/EI_group_regional_age.mat'

    EI_regional_list = []
    age_list = []
    for group_nbr in range(1, 35):
        age_path = f'/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/age_group/{group_nbr}/validation/validation_subject_demo.txt'
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


def compare_test_results_individual(trial_1, trial_2):
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/test'
    all_loss_list_1 = []
    all_loss_list_2 = []
    corr_loss_list1 = []
    corr_loss_list2 = []
    l1_loss_list1 = []
    l1_loss_list2 = []
    ks_loss_list1 = []
    ks_loss_list2 = []
    for sub_nbr in range(862, 1029):
        # for sub_nbr in sub_list:
        path_1 = os.path.join(
            parent_dir,
            f'trial{trial_1}/split1_2RLtest_2LRval_1LR1RLtrain/sub{sub_nbr}/test_results.pth'
        )
        if not os.path.exists(path_1):
            continue
        path_2 = os.path.join(
            parent_dir,
            f'trial{trial_2}/split1_2RLtest_2LRval_1LRtrain/sub{sub_nbr}/test_results.pth'
        )
        if not os.path.exists(path_2):
            continue
        res_1 = torch.load(path_1)
        # total_loss_1, index_min_1 = torch.min(res_1['corr_loss'] + res_1['l1_loss'] + res_1['ks_loss'], dim=0)  # min of all test loss
        total_loss_1 = res_1['corr_loss'][0] + res_1['l1_loss'][0] + res_1[
            'ks_loss'][0]  # parameter with min validation loss
        index_min_1 = 0
        res_2 = torch.load(path_2)
        # total_loss_2, index_min_2 = torch.min(res_2['corr_loss'] + res_2['l1_loss'] + res_2['ks_loss'], dim=0)
        total_loss_2 = res_2['corr_loss'][0] + res_2['l1_loss'][0] + res_2[
            'ks_loss'][0]
        index_min_2 = 0
        all_loss_list_1.append(total_loss_1.item())
        all_loss_list_2.append(total_loss_2.item())
        corr_loss_list1.append(res_1['corr_loss'][index_min_1].item())
        corr_loss_list2.append(res_2['corr_loss'][index_min_2].item())
        l1_loss_list1.append(res_1['l1_loss'][index_min_1].item())
        l1_loss_list2.append(res_2['l1_loss'][index_min_2].item())
        ks_loss_list1.append(res_1['ks_loss'][index_min_1].item())
        ks_loss_list2.append(res_2['ks_loss'][index_min_2].item())
    print(len(all_loss_list_1))
    return all_loss_list_1, all_loss_list_2, corr_loss_list1, corr_loss_list2, l1_loss_list1, l1_loss_list2, ks_loss_list1, ks_loss_list2


def compare_test_results_plot():
    test_dir_1 = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/test/trial24/split1'
    test_dir_2 = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/test/trial26/split1'
    best_val_or_test_mean = 'best_val'
    prefix = 'sub'
    nbr_range = np.arange(860, 1029)
    loss_lists = analysis_functions.compare_test_results_two_lists(
        test_dir_1, test_dir_2, best_val_or_test_mean, prefix, nbr_range)

    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/figures/compare_L1_MAE'
    fig_name = f'L1_vs_MAE_{best_val_or_test_mean}'
    labels = ['L1', 'MAE']
    need_plot = False
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


def compare_group_params_apply_to_individual():
    all_loss_group = []
    all_loss_individual = []
    corr_loss_group = []
    corr_loss_individual = []
    l1_loss_group = []
    l1_loss_individual = []
    ks_loss_group = []
    ks_loss_individual = []
    group_losses = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/trial3/seed1/loss_group_to_subj_2RLtest_MAE.pth'
    )
    total_loss = torch.sum(group_losses, dim=2)

    for i in range(len(group_losses)):
        sub_nbr = sub_109[i]  # noqa

        path_sub = (
            '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/'
            'individual/test/trial26/split1/sub{sub_nbr}/test_results.pth')
        if not os.path.exists(path_sub):
            continue
        res_sub = torch.load(path_sub)
        # corr_loss_individual.append(res_sub['corr_loss'][0].item())
        # l1_loss_individual.append(res_sub['l1_loss'][0].item())
        # ks_loss_individual.append(res_sub['ks_loss'][0].item())
        # all_loss_individual.append(res_sub['corr_loss'][0].item() + res_sub['l1_loss'][0].item() + res_sub['ks_loss'][0].item())
        corr_loss_individual.append(torch.mean(res_sub['corr_loss']).item())
        l1_loss_individual.append(torch.mean(res_sub['l1_loss']).item())
        ks_loss_individual.append(torch.mean(res_sub['ks_loss']).item())
        all_loss_individual.append(
            torch.mean(res_sub['corr_loss'] + res_sub['l1_loss'] +
                       res_sub['ks_loss']).item())

        # corr_loss_group.append(group_losses[i, 0, 0].item())
        # l1_loss_group.append(group_losses[i, 0, 1].item())
        # ks_loss_group.append(group_losses[i, 0, 2].item())
        # all_loss_group.append(total_loss[i, 0].item())
        corr_loss_group.append(torch.mean(group_losses[i, :, 0]).item())
        l1_loss_group.append(torch.mean(group_losses[i, :, 1]).item())
        ks_loss_group.append(torch.mean(group_losses[i, :, 2]).item())
        all_loss_group.append(torch.mean(total_loss[i]).item())

    print("Valid subjects: ", len(all_loss_individual))

    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/figures/group_apply_to_subj'
    fig_name = 'mean_group_params_apply_to_subj_2RLtest_MAE_1seed'
    labels = ['Group', 'Individual']
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        all_loss_group,
        all_loss_individual,
        True,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_all_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Total Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        corr_loss_group,
        corr_loss_individual,
        True,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_corr_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Corr Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        l1_loss_group,
        l1_loss_individual,
        True,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_L1_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='L1 Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        ks_loss_group,
        ks_loss_individual,
        True,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_KS_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='KS Loss')


def compare_group_test_results():
    res_1 = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029/test/trial2/seed1/test_results.pth'
    )
    res1_total_loss = res_1['corr_loss'] + res_1['l1_loss'] + res_1['ks_loss']
    res_2 = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029/test/trial1/seed5/test_results.pth'
    )
    res2_total_loss = res_2['corr_loss'] + res_2['l1_loss'] + res_2['ks_loss']

    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029/figures'
    fig_name = 'Hybrid_vs_Normal'
    labels = ['Hybrid 1', 'Normal 1']
    print("Res 1 min loss: ", torch.min(res1_total_loss))
    print("Res 2 min loss: ", torch.min(res2_total_loss))
    need_plot = True
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        res1_total_loss.numpy(),
        res2_total_loss.numpy(),
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_all_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Total Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        res_1['corr_loss'].numpy(),
        res_2['corr_loss'].numpy(),
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_corr_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Corr Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        res_1['l1_loss'].numpy(),
        res_2['l1_loss'].numpy(),
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_L1_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='L1 Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        res_1['ks_loss'].numpy(),
        res_2['ks_loss'].numpy(),
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_KS_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='KS Loss')


def analyze_run(target, trial_idx, seed_idx):
    for k in [1, 10]:
        plot_train_loss(DS_NAME,
                        target,
                        trial_idx,
                        seed_idx,
                        None,
                        epoch_range=range(99),
                        lowest_top_k=k)
    # plot_train_loss(DS_NAME,
    #                 target,
    #                 trial_idx,
    #                 seed_idx,
    #                 None,
    #                 epoch_range=range(49),
    #                 show_individual_loss='r_E_reg_loss')
    for top_k in [10, 100]:
        for plot_outlier_r_E in [True, False]:
            boxplot_train_r_E(DS_NAME,
                              target,
                              trial_idx,
                              seed_idx,
                              None, [0, 19, 39, 59, 79, 99],
                              top_k=top_k,
                              plot_outlier_r_E=plot_outlier_r_E)
    # visualize_train_r_E_for_multi_epochs(DS_NAME, target, trial_idx, seed_idx,
    #                                      None, N_EPOCHS, 2, 3)


def analyze_phase(target, trial_idx, seed_idx, phase):
    plot_sim_fc_fcd(DS_NAME, target, phase, trial_idx, seed_idx, 0)
    sim_fc_path = get_sim_fig_path(DS_NAME, target, phase, trial_idx, seed_idx,
                                   'sim_fc.csv')
    sim_fcd_path = get_sim_fig_path(DS_NAME, target, phase, trial_idx,
                                    seed_idx, 'sim_fcd.csv')
    emp_fc_fig_dir = get_emp_fig_dir(DS_NAME, target, 'FC')
    emp_fcd_fig_dir = get_emp_fig_dir(DS_NAME, target, 'FCD')
    range_start, range_end = SUBJECT_ID_RANGE[DS_NAME][phase]
    emp_fc_path = os.path.join(emp_fc_fig_dir,
                               f'group_emp_fc_{range_start}_{range_end}.csv')
    emp_fcd_path = os.path.join(
        emp_fcd_fig_dir, f'group_emp_fcd_{range_start}_{range_end}.csv')

    compare_fc(sim_fc_path, emp_fc_path)
    compare_fcd(sim_fcd_path, emp_fcd_path)

    # TODO: Compare TC


if __name__ == "__main__":
    ALL_TARGETS = ['all_participants']
    # * Epoch-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in [3, 6]:
    #         for seed_idx in [1]:
    #             for group_idx in [1]:
    #                 for epoch_idx in [0, 9, 19, 29, 39, 49]:
    #                     analyze_epoch(target, trial_idx, seed_idx, group_idx,
    #                                   epoch_idx)

    # * phase-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in range(31, 35):
    #         for seed_idx in range(1, 3):
    #             for phase in ['train']:
    #                 analyze_phase(target, trial_idx, seed_idx, phase)

    # # * Run-level analysis
    for target in ALL_TARGETS:
        for trial_idx in range(31, 35):
            for seed_idx in range(1, 3):
                analyze_run(target, trial_idx, seed_idx)

    # * Trial-level analysis
    # for target in ALL_TARGETS:
    #     for trial_idx in [10]:
    #         analyze_trial(target, trial_idx)

    # * Target-level analysis
    # for target in ALL_TARGETS:
    #     analyze_target(target)

    # * Others
    # plot_HCPYA_group_emp_fc_fcd(0, 343)
    # plot_HCPYA_group_emp_fc_fcd(343, 686)
    # plot_HCPYA_group_emp_fc_fcd(686, 1029)

    # compare_test_results_plot()
    # compare_group_params_apply_to_individual()
    # compare_group_test_results()
    # plot_pred_loss()

    # for epoch in [3]:
    #     for r_E_range in [[float('-inf'), float('inf')], [1, 5]]:
    #         for total_loss_range in [[float('-inf'), float('inf')], [0, 1]]:
    #             # if r_E_range == [float('-inf'), float('inf')] and total_loss_range == [float('-inf'), float('inf')]:
    #             #     continue
    #             scatter_loss_vs_r_E('HCPYA',
    #                                 'group_860_1029',
    #                                 13,
    #                                 range(1, 1001),
    #                                 None, [epoch],
    #                                 r_E_range=r_E_range,
    #                                 total_loss_range=total_loss_range,
    #                                 plot_param_ranges=True)
