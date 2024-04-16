import os
import pandas as pd
import numpy as np
import torch
import scipy.io as sio
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
import CBIG_func
from tzeng_func_torch import tzeng_KS_distance, parameterize_myelin_rsfc
import tzeng_func


def generate_sublist():
    save_param_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/EI_ratio'

    subjects_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_age = np.array(subjects_age)

    subject_id_list = []
    for sub_nbr in range(0, 885):
        final_state_path = os.path.join(save_param_dir, f'sub{sub_nbr}.pth')
        if not os.path.exists(final_state_path):
            continue
        subject_id_list.append(subjects_age[sub_nbr, 0])
    print("Valid subject number: ", len(subject_id_list))
    subject_id_file = pd.DataFrame(subject_id_list, columns=['subject_id'])
    subject_id_file.to_csv(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/subject_list.txt',
        sep='\t',
        header=False,
        index=False)
    print("Saved.")


def EI_separate():
    threshold = 220

    sub_list_above = []
    sub_list_below = []
    for sub_nbr in range(0, 885):
        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/EI_ratio/sub{sub_nbr}.pth'
        if not os.path.exists(EI_path):
            continue
        d = torch.load(EI_path)
        if torch.mean(d['ei_ratio']) < (threshold / 100):
            sub_list_below.append(sub_nbr)
        else:
            sub_list_above.append(sub_nbr)
    print("Above sub num: ", len(sub_list_above))
    print("Below sub num: ", len(sub_list_below))
    sub_list_above = np.array(sub_list_above)
    sub_list_below = np.array(sub_list_below)

    # np.save(f'/home/tzeng/storage/Python/MFMApplication/analysis/first_third/sub_above_{threshold}.npy', sub_list_above)
    # np.save(f'/home/tzeng/storage/Python/MFMApplication/analysis/first_third/sub_below_{threshold}.npy', sub_list_below)
    print(sub_list_above)
    return sub_list_above, sub_list_below


def EI_comparison_normal_hybrid():
    ei_1_list = []
    ei_2_list = []
    for sub_nbr in range(0, 885):
        EI_path_1 = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/EI_ratio/sub{sub_nbr}.pth'
        EI_path_2 = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/EI_ratio/sub{sub_nbr}.pth'
        if not os.path.exists(EI_path_1) or not os.path.exists(EI_path_2):
            continue
        d_1 = torch.load(EI_path_1)
        d_2 = torch.load(EI_path_2)
        ei_1 = torch.mean(d_1['ei_ratio'])
        ei_2 = torch.mean(d_2['ei_ratio'])

        ei_1_list.append(ei_1)
        ei_2_list.append(ei_2)
    plt.figure()
    plt.scatter(ei_1_list, ei_2_list, s=10)
    plt.xlabel('normal EI')
    plt.ylabel('hybrid EI')
    plt.savefig(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/compare_normal_hybrid_ei.png'
    )
    plt.close()
    print("Saved.")


def EI_inves():

    sub_list = np.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/sub_below_210.npy'
    )
    subjects_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_age = np.array(subjects_age)

    sub_num = len(sub_list)
    print("Subject number: ", sub_num)
    sub_mask = torch.triu(torch.ones(sub_num, sub_num, dtype=torch.bool), 1)
    fc_mask = torch.triu(torch.ones(68, 68, dtype=torch.bool), 1)
    fc_emp_list = torch.zeros(sub_num, 2278)
    emp_fcd_list = torch.zeros(sub_num, 10000)

    for i in range(len(sub_list)):
        sub_nbr = sub_list[i]
        subject_id = subjects_age[sub_nbr, 0]

        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/Individual/EI_ratio/sub{sub_nbr}.pth'
        emp_fc = pd.read_csv(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv',
            header=None,
            index_col=False)
        emp_fc = torch.as_tensor(np.array(emp_fc))
        emp_fcd_cum = sio.loadmat(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FCD/sub-{subject_id}.mat'
        )
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum['FCD_CDF'].astype(
            np.float64)).T
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        fc_emp_list[i] = emp_fc[fc_mask]
        emp_fcd_list[i] = torch.squeeze(emp_fcd_cum)

    # Corr and KS
    '''fc_corr = CBIG_corr(fc_emp_list.T)
    torch.save(fc_corr, '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fc_corr_above210.pth')'''

    fcd_ks = tzeng_KS_distance(emp_fcd_list)
    print("KS: ", fcd_ks.shape)
    torch.save(
        fcd_ks,
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fcd_ks_below210.pth'
    )
    print("Saved.")


def EI_inves_2():

    subjects_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_age = np.array(subjects_age)

    sub_list_1 = np.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/sub_above_210.npy'
    )
    sub_num = len(sub_list_1)
    print("Subject number: ", sub_num)
    fc_mask = torch.triu(torch.ones(68, 68, dtype=torch.bool), 1)
    fc_emp_list_1 = torch.zeros(sub_num, 2278)
    emp_fcd_list_1 = torch.zeros(sub_num, 10000)
    for i in range(sub_num):
        sub_nbr = sub_list_1[i]
        subject_id = subjects_age[sub_nbr, 0]

        emp_fc = pd.read_csv(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv',
            header=None,
            index_col=False)
        emp_fc = torch.as_tensor(np.array(emp_fc))
        emp_fcd_cum = sio.loadmat(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FCD/sub-{subject_id}.mat'
        )
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum['FCD_CDF'].astype(
            np.float64)).T
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        fc_emp_list_1[i] = emp_fc[fc_mask]
        emp_fcd_list_1[i] = torch.squeeze(emp_fcd_cum)

    sub_list_2 = np.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/sub_below_210.npy'
    )
    sub_num = len(sub_list_2)
    print("Subject number: ", sub_num)
    fc_mask = torch.triu(torch.ones(68, 68, dtype=torch.bool), 1)
    fc_emp_list_2 = torch.zeros(sub_num, 2278)
    emp_fcd_list_2 = torch.zeros(sub_num, 10000)
    for i in range(sub_num):
        sub_nbr = sub_list_2[i]
        subject_id = subjects_age[sub_nbr, 0]

        emp_fc = pd.read_csv(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv',
            header=None,
            index_col=False)
        emp_fc = torch.as_tensor(np.array(emp_fc))
        emp_fcd_cum = sio.loadmat(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FCD/sub-{subject_id}.mat'
        )
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum['FCD_CDF'].astype(
            np.float64)).T
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        fc_emp_list_2[i] = emp_fc[fc_mask]
        emp_fcd_list_2[i] = torch.squeeze(emp_fcd_cum)

    # Corr and KS
    '''fc_corr = CBIG_corr(fc_emp_list_1.T, fc_emp_list_2.T)
    print("FC Corr: ", fc_corr)
    torch.save(fc_corr, '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fc_corr_above_vs_below.pth')
    print("Saved.")'''

    fcd_ks = tzeng_KS_distance(emp_fcd_list_1, emp_fcd_list_2)
    print("FCD KS: ", fcd_ks)
    torch.save(
        fcd_ks,
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fcd_ks_above_vs_below.pth'
    )
    print("Saved.")


def EI_inves_3():
    metric_above = torch.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fcd_ks_above210.pth'
    )
    metric_below = torch.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fcd_ks_below210.pth'
    )
    metric_above_below = torch.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/fcd_ks_above_vs_below.pth'
    )

    above_mask = torch.eye(metric_above.shape[0], dtype=torch.bool)
    metric_above[above_mask] = float('nan')
    below_mask = torch.eye(metric_below.shape[0], dtype=torch.bool)
    metric_below[below_mask] = float('nan')

    ave_above = torch.nanmean(metric_above, dim=1)
    ave_below = torch.nanmean(metric_below, dim=1)
    ave_above_below = torch.mean(metric_above_below, dim=1)
    ave_below_above = torch.mean(metric_above_below.T, dim=1)
    statistics, p_value = stats.ttest_ind(
        torch.cat((ave_above, ave_below)),
        torch.cat((ave_above_below, ave_below_above)))
    print(f"T-test results: statistics: {statistics}; p-value: {p_value}")
    print("Average: ",
          torch.mean(ave_above).item(),
          torch.mean(ave_below).item(),
          torch.mean(ave_above_below).item())


def ei_cognitive_age():
    pnc_demo = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/pnc_demographics.csv')
    scan_ids_demo = np.array(pnc_demo['scanid'])
    re_ids = np.array(pnc_demo['reid'])

    cognitive_file = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC_behav/n1601_cnb_factor_scores_tymoore_20151006.csv'
    )
    scan_ids_cog = np.array(cognitive_file['scanid'])
    overall_accuracy = np.array(cognitive_file['Overall_Accuracy'])

    id_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    id_age = np.array(id_age)

    threshold = 220
    sub_list_1 = np.load(
        f'/home/tzeng/storage/Python/MFMApplication/analysis/first_third/sub_above_{threshold}.npy'
    )
    subjects_id_1 = id_age[sub_list_1, 0]
    overall_accuracy_1 = np.zeros((len(sub_list_1), ))
    for i in range(len(sub_list_1)):
        subject_id = subjects_id_1[i]
        scan_id = scan_ids_demo[re_ids == subject_id][0]
        overall_accuracy_1[i] = overall_accuracy[scan_ids_cog == scan_id][0]
    ages_1 = id_age[sub_list_1, 1]
    nan_mask = np.isnan(overall_accuracy_1)
    overall_accuracy_1 = overall_accuracy_1[~nan_mask]
    print("Len(1): ", len(overall_accuracy_1))

    sub_list_2 = np.load(
        f'/home/tzeng/storage/Python/MFMApplication/analysis/first_third/sub_below_{threshold}.npy'
    )
    subjects_id_2 = id_age[sub_list_2, 0]
    overall_accuracy_2 = np.zeros((len(sub_list_2), ))
    for i in range(len(sub_list_2)):
        subject_id = subjects_id_2[i]
        scan_id = scan_ids_demo[re_ids == subject_id][0]
        overall_accuracy_2[i] = overall_accuracy[scan_ids_cog == scan_id][0]
    ages_2 = id_age[sub_list_2, 1]
    nan_mask = np.isnan(overall_accuracy_2)
    overall_accuracy_2 = overall_accuracy_2[~nan_mask]
    print("Len(2): ", len(overall_accuracy_2))

    print("Ages: ", np.amax(ages_1), np.amax(ages_2))
    age_statistics, age_p_value = stats.ttest_ind(ages_1, ages_2)
    print(
        f"Age T-test results: statistics: {age_statistics}; p-value: {age_p_value}"
    )
    cog_statistics, cog_p_value = stats.ttest_ind(overall_accuracy_1,
                                                  overall_accuracy_2)
    print(
        f"Cognitive T-test results: statistics: {cog_statistics}; p-value: {cog_p_value}"
    )

    need_boxplot = False
    if need_boxplot:
        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([ages_1, ages_2],
                    labels=[f'EI_above_{threshold}', f'EI_below_{threshold}'],
                    showfliers=False)
        plt.xlabel('Different group')
        plt.ylabel('Ages / month')
        plt.title(f'Compare_Ages, p={age_p_value}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/EI',
                f'Compare_Ages_{threshold}.png'))
        plt.close()
        print("Age figure saved.")

        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([overall_accuracy_1, overall_accuracy_2],
                    labels=[f'EI_above_{threshold}', f'EI_below_{threshold}'],
                    showfliers=False)
        plt.xlabel('Different group')
        plt.ylabel('Overall accuracy')
        plt.title(f'Compare_Overall_accuracy, p={cog_p_value}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/EI',
                f'Compare_Overall_accuracy_{threshold}.png'))
        plt.close()
        print("Cog figure saved.")


def EI_ratio_individual():
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/Individual'
    group_mats = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    myelin = torch.as_tensor(group_mats['myelin_group_1029'])
    rsfc_gradient = torch.as_tensor(group_mats['rsfc_group_1029'])
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration
    euler_range = range(67, 100)
    euler_epochs = len(euler_range)

    wei_average = []
    for sub_nbr in range(0, 885):
        save_param_dir = os.path.join(parent_dir, f'train/trial1/sub{sub_nbr}')

        if not os.path.exists(os.path.join(save_param_dir, 'final_state.pth')):
            print("This subject does not have final state file.")
            continue

        param_dim = 10
        param_10_sets = torch.zeros(param_dim, euler_epochs)  # [10, 33]
        loss_sets = torch.ones(euler_epochs) * 3
        for epoch in euler_range:
            param_10_path = os.path.join(save_param_dir,
                                         f'param_save_epoch{epoch}.pth')
            d = torch.load(param_10_path)

            valid_param_indices_pre = d['valid_param_indices']
            param_10 = d['param_10']
            param_10 = param_10[:, valid_param_indices_pre]
            total_loss = d['corr_loss'] + d['L1_loss'] + d['ks_loss']  # [xxx]
            best_param_ind = torch.argmin(total_loss)
            param_10 = param_10[:, best_param_ind]  # [10]

            record_ind = epoch - euler_range[0]
            param_10_sets[:, record_ind] = param_10
            loss_sets[record_ind] = total_loss[best_param_ind]
        best_sets_ind = torch.argmin(loss_sets)
        # print("Best train loss: ", loss_sets[best_sets_ind])
        best_param_10 = param_10_sets[:, best_sets_ind].unsqueeze(1)

        best_parameter = parameterize_myelin_rsfc(myelin, rsfc_gradient,
                                                  best_param_10)
        wei = best_parameter[68:136]
        wei_average.append(torch.mean(wei).item())

    wei_average = np.array(wei_average)
    print(wei_average)
    print(np.sum(wei_average > 4.5))
    return 0


def age_match():
    pnc_demo = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/pnc_demographics.csv')
    scan_ids_demo = np.array(pnc_demo['scanid'])
    re_ids = np.array(pnc_demo['reid'])

    cognitive_file = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC_behav/n1601_cnb_factor_scores_tymoore_20151006.csv'
    )
    scan_ids_cog = np.array(cognitive_file['scanid'])
    overall_accuracy = np.array(cognitive_file['Overall_Accuracy'])

    id_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    id_age = np.array(id_age)

    valid_sub_list = []
    ei_ratio_list = []
    for sub_nbr in range(0, 885):
        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/Individual/EI_ratio/sub{sub_nbr}.pth'
        if not os.path.exists(EI_path):
            continue
        valid_sub_list.append(sub_nbr)
        d = torch.load(EI_path)
        ei_ratio_list.append(torch.mean(d['ei_ratio']).item())
    print("Valid number: ", len(valid_sub_list))

    ei_1_list = []
    ei_2_list = []
    age_1_list = []
    age_2_list = []
    overall_1_list = []
    overall_2_list = []
    for i in range(0, len(valid_sub_list), 2):
        if i + 2 >= len(valid_sub_list):
            break
        sub_nbr_1 = valid_sub_list[i]
        sub_nbr_2 = valid_sub_list[i + 1]
        subject_id_1 = id_age[sub_nbr_1, 0]
        subject_id_2 = id_age[sub_nbr_2, 0]
        age_1 = id_age[sub_nbr_1, 1]
        age_2 = id_age[sub_nbr_2, 1]
        scan_id_1 = scan_ids_demo[re_ids == subject_id_1][0]
        scan_id_2 = scan_ids_demo[re_ids == subject_id_2][0]
        overall_accuracy_1 = overall_accuracy[scan_ids_cog == scan_id_1][0]
        overall_accuracy_2 = overall_accuracy[scan_ids_cog == scan_id_2][0]
        if np.isnan(overall_accuracy_1) or np.isnan(overall_accuracy_2):
            continue
        ei_1 = ei_ratio_list[i]
        ei_2 = ei_ratio_list[i + 1]
        if overall_accuracy_1 > overall_accuracy_2:
            overall_1_list.append(overall_accuracy_1)
            overall_2_list.append(overall_accuracy_2)
            ei_1_list.append(ei_1)
            ei_2_list.append(ei_2)
            age_1_list.append(age_1)
            age_2_list.append(age_2)
        else:
            overall_1_list.append(overall_accuracy_2)
            overall_2_list.append(overall_accuracy_1)
            ei_1_list.append(ei_2)
            ei_2_list.append(ei_1)
            age_1_list.append(age_2)
            age_2_list.append(age_1)

    ei_1_list = np.array(ei_1_list)
    ei_2_list = np.array(ei_2_list)
    age_1_list = np.array(age_1_list)
    age_2_list = np.array(age_2_list)
    overall_1_list = np.array(overall_1_list)
    overall_2_list = np.array(overall_2_list)

    age_statistics, age_p_value = stats.ttest_ind(age_1_list, age_2_list)
    print('Age average: ', np.mean(age_1_list), np.mean(age_2_list))
    print(
        f"Age T-test results: statistics: {age_statistics}; p-value: {age_p_value}"
    )
    cog_statistics, cog_p_value = stats.ttest_ind(overall_1_list,
                                                  overall_2_list)
    print(
        f"Cognition T-test results: statistics: {cog_statistics}; p-value: {cog_p_value}"
    )
    ei_statistics, ei_p_value = stats.ttest_ind(ei_1_list, ei_2_list)
    print(
        f"EI T-test results: statistics: {ei_statistics}; p-value: {ei_p_value}"
    )
    print('EI average: ', np.mean(ei_1_list), np.mean(ei_2_list))

    need_boxplot = False
    if need_boxplot:
        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([age_1_list, age_2_list],
                    labels=[f'high performance', f'low performance'],
                    showfliers=False)
        plt.xlabel('Different group')
        plt.ylabel('Ages / month')
        plt.title(f'Compare_Ages, p={age_p_value:.4f}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/EI',
                f'Age_match_Compare_Ages.png'))
        plt.close()
        print("Age figure saved.")

        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([overall_1_list, overall_2_list],
                    labels=[f'high performance', f'low performance'],
                    showfliers=False)
        plt.xlabel('Different group')
        plt.ylabel('Overall accuracy')
        plt.title(f'Compare_Overall_accuracy, p={1.24e-43}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/EI',
                'Age_match_Compare_Overall_accuracy.png'))
        plt.close()
        print("Cog figure saved.")

        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([ei_1_list, ei_2_list],
                    labels=['high performance', 'low performance'],
                    showfliers=False)
        plt.xlabel('Different group')
        plt.ylabel('EI ratio')
        plt.title(f'Compare_EI_ratio, p={ei_p_value:.4f}')
        plt.savefig(
            os.path.join(
                '/home/ftian/storage/projects/MFM_exploration/reports/figures/EI',
                f'Age_match_Compare_EI.png'))
        plt.close()
        print("EI figure saved.")


def cognitive_scan_id_to_subject_id():
    pnc_demo = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/pnc_demographics.csv')
    scan_ids_demo = np.array(pnc_demo['scanid'])
    re_ids = np.array(pnc_demo['reid'])

    behavior_file = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC_behav/n1601_cnb_factor_scores_tymoore_20151006.csv'
    )
    scan_ids_cog = np.array(behavior_file['scanid'])

    subject_id_cog = []
    for i in range(len(scan_ids_cog)):
        for j in range(len(scan_ids_demo)):
            if scan_ids_cog[i] == scan_ids_demo[j]:
                subject_id_cog.append(re_ids[j])
    behavior_file.insert(0, 'subject_id', subject_id_cog)
    behavior_file.to_csv(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/behavior.csv',
        index=False)
    print("Saved.")


def motion_extract():
    subjects_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_age = np.array(subjects_age)

    sub_num = len(subjects_age)

    sub_list = []
    ave_dvars = []
    ave_framewise_displacement = []

    for sub_nbr in range(sub_num):
        subject_id = subjects_age[sub_nbr, 0]
        # Of course you can use package glob or using listdir first
        motion_path_1 = f'/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC/sub-{subject_id}/ses-PNC1/func/sub-{subject_id}_ses-PNC1_task-rest_acq-singleband_desc-confounds_timeseries.tsv'
        motion_path_2 = f'/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC/sub-{subject_id}/ses-PNC1/func/sub-{subject_id}_ses-PNC1_task-rest_acq-singlebandVARIANTNoFmap_desc-confounds_timeseries.tsv'
        motion_path_3 = f'/mnt/isilon/CSC2/Yeolab/Data/PNC/PNC/sub-{subject_id}/ses-PNC1/func/sub-{subject_id}_ses-PNC1_task-rest_acq-singlebandVARIANTObliquity_desc-confounds_timeseries.tsv'
        if not os.path.exists(motion_path_1):
            if not os.path.exists(motion_path_2):
                if not os.path.exists(motion_path_3):
                    print(subject_id)
                    continue
                else:
                    motion_file = pd.read_table(motion_path_3, index_col=False)
            else:
                motion_file = pd.read_table(motion_path_2)
        else:
            motion_file = pd.read_table(motion_path_1)
        sub_list.append(subject_id)
        ave_dvars.append(np.nanmean(motion_file['dvars']))
        ave_framewise_displacement.append(
            np.nanmean(motion_file['framewise_displacement']))
    save_file = pd.DataFrame(sub_list, columns=['subject_id'])
    save_file['ave_dvars'] = ave_dvars
    save_file['ave_framewise_displacement'] = ave_framewise_displacement
    save_file.to_csv(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/motion.csv',
        index=False)
    print("Saved.")


def motion_check():
    subjects_age = pd.read_csv(
        '/mnt/isilon/CSC2/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_age = np.array(subjects_age)

    sub_list_above = np.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/sub_above_210.npy'
    )
    sub_list_below = np.load(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/sub_below_210.npy'
    )

    motion_file = pd.read_csv(
        '/home/tzeng/storage/Python/MFMApplication/analysis/whole/motion.csv')

    sub_num_above = len(sub_list_above)
    dvars_above = []
    fd_above = []
    for i in range(sub_num_above):
        sub_nbr = sub_list_above[i]
        dvars_above.append(motion_file.loc[sub_nbr, 'ave_dvars'])
        fd_above.append(motion_file.loc[sub_nbr, 'ave_framewise_displacement'])

    sub_num_below = len(sub_list_below)
    dvars_below = []
    fd_below = []
    for i in range(sub_num_below):
        sub_nbr = sub_list_below[i]
        dvars_below.append(motion_file.loc[sub_nbr, 'ave_dvars'])
        fd_below.append(motion_file.loc[sub_nbr, 'ave_framewise_displacement'])

    tzeng_func.tzeng_2_sample_t_test_n_plot(dvars_above,
                                            dvars_below,
                                            need_boxplot=False)
    tzeng_func.tzeng_2_sample_t_test_n_plot(fd_above,
                                            fd_below,
                                            need_boxplot=False)


def generate_parameter_mat():
    group_mats = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    myelin = torch.as_tensor(group_mats['myelin_group_1029'])
    rsfc_gradient = torch.as_tensor(group_mats['rsfc_group_1029'])
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])

    param_10_array = torch.zeros(10, 885)
    valid_count = 0
    for sub_nbr in range(0, 885):
        if sub_nbr == 135:  # This subject doesn't have behavior scores
            continue
        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/EI_ratio/sub{sub_nbr}.pth'
        if not os.path.exists(EI_path):
            continue
        d = torch.load(EI_path)
        param_10_array[:, valid_count] = d['param_10'].squeeze()
        valid_count += 1
    print("Valid count: ", valid_count)
    param_10_array = param_10_array[:, :valid_count]
    parameter = parameterize_myelin_rsfc(myelin, rsfc_gradient, param_10_array)
    print(parameter.shape)
    parameter = parameter.numpy()
    save_dict = {
        'parameter': parameter,
        'wee': parameter[0:68],
        'wei': parameter[68:136],
        'sigma': parameter[137:]
    }
    sio.savemat(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/subject_parameters.mat',
        save_dict)
    print("Saved.")


def generate_emp_fc_mat():
    id_list = pd.read_csv(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/subject_list.txt',
        header=None,
        index_col=False)
    id_list = np.squeeze(np.array(id_list))

    fc_mask = np.triu(np.ones((68, 68), dtype=bool), 1)
    fc_emps = np.zeros((2278, len(id_list)))

    for i in range(len(id_list)):
        subject_id = id_list[i]
        emp_fc = pd.read_csv(
            f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_surface/FC/sub-{subject_id}.csv',
            header=None,
            index_col=False)
        emp_fc = np.array(emp_fc)
        fc_emps[:, i] = emp_fc[fc_mask]
    sio.savemat(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/emp_fc.mat',
        {'empfc': fc_emps})
    print("Saved.")


def generate_sim_fc_mat():

    fc_mask = np.triu(np.ones((68, 68), dtype=bool), 1)
    fc_sims = np.zeros((2278, 614))

    valid_count = 0
    for i in range(0, 885):
        sim_fc_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/sim_fc/sub{i}.pth'
        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual_shaoshi/EI_ratio/sub{i}.pth'
        if not os.path.exists(sim_fc_path):
            continue
        d = torch.load(sim_fc_path)
        fc_sim = d['fc'].numpy()
        fc_sims[:, valid_count] = fc_sim[fc_mask]
        valid_count += 1
    print("Valid count: ", valid_count)
    sio.savemat(
        '/home/tzeng/storage/Matlab/PNC/predictive/KRR/normal/fc_sim.mat',
        {'fc': fc_sims})
    print("Saved.")


def EI_group_check(group_nbr):
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/group'
    group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}'

    myelin = np.array(
        pd.read_csv(os.path.join(group_mats_path, 'myelin.csv'),
                    header=None,
                    index_col=False))
    myelin = torch.as_tensor(myelin)
    rsfc_gradient = np.array(
        pd.read_csv(os.path.join(group_mats_path, 'rsfc_gradient.csv'),
                    header=None,
                    index_col=False))
    rsfc_gradient = torch.as_tensor(rsfc_gradient)
    sc_mat = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'SC_validation.csv'),
                    header=None,
                    index_col=False))
    sc_mat = torch.as_tensor(sc_mat)
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    EI_save_path = os.path.join(parent_dir,
                                f'EI_ratio_before/group{group_nbr}.pth')
    d = torch.load(EI_save_path)
    param_10 = d['param_10']
    parameter = parameterize_myelin_rsfc(myelin, rsfc_gradient, param_10)
    wee = parameter[0:68]
    wei = parameter[68:136]
    # print(f'{torch.mean(wee).item():.2f}', end=' ')
    print(f'{torch.mean(wee / wei).item():.2f}', end=' ')
    # /home/tzeng/storage/Python/MFMApplication/sh_mfm_application/server_slow_logs


def param_vs_age():
    trial_nbr = 1
    test_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/group/test/trial{trial_nbr}'
    wee_wei_ratio = []
    wee_list = []
    wei_list = []
    age_list = []
    for group_nbr in range(1, 30):
        test_res_path = os.path.join(test_dir, f'group{group_nbr}',
                                     'val_results.pth')
        if not os.path.exists(test_res_path):
            continue
        test_res = torch.load(test_res_path)
        wee = test_res['parameter'][0:68, 0].numpy()
        wei = test_res['parameter'][68:136, 0].numpy()
        wee_list.append(np.mean(wee))
        wei_list.append(np.mean(wei))
        wee_wei_ratio.append(np.mean((wee / wei)))

        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        age_list.append(np.mean(age))

    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/PNC/wee_group_trial{trial_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wee_list,
                                               save_fig_path,
                                               figure_title='w_EE vs age',
                                               xlabel='Age',
                                               ylabel='w_EE')

    save_fig_path = f'/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/PNC/wei_group_trial{trial_nbr}.png'
    tzeng_func.tzeng_scatter_with_regress_line(age_list,
                                               wei_list,
                                               save_fig_path,
                                               figure_title='w_EI vs age',
                                               xlabel='Age',
                                               ylabel='w_EI')


def mean_fc_profile_vs_age():
    fc_list = np.zeros((29, 68))
    age_list = np.zeros((29, ))
    valid_count = 0
    for group_nbr in range(1, 30):
        group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}'
        emp_fc = np.array(
            pd.read_csv(os.path.join(group_mats_path, f'FC_train.csv'),
                        header=None,
                        index_col=False))
        fc_list[valid_count] = np.mean(emp_fc, axis=1)

        age_path = f'/home/shaoshi.z/storage/MFM/PNC/rest/age_results/seed_296/input/{group_nbr}/validation_subject_age.txt'
        age = np.array(pd.read_csv(age_path, header=None, index_col=False))
        age_list[valid_count] = np.mean(age)

        valid_count += 1

    corr = CBIG_func.CBIG_corr(fc_list,
                               age_list.astype(np.float64).reshape((-1, 1)))
    corr = pd.DataFrame(corr)
    corr.to_csv(
        '/home/tzeng/storage/Python/MFMApplication/usage_files/files/PNC/mean_fc_profiles_group.txt',
        sep='\t',
        header=False,
        index=False)


def final_state_check(trial_nbr):
    parent_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/perturbation/overall_acc_group/high/train/trial{trial_nbr}/seed1'
    nbr_range = np.arange(1, 15)
    flag = 0
    for i in nbr_range:
        final_state_path = os.path.join(parent_path, f'group{i}',
                                        'param_save_epoch49.pth')
        if not os.path.exists(final_state_path):
            print(f'nbr: {i}, trial nbr: {trial_nbr}')
            flag = 1
            break
    # if flag == 0:
    #     print(f'{trial_nbr}', end=',')


def residual_1():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/train/trial1/seed1'
    parent_path_2 = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/train/trial1/seed1'
    nbr_range = np.arange(0, 885)
    prefix = 'sub'

    count = 0
    for nbr in nbr_range:
        sub_path = os.path.join(parent_path, f'{prefix}{nbr}',
                                'final_state_pFIC.pth')
        sub_path_2 = os.path.join(parent_path_2, f'{prefix}{nbr}',
                                  'final_state.pth')
        if os.path.exists(sub_path) and not os.path.exists(sub_path_2):
            print(nbr, end=' ')
            count += 1
    print("\ntotal count: ", count)


def residual_2():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/train/trial1/seed1'
    parent_path_2 = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/EI_ratio_rFIC/trial1/seed1'
    nbr_range = np.arange(0, 885)
    prefix = 'sub'

    count = 0
    for nbr in nbr_range:
        sub_path = os.path.join(parent_path, f'{prefix}{nbr}',
                                'final_state.pth')
        sub_path_2 = os.path.join(parent_path_2, f'{prefix}{nbr}.pth')
        if os.path.exists(sub_path) and not os.path.exists(sub_path_2):
            print(nbr, end=' ')
            count += 1
    print("\ntotal count: ", count)


def get_subjects_low_EI_cluster():
    bad_ei_sub_list = []
    for sub_nbr in range(401, 885):
        EI_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/EI_ratio/trial2/seed1/sub{sub_nbr}.pth'
        if not os.path.exists(EI_path):
            continue
        EI_ratio = torch.load(EI_path)
        EI_ratio = torch.mean(EI_ratio['ei_ratio']).item()
        if EI_ratio < 2:
            bad_ei_sub_list.append(sub_nbr)
            print(sub_nbr, end=' ')
    print(bad_ei_sub_list)


def check_wEI_n_range_individual():
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual'
    trial_nbr = 4
    seed_nbr = 1
    save_fig_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual/figures/trial{trial_nbr}/wei_range_check.png'

    wei_min_list = []
    wei_max_list = []
    wee_list = []
    wei_list = []
    for sub_nbr in range(0, 885):
        train_final = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/sub{sub_nbr}/final_state.pth'
        )
        if not os.path.exists(train_final):
            continue
        train_final = torch.load(train_final)
        wei_range = train_final['wEI_search_range']
        wei_range_min = torch.mean(wei_range[:, 0])
        wei_range_max = torch.mean(wei_range[:, 1])

        ei_ratio = os.path.join(
            parent_dir,
            f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}/sub{sub_nbr}.pth')
        if not os.path.exists(ei_ratio):
            print(sub_nbr)
            continue
        ei_ratio = torch.load(ei_ratio)
        wee = torch.mean(ei_ratio['parameter'][0:68])
        wei = torch.mean(ei_ratio['parameter'][68:137])

        wei_min_list.append(wei_range_min)
        wei_max_list.append(wei_range_max)
        wei_list.append(wei)
        wee_list.append(wee)

    x = np.arange(len(wei_list))
    plt.figure()
    plt.scatter(x, wee_list, c='y')
    plt.scatter(x, wei_list, c='b')
    plt.scatter(x, wei_min_list, c='r')
    plt.scatter(x, wei_max_list, c='g')
    plt.savefig(save_fig_path)
    plt.close()
    print("Done.")
    return


############################################################
# Written by Tian Fang
############################################################
def gather_baseline_losses():
    """
    baseline results are stored in /home/shaoshi.z/storage/MFM/Zhang2023_pFIC_github_pr_231128/replication/PNC/age_effect/reference_output/test/{age_group}/test_all.csv
    where age group is within range(1, 30)
    the 11th to 14th (1-indexed) rows are FC corr, FC L1, FCD, and total loss, respectively.
    We want to extract these values from different groups and put them into a dictionary,
    with keys ['total_loss', 'corr_loss', 'l1_loss', 'ks_loss'], and values the corresponding losses in torch tensor of size (num_of_groups, )
    Finally, use torch.save to save the dictionary to the given save_path
    """
    save_path = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/age_group/test/trial0/seed_best_among_all/lowest_losses.pth'
    losses = {
        'total_loss': torch.zeros(29),
        'corr_loss': torch.zeros(29),
        'l1_loss': torch.zeros(29),
        'ks_loss': torch.zeros(29)
    }
    for age_group in range(1, 30):
        test_path = f'/home/shaoshi.z/storage/MFM/Zhang2023_pFIC_github_pr_231128/replication/PNC/age_effect/reference_output/test/{age_group}/test_all.csv'
        test_res = pd.read_csv(test_path, header=None, index_col=False)
        losses['corr_loss'][age_group - 1] = test_res.iloc[10, 0]
        losses['l1_loss'][age_group - 1] = test_res.iloc[11, 0]
        losses['ks_loss'][age_group - 1] = test_res.iloc[12, 0]
        losses['total_loss'][age_group - 1] = test_res.iloc[13, 0]
    torch.save(losses, save_path)
    print("Saved.")


def gather_baseline_losses_cognition():
    """
    baseline results are stored in /home/shaoshi.z/storage/MFM/Zhang2023_pFIC_github_pr_231128/replication/PNC/cognition_effect/reference_output/{perf_group}_performance/test/{group}/test_all.csv
    where group is within range(1, 15), and perf_group is either 'low' or 'high'
    the 11th to 14th (1-indexed) rows are FC corr, FC L1, FCD, and total loss, respectively.
    We want to extract these values from different groups and put them into a dictionary,
    with keys ['total_loss', 'corr_loss', 'l1_loss', 'ks_loss'], and values the corresponding losses in torch tensor of size (num_of_groups, )
    Finally, use torch.save to save the dictionary to the given save_path
    """

    for perf_group in ['low', 'high']:
        losses = {
            'total_loss': torch.zeros(14),
            'corr_loss': torch.zeros(14),
            'l1_loss': torch.zeros(14),
            'ks_loss': torch.zeros(14)
        }
        save_path = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/{perf_group}/test/trial0/seed_best_among_all/lowest_losses.pth'
        # create parent dir of save_path is necessary
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for group in range(1, 15):
            test_path = f'/home/shaoshi.z/storage/MFM/Zhang2023_pFIC_github_pr_231128/replication/PNC/cognition_effect/reference_output/{perf_group}_performance/test/{group}/test_all.csv'
            test_res = pd.read_csv(test_path, header=None, index_col=False)
            losses['corr_loss'][group - 1] = test_res.iloc[10, 0]
            losses['l1_loss'][group - 1] = test_res.iloc[11, 0]
            losses['ks_loss'][group - 1] = test_res.iloc[12, 0]
            losses['total_loss'][group - 1] = test_res.iloc[13, 0]

        torch.save(losses, save_path)
    print("Saved.")


if __name__ == "__main__":
    gather_baseline_losses_cognition()
