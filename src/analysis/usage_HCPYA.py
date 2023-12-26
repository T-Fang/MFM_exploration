import os
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import usage_functions
import sys

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
sys.path.append('/home/tzeng/storage/Python/MFMApplication/scripts')
from Hybrid_CMA_ES_classes import DLVersionCMAESForward, DLVersionCMAESValidator, DLVersionCMAESTester, get_EI_ratio, simulate_fc_fcd
from tzeng_func_torch import parameterize_myelin_rsfc
import tzeng_func
import CBIG_func

sys.path.append('/home/tzeng/storage/Python/MFMmodel')
from mfm_model_2014_general import MfmModel2014


def final_state_check():
    train_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/DL_dataset/Yan100/train/trial1/seed1'
    count = 0
    for nbr in range(0, 64):
        train_final_state = os.path.join(train_dir,
                                         f'group{nbr}/final_state.pth')
        if not os.path.exists(train_final_state):
            print(f"{nbr}", end=' ')
            count += 1
    print(f'\nin total {count}.')


def roi_fc_to_network_fc():
    # Reorganize index list and network length list
    network_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yeo17_fslr32.mat')
    network_dlabel = network_dlabel['dlabel']
    roi_dlabel = sio.loadmat(
        '/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
    roi_dlabel = roi_dlabel['dlabel']

    sublist = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/subject_1029.mat'
    )
    sublist = np.squeeze(sublist['subject_1029'])

    roi_fc_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/within_between_FC/Yan400_Yeo7_rfMRI_ICAFIX'
    save_roi2network_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/within_between_FC/Yan400_Yeo17_roi2network'

    usage_functions.roi_fc_2_network_fc(network_dlabel, roi_dlabel, sublist,
                                        roi_fc_dir, save_roi2network_dir)
    print("Done.")


def check_within_between_fc():
    roi2network_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/within_between_FC/Yan400_Yeo17_roi2network'
    id_age_file = pd.read_csv(
        '/home/tzeng/storage/Matlab/HCPS1200/txt_files/HCPYA_ages_1029.txt',
        sep='\t',
        header=0,
        index_col=False)
    save_fig_dir = '/home/ftian/storage/projects/MFM_exploration/reports/figures/HCPYA'
    fig_postfix = '_ICAFIX_Yan400_Yeo17'

    usage_functions.check_within_between_fc(id_age_file, roi2network_dir,
                                            save_fig_dir, fig_postfix)


def remove_empty_dir():
    test_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/individual/test'
    for sub_nbr in range(687, 901):
        sub_dir = os.path.join(test_dir, f'sub{sub_nbr}')
        if os.path.exists(sub_dir):
            dir_content = os.listdir(sub_dir)
            if len(dir_content) == 0:
                print(f"{sub_nbr} Empty.")
                os.rmdir(sub_dir)


def compare_group_individual():
    group_list = []
    indi_list = []
    for sub_nbr in range(684, 901):
        group_path = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/test_groupmats/trial1/sub{sub_nbr}/test_results.pth'
        if not os.path.exists(group_path):
            continue
        b = torch.load(group_path)
        if len(b['corr_loss']) > 0:
            a = torch.load(
                f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/individual/test/sub{sub_nbr}/test_results.pth'
            )
            indi_list.append(
                torch.min(a['corr_loss'] + a['l1_loss'] +
                          a['ks_loss']).numpy())
            group_list.append(
                (b['corr_loss'] + b['l1_loss'] + b['ks_loss']).numpy()[0])
    df = pd.DataFrame([])
    df['group'] = group_list
    df['individual'] = indi_list
    df.to_csv(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/files/compare.txt',
        sep='\t',
        header=True,
        index=False)
    # group_list = np.array(group_list)
    # indi_list = np.array(indi_list)
    # print(np.sum(group_list > indi_list))  # 123
    # print(np.sum((group_list <= indi_list)))  # 19
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        group_list,
        indi_list,
        need_boxplot=True,
        labels=['group', 'individual'],
        save_fig_path=
        '/home/tzeng/storage/Python/MFMApplication/usage_files/figures2/HCPYA/compare_group_indi.png',
        fig_title='Compare Group Individual',
        xlabel='',
        ylabel='Loss')


def visualize_FCD():
    epoch = 99
    fcd_mat = torch.load(
        f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/sim_fc_fcd/fcd/val_epoch{epoch}.pth'
    )
    fcd_mat = fcd_mat['fcd'].numpy()
    for i in range(3):
        tmp_fcd = fcd_mat[i]
        save_fig_path = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/sim_fc_fcd/visualize_fcd/e{epoch}_{i}.png'
        tzeng_func.tzeng_visualize_FCD(tmp_fcd, save_fig_path)


def check_test_results_params():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/test/trial9/split1'
    count = 0
    rec = np.zeros((106, 10))
    for sub_nbr in range(860, 1029):
        sub_path = os.path.join(parent_path, f'sub{sub_nbr}',
                                'test_results.pth')
        if os.path.exists(sub_path):
            param = torch.load(sub_path)
            param = param['param_10'][:, 0]
            rec[count] = param.numpy()
            count += 1
    print(f'\n{count}')
    corr = CBIG_func.CBIG_corr(rec.T)
    print(corr)
    print(np.mean(np.abs(corr)))


def read_train_logs(trial_nbr):
    subnbr_list = []
    randseed_list = []
    fail_count = 0
    for sub_nbr in range(862, 1029):
        # for sub_nbr in [873,880,883,884,897,899,913,917,919,936,938,943,944,945,950,952,954,961,963,965,966,978,983,986,990,993,1009,1011,1020,1026]:

        out_file_path = f'/home/tzeng/storage/Python/MFMApplication/logs/hybrid_hcpya_logs/individual/train/trial{trial_nbr}/s{sub_nbr}_out.log'
        if not os.path.exists(out_file_path):
            continue
        randseed_count = 0
        end_flag = 0
        with open(out_file_path, 'r') as f:
            for line in f:
                if "Epoch: [1/100]" in line:
                    randseed_count += 1
                if "Epoch: [100/100]" in line:
                    end_flag = 1
        if randseed_count == 0:
            continue
        if end_flag == 0:
            randseed_count = -randseed_count
            fail_count += 1
        subnbr_list.append(sub_nbr)
        randseed_list.append(randseed_count)
    print("subjects in total: ", len(subnbr_list))
    print("Failed: ", fail_count)
    record_file = pd.DataFrame([])
    record_file['sub_list'] = subnbr_list
    record_file['randseed_count'] = randseed_list
    record_file.to_csv(
        f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial{trial_nbr}_randseed_.txt',
        sep=' ',
        header=False,
        index=False)
    return 0


def combine_train_logs(origin_txt_path, new_txt_path, save_path):
    # log_dir = f'/home/tzeng/storage/Python/MFMApplication/logs/hybrid_hcpya_logs/individual/train/trial{new_trial_nbr}'
    origin_txt = pd.read_csv(origin_txt_path,
                             sep=' ',
                             header=None,
                             index_col=False)
    new_txt = pd.read_csv(new_txt_path, sep=' ', header=None, index_col=False)
    for i in range(len(origin_txt)):
        if origin_txt.iloc[i, 1] >= 0:
            continue
        for j in range(len(new_txt)):
            if new_txt.iloc[j, 0] == origin_txt.iloc[i, 0]:
                if new_txt.iloc[j, 1] < 0:
                    origin_txt.iloc[
                        i, 1] = origin_txt.iloc[i, 1] + new_txt.iloc[j, 1]
                else:
                    origin_txt.iloc[
                        i, 1] = -origin_txt.iloc[i, 1] + new_txt.iloc[j, 1]
                break

    origin_txt.to_csv(save_path, sep=' ', header=False, index=False)
    return 0


def stat_random_seed(trial_nbr_list):
    trial_9 = pd.read_csv(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial9_randseed.txt',
        sep=' ',
        header=None,
        index_col=False)
    randseed_array = np.ones((len(trial_9), ))
    for trial_nbr in trial_nbr_list:
        trial_randseed_file = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial{trial_nbr}_randseed.txt'
        trial_randseed_file = pd.read_csv(trial_randseed_file,
                                          sep=' ',
                                          header=None,
                                          index_col=False)
        for i in range(len(trial_9)):
            if trial_randseed_file.iloc[i, 1] > 0:
                randseed_array[i] += 1
    print(len(randseed_array), randseed_array)
    for j in range(len(trial_nbr_list) + 2):
        print(j, len(randseed_array[randseed_array == j]))


def check_logfile_validation(trial_nbr):
    for sub_nbr in range(860, 1029):
        log_dir = f'/home/tzeng/storage/Python/MFMApplication/logs/hybrid_hcpya_logs/individual/validation/sub{sub_nbr}'
        if not os.path.exists(log_dir):
            return
        loglist = np.arange(0, 100)
        empty_list, _, nonexist_list = tzeng_func.tzeng_check_logfile_empty(
            log_dir,
            loglist,
            logfile_prefix='e',
            logfile_postfix=f't{trial_nbr}_out.log')
        if len(empty_list) > 0:
            print(sub_nbr, trial_nbr)
            print(empty_list)
            # print(nonexist_list)


def check_logfile_test(trial_nbr):
    log_dir = f'/home/tzeng/storage/Python/MFMApplication/logs/hybrid_hcpya_logs/individual/test'
    if not os.path.exists(log_dir):
        return
    loglist = np.arange(862, 1029)
    empty_list, _, nonexist_list = tzeng_func.tzeng_check_logfile_empty(
        log_dir,
        loglist,
        logfile_prefix='s',
        logfile_postfix=f't{trial_nbr}_out.log')
    if len(empty_list) > 0:
        print(empty_list)


def select_best_from_validation():
    group_mats = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    myelin = torch.as_tensor(group_mats['myelin_group_1029'])
    rsfc_gradient = torch.as_tensor(group_mats['rsfc_group_1029'])
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    val_dirs = [
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/val/trial1'
    ]
    test_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/val/best_from_trial1'
    tester = DLVersionCMAESTester(val_dirs, test_dir, 'HCP')
    tester.select_best_from_val(myelin, rsfc_gradient)


def compare_train_logs_main(trial_nbr):
    combine_train_logs(
        f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial{trial_nbr}_randseed.txt',
        f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial{trial_nbr}_randseed_.txt',
        f'/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/record_files/trial{trial_nbr}_randseed.txt'
    )


def group_sim_fc_fcd_to_individual(sub_nbr, group_same_fc_fcd):

    sub_nbr = int(sub_nbr)

    fc_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/FC_rest'
    fc_path = os.path.join(fc_dir, f's{sub_nbr + 1}_fc.mat')
    if not os.path.exists(fc_path):
        print(f"FC not exist.")
        return
    fc_runs = sio.loadmat(fc_path)
    for key in ['fc_REST1_LR', 'fc_REST1_RL', 'fc_REST2_LR', 'fc_REST2_RL']:
        if key not in fc_runs:
            print(f"Valid FC runs are not 4.")
            return

    fcd_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/FCD'
    fcd_path = os.path.join(fcd_dir, f's{sub_nbr + 1}_fcd_cdf.mat')
    if not os.path.exists(fcd_path):
        print(f"FCD not exist.")
        return
    fcd_runs = sio.loadmat(fcd_path)
    for key in ['fcd_cdf_1', 'fcd_cdf_2', 'fcd_cdf_3', 'fcd_cdf_4']:
        if key not in fcd_runs:
            print(f"Valid FCD runs are not 4.")
            return

    # fc_emp = np.array([fc_runs['fc_REST2_LR'], fc_runs['fc_REST2_RL']])
    # fc_emp = tzeng_func.tzeng_fisher_average(fc_emp)
    fc_emp = np.array(fc_runs['fc_REST2_RL'])
    fc_emp = torch.as_tensor(fc_emp)
    # emp_fcd_cum = (fcd_runs['fcd_cdf_3'] + fcd_runs['fcd_cdf_4'])
    emp_fcd_cum = fcd_runs['fcd_cdf_4']
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    fc_sim = group_same_fc_fcd['fc']
    fcd_pdf = group_same_fc_fcd['fcd_pdf']
    total_loss, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
        fc_sim, fcd_pdf, fc_emp, emp_fcd_cum)
    save_loss = torch.hstack(
        (corr_loss.unsqueeze(1), L1_loss.unsqueeze(1), ks_loss.unsqueeze(1)))

    return save_loss


def group_sim_fc_fcd_to_individual_main():
    group_same_fc_fcd = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/trial3/seed1/sim_results.pth'
    )
    save_losses = torch.zeros((109, 10, 3))
    sub_109 = [
        862, 863, 864, 865, 866, 868, 869, 870, 871, 872, 873, 874, 875, 877,
        879, 881, 883, 885, 888, 889, 890, 891, 892, 895, 896, 897, 900, 901,
        902, 904, 905, 907, 911, 912, 914, 916, 918, 919, 921, 924, 925, 929,
        930, 931, 933, 935, 936, 937, 938, 939, 941, 943, 944, 945, 946, 949,
        950, 951, 952, 953, 954, 955, 956, 960, 962, 964, 966, 967, 968, 969,
        970, 971, 972, 973, 974, 976, 977, 979, 981, 983, 984, 986, 987, 988,
        989, 990, 991, 993, 994, 996, 998, 999, 1001, 1002, 1003, 1004, 1006,
        1008, 1014, 1016, 1017, 1019, 1021, 1022, 1023, 1024, 1025, 1026, 1028
    ]
    for s_nbr in range(len(sub_109)):
        save_losses[s_nbr] = group_sim_fc_fcd_to_individual(
            sub_109[s_nbr], group_same_fc_fcd)
    torch.save(
        save_losses,
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/trial3/seed1/loss_group_to_subj_2RLtest_MAE.pth'
    )


def compute_KS_distance():
    fcd_mat = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/compare_group_individual/group/sim_fc_fcd/fcd/val_epoch99.pth'
    )
    fcd_mat = fcd_mat['fcd']
    window_num = fcd_mat.shape[1]
    bins = 10000
    fcd_mask = torch.triu(torch.ones(window_num, window_num, dtype=torch.bool),
                          1)
    fcd_vec = fcd_mat[:, fcd_mask]
    fcd_hist = torch.ones(bins, fcd_mat.shape[0])

    for hist_i in range(fcd_mat.shape[0]):
        fcd_hist[:, hist_i] = torch.histc(fcd_vec[hist_i],
                                          bins=bins,
                                          min=-1.,
                                          max=1.)
    fcd_cdf = torch.cumsum(fcd_hist, dim=0)
    fcd_cdf = fcd_cdf / fcd_cdf[-1:, :]

    ks_loss = tzeng_func.tzeng_KS_distance(fcd_cdf.numpy().T)
    print(ks_loss)


def visualize_FC():
    # fc_1029 = sio.loadmat('/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fc_roi_1029.mat')
    # fc_1029 = fc_1029['fc_roi_1029']
    # fc_emp = np.array(fc_1029[860:1029])
    # fc = tzeng_func.tzeng_fisher_average(fc_emp)
    fc = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/trial1/sim_results_5seeds.pth'
    )
    fc = fc['fc'][0].numpy()
    plt.imshow(fc)
    plt.savefig(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/figures/visualize_fc_fcd/fc_sim_trial1_5seeds.png'
    )
    plt.close()


def simulate_results_loss_analysis():
    subrange = [0, 340]
    fc_1029 = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fc_roi_1029.mat'
    )
    fc_1029 = fc_1029['fc_roi_1029']
    fc_emp = np.array(fc_1029[subrange[0]:subrange[1]])
    fc_emp = tzeng_func.tzeng_fisher_average(fc_emp)
    fc_emp = torch.as_tensor(fc_emp)
    fcd_1029 = sio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fcd_cum_1029.mat'
    )
    fcd_1029 = fcd_1029['fcd_cum_1029']
    emp_fcd_cum = torch.as_tensor(fcd_1029[subrange[0]:subrange[1]].astype(
        np.float64))
    emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
    emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)

    sim_res = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/trial3/seed1/sim_results_mats.pth'
    )
    print(sim_res['fcd_pdf'].shape)
    fc_sim = sim_res['fc']
    fc_losses = np.zeros((10, 3))
    for i in range(10):
        _, fc_losses[i] = MfmModel2014.FC_correlation_n_L1_cost(
            fc_sim[i], fc_emp)
    print(fc_losses)
    # fcd_pdf_sim  = sim_res['fcd_pdf']
    # fcd_losses = np.zeros((10, 3))
    # for i in range(10):
    #     fcd_losses[i] = MfmModel2014.KS_cost(fcd_pdf_sim[i], emp_fcd_cum)
    # print(fcd_losses)


if __name__ == "__main__":
    final_state_check()
