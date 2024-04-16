import numpy as np
import os
import pandas as pd
import scipy.io as spio
import torch
import sys

sys.path.append('/home/tzeng/storage/Python/MFMmodel')
from mfm_model_2014 import MfmModel2014

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
import tzeng_func


def residual_1():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/train/trial2/split1'
    parent_path_2 = '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/train/trial2/split1'
    nbr_range = np.arange(0, 45)
    prefix = 'sub'

    count = 0
    for nbr in nbr_range:
        sub_path = os.path.join(parent_path, f'{prefix}{nbr}')
        sub_path_2 = os.path.join(parent_path_2, f'{prefix}{nbr}',
                                  'final_state.pth')
        if os.path.exists(sub_path) and not os.path.exists(sub_path_2):
            print(nbr, end=' ')
            count += 1
    print("\ntotal count: ", count)


def residual_2():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/train/trial23/split1'
    parent_path_2 = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual/validation/trial23/split1'
    nbr_range = np.arange(860, 1029)
    prefix = 'sub'

    count = 0
    for nbr in nbr_range:
        sub_path = os.path.join(parent_path, f'{prefix}{nbr}',
                                'final_state.pth')
        sub_path_2 = os.path.join(parent_path_2, f'{prefix}{nbr}')
        if os.path.exists(sub_path) and not os.path.exists(sub_path_2):
            print(nbr, end=' ')
            count += 1
    print("\ntotal count: ", count)


def residual_3():
    parent_path = '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/validation/trial2/split1'
    parent_path_2 = '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/test/trial2/split1'
    count = 0
    for sub_nbr in range(0, 45):
        sub_path = os.path.join(parent_path, f'sub{sub_nbr}')
        sub_path_2 = os.path.join(parent_path_2, f'sub{sub_nbr}',
                                  'test_results.pth')
        if os.path.exists(sub_path) and not os.path.exists(sub_path_2):
            print(sub_nbr, end=' ')
            count += 1
    print("\ntotal count: ", count)


def group_sim_fc_fcd_to_individual(subject_id):

    fc_prev_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rfMRI_FC_ICAFIX_4runs'
    fc_prev_path = os.path.join(fc_prev_dir, f'{subject_id}.mat')
    fc_prev_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fc_prev_path,
        ['fc_REST1_LR', 'fc_REST1_RL', 'fc_REST2_LR', 'fc_REST2_RL'])
    if fc_prev_runs is None:
        return

    fcd_prev_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rfMRI_FCD_CDF_ICAFIX_4runs'
    fcd_prev_path = os.path.join(fcd_prev_dir, f'{subject_id}.mat')
    fcd_prev_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fcd_prev_path, [
            'fcd_cdf_REST1_LR', 'fcd_cdf_REST1_RL', 'fcd_cdf_REST2_LR',
            'fcd_cdf_REST2_RL'
        ])
    if fcd_prev_runs is None:
        return

    fc_retest_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/HCP_Retest/rfMRI_FC_ICAFIX'
    fc_retest_path = os.path.join(fc_retest_dir, f'{subject_id}.mat')
    fc_retest_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fc_retest_path,
        ['fc_REST1_LR', 'fc_REST1_RL', 'fc_REST2_LR', 'fc_REST2_RL'])
    if fc_retest_runs is None:
        return

    fcd_retest_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/HCP_Retest/rfMRI_FCD_CDF_ICAFIX'
    fcd_retest_path = os.path.join(fcd_retest_dir, f'{subject_id}.mat')
    fcd_retest_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fcd_retest_path, [
            'fcd_cdf_REST1_LR', 'fcd_cdf_REST1_RL', 'fcd_cdf_REST2_LR',
            'fcd_cdf_REST2_RL'
        ])
    if fcd_retest_runs is None:
        return

    emp_fc = np.array([
        fc_retest_runs['fc_REST1_LR'], fc_retest_runs['fc_REST1_RL'],
        fc_retest_runs['fc_REST2_LR'], fc_retest_runs['fc_REST2_RL']
    ])
    emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)
    emp_fc = torch.as_tensor(emp_fc)

    emp_fcd_cum = fcd_retest_runs['fcd_cdf_REST1_LR'] + fcd_retest_runs[
        'fcd_cdf_REST1_RL'] + fcd_retest_runs[
            'fcd_cdf_REST2_LR'] + fcd_retest_runs['fcd_cdf_REST2_RL']
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    group_same_fc_fcd = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_340/simulate/sim_results_1seed.pth'
    )

    fc_sim = group_same_fc_fcd['fc']
    fcd_pdf = group_same_fc_fcd['fcd_pdf']
    total_loss, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
        fc_sim, fcd_pdf, emp_fc, emp_fcd_cum)
    save_loss = torch.hstack(
        (corr_loss.unsqueeze(1), L1_loss.unsqueeze(1), ks_loss.unsqueeze(1)))

    return save_loss


def group_sim_fc_fcd_to_individual_main():
    sublist = pd.read_csv(
        '/home/tzeng/storage/Matlab/HCPS1200/txt_files/HCP_Retest_sublist_45.txt',
        sep='\t',
        header=None,
        index_col=False)
    sublist = np.squeeze(np.array(sublist))

    save_losses = torch.ones((45, 10, 3)) * float('nan')
    nbr_range = np.arange(0, 45, 1)
    for sub_nbr in nbr_range:
        subject_id = sublist[sub_nbr]
        save_loss = group_sim_fc_fcd_to_individual(subject_id)
        if save_loss is not None:
            save_losses[sub_nbr] = save_loss
    torch.save(
        save_losses,
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/loss_group340_to_subj_retest_1seed.pth'
    )


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
        '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/loss_group340_to_subj_retest_1seed.pth'
    )
    total_loss = torch.sum(group_losses, dim=2)

    for i in range(len(group_losses)):
        sub_nbr = i
        if torch.isnan(group_losses[i]).any():
            continue
        corr_loss_group.append(group_losses[i, 0, 0].item())
        l1_loss_group.append(group_losses[i, 0, 1].item())
        ks_loss_group.append(group_losses[i, 0, 2].item())
        all_loss_group.append(total_loss[i, 0].item())
        # corr_loss_group.append(torch.mean(group_losses[i, :, 0]).item())
        # l1_loss_group.append(torch.mean(group_losses[i, :, 1]).item())
        # ks_loss_group.append(torch.mean(group_losses[i, :, 2]).item())
        # all_loss_group.append(torch.mean(total_loss[i]).item())

        path_sub = f'/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/test/trial2/split1/sub{sub_nbr}/test_results.pth'
        if not os.path.exists(path_sub):
            continue
        res_sub = torch.load(path_sub)
        corr_loss_individual.append(res_sub['corr_loss'][0].item())
        l1_loss_individual.append(res_sub['l1_loss'][0].item())
        ks_loss_individual.append(res_sub['ks_loss'][0].item())
        all_loss_individual.append(res_sub['corr_loss'][0].item() +
                                   res_sub['l1_loss'][0].item() +
                                   res_sub['ks_loss'][0].item())
        # corr_loss_individual.append(torch.mean(res_sub['corr_loss']).item())
        # l1_loss_individual.append(torch.mean(res_sub['l1_loss']).item())
        # ks_loss_individual.append(torch.mean(res_sub['ks_loss']).item())
        # all_loss_individual.append(torch.mean(res_sub['corr_loss'] + res_sub['l1_loss'] + res_sub['ks_loss']).item())

    print("Valid count: ", len(all_loss_group))
    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPretest/individual/figures/compare_group_apply_to_individual'
    fig_name = 'group_params_apply_to_subj_retest_best_val_1_seed'
    labels = ['Group', 'Individual']
    need_plot = True
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        all_loss_group,
        all_loss_individual,
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_all_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Total Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        corr_loss_group,
        corr_loss_individual,
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_corr_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='Corr Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        l1_loss_group,
        l1_loss_individual,
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_L1_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='L1 Loss')
    tzeng_func.tzeng_2_sample_t_test_n_plot(
        ks_loss_group,
        ks_loss_individual,
        need_plot,
        labels=labels,
        save_fig_path=f'{save_dir}/{fig_name}_KS_loss.png',
        fig_title=fig_name,
        xlabel='Method',
        ylabel='KS Loss')


if __name__ == "__main__":
    residual_1()
