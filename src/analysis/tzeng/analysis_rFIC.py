import torch
import sys
import scipy.io as spio
import scipy.stats as stats
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append('/home/tzeng/storage/Python/MFMApplication/scripts')
import analysis_functions

sys.path.append('/home/tzeng/storage/Python/UtilsTzeng')
import CBIG_func
import tzeng_func


def plot_pred_loss(epochs, param_save_dir, figure_path):

    corr_list = []
    l1_list = []
    ks_list = []
    for i in range(epochs):
        d = torch.load(os.path.join(param_save_dir,
                                    f'param_save_epoch{i}.pth'))
        if 'corr_loss' in d:
            corr_loss = d['corr_loss']
            l1_loss = d['L1_loss']
            ks_loss = d['ks_loss']
        elif 'pred_loss' in d:
            pred_all_losses = d['pred_loss']
            corr_loss = pred_all_losses[:, 0]
            l1_loss = pred_all_losses[:, 1]
            ks_loss = pred_all_losses[:, 2]
        elif 'FC_FCD_loss' in d:
            pred_all_losses = d['FC_FCD_loss']
            corr_loss = pred_all_losses[:, 0]
            l1_loss = pred_all_losses[:, 1]
            ks_loss = pred_all_losses[:, 2]

        corr_list.append(torch.mean(corr_loss).item())
        l1_list.append(torch.mean(l1_loss).item())
        ks_list.append(torch.mean(ks_loss).item())

    x = np.arange(0, epochs, 1)
    plt.figure()
    plt.plot(x, corr_list, 'r-', label='Corr loss')
    plt.plot(x, l1_list, 'b-', label='L1 loss')
    plt.plot(x, ks_list, 'g-', label='KS loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title(f'3 losses')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()
    print('Saved.')


def compare_group_test_results_2lists():
    res_1 = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/test/trial6/test_results.pth'
    )
    res1_total_loss = res_1['corr_loss'] + res_1['l1_loss'] + res_1['ks_loss']
    res_2 = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/test/trial7/test_results.pth'
    )
    res2_total_loss = res_2['corr_loss'] + res_2['l1_loss'] + res_2['ks_loss']

    save_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/figures/compare_hybrid_normal'
    fig_name = 'Shaoshi_1_vs_Shaoshi_5'
    labels = ['Shaoshi_1', 'Shaoshi_5']
    print("Res 1 min loss: ", torch.min(res1_total_loss))
    print("Res 2 min loss: ", torch.min(res2_total_loss))
    need_plot = False
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


def compare_group_test_results_many_lists():
    total_lists = []
    corr_lists = []
    l1_lists = []
    ks_lists = []

    path_lists = [
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial3/seed1/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial3/seed5/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial1/seed1/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial1/seed5/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial4/seed1/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/test/trial4/seed5/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/test/trial1/test_results.pth',
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/group_340/test/trial5/test_results.pth'
    ]
    for path in path_lists:
        res = torch.load(path)
        res_total_loss = res['corr_loss'] + res['l1_loss'] + res['ks_loss']
        total_lists.append(res_total_loss)
        corr_lists.append(res['corr_loss'])
        l1_lists.append(res['l1_loss'])
        ks_lists.append(res['ks_loss'])

    save_fig_dir = '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/figures/compare_50_50_5seeds'
    if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)
    labels = [
        '50 DL pFIC+\n50 Shaoshi rFIC\n1 seed',
        '50 DL pFIC+\n50 Shaoshi rFIC\n5 seeds',
        '50 Shaoshi pFIC+\n50 Shaoshi rFIC\n1 seed',
        '50 Shaoshi pFIC+\n50 Shaoshi rFIC\n5 seeds',
        '50 DL pFIC+\n50 Shaoshi pFIC\n1 seed',
        '50 DL pFIC+\n50 Shaoshi pFIC\n5 seeds', '100 Shaoshi pFIC\n1 seed',
        '100 Shaoshi pFIC\n5 seeds'
    ]
    plt.rcParams.update({'font.size': 5})
    tzeng_func.tzeng_boxplot(total_lists,
                             labels,
                             save_fig_path=f'{save_fig_dir}/total_loss.png',
                             positions=None,
                             fig_title='HCPYA group total loss comparisons',
                             xlabel='Methods',
                             ylabel='Total loss')
    tzeng_func.tzeng_boxplot(corr_lists,
                             labels,
                             save_fig_path=f'{save_fig_dir}/corr_loss.png',
                             positions=None,
                             fig_title='HCPYA group corr loss comparisons',
                             xlabel='Methods',
                             ylabel='Corr loss')
    tzeng_func.tzeng_boxplot(l1_lists,
                             labels,
                             save_fig_path=f'{save_fig_dir}/l1_loss.png',
                             positions=None,
                             fig_title='HCPYA group L1 loss comparisons',
                             xlabel='Methods',
                             ylabel='L1 loss')
    tzeng_func.tzeng_boxplot(ks_lists,
                             labels,
                             save_fig_path=f'{save_fig_dir}/ks_loss.png',
                             positions=None,
                             fig_title='HCPYA group KS loss comparisons',
                             xlabel='Methods',
                             ylabel='KS loss')


def save_parameters_to_matfile():
    pFIC_sample = '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/validation/trial1/seed1/best_param49.pth'
    pFIC_sample = torch.load(pFIC_sample)
    spio.savemat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rFIC/group/pFIC49.mat',
        {'parameter': pFIC_sample['parameter'].numpy()})
    rFIC_sample = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/validation/trial1/seed1/best_param99.pth'
    )
    spio.savemat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rFIC/group/rFIC99.mat',
        {'parameter': rFIC_sample['parameter'].numpy()})
    print("Done.")
    return


def corr_parameters():
    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    rsfc_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'rsfc_roi_1029.mat'))
    rsfc_indi = rsfc_indi['rsfc_roi_1029']
    rsfc_gradient = np.nanmean(rsfc_indi[0:340], axis=0)
    rsfc_gradient = torch.as_tensor(rsfc_gradient).unsqueeze(1).numpy()

    pFIC_sample = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/validation/trial1/seed1/best_param49.pth'
    )
    co = CBIG_func.CBIG_corr(pFIC_sample['parameter'].numpy()[137:],
                             rsfc_gradient)
    print(co)
    rFIC_sample = torch.load(
        '/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/validation/trial1/seed1/best_param99.pth'
    )
    co = CBIG_func.CBIG_corr(rFIC_sample['parameter'].numpy()[137:],
                             rsfc_gradient)
    print(co)


def plot_main():
    trial_nbr = 4
    param_save_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/train/trial{trial_nbr}'
    figure_path = f'/home/ftian/storage/projects/MFM_exploration/logs/params/HCPYAParams/rFIC_group/figures/train_curve/trial{trial_nbr}.png'
    plot_pred_loss(epochs=100,
                   param_save_dir=param_save_dir,
                   figure_path=figure_path)


if __name__ == "__main__":
    compare_group_test_results_many_lists()
