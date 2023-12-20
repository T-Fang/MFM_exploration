import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import scipy.stats as stats


def plot_train_val_loss(loss_file_dir, loss_file_path):
    loss_file = pd.read_csv(os.path.join(loss_file_dir,
                                         loss_file_path + '.csv'),
                            sep=',',
                            header=0,
                            index_col=False)
    epochs = np.arange(0, len(loss_file), 1)
    train_loss = loss_file['train_loss']
    val_loss = loss_file['val_loss']
    plt.figure()
    plt.plot(epochs, train_loss, 'r-', label='Train loss')
    plt.plot(epochs, val_loss, 'g-', label='Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(loss_file_path.split('/')[-1])
    plt.legend()
    plt.savefig(os.path.join(loss_file_dir, loss_file_path + '.png'))
    plt.close()
    print('Figure saved.')


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
        else:
            raise Exception('Key error.')

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
    plt.title('3 losses')
    plt.legend()
    plt.savefig(figure_path)
    plt.close()
    print('Saved.')


def check_dl_version_test_results_one_epoch(train_results_dir,
                                            val_results_dir,
                                            epoch,
                                            save_compare_figure=False,
                                            save_figure_dir=None):
    count = 0
    mse_corr_total = 0
    mse_l1_total = 0
    mse_ks_total = 0
    corr_dl_list = []
    corr_euler_list = []
    l1_dl_list = []
    l1_euler_list = []
    ks_dl_list = []
    ks_euler_list = []

    d_dl = torch.load(
        os.path.join(train_results_dir, f'param_save_epoch{epoch}.pth'))

    test_results_path = os.path.join(val_results_dir,
                                     f'Euler_epoch{epoch}.pth')

    if os.path.exists(test_results_path):
        d = torch.load(test_results_path)

        dl_valid_param_length = len(d['valid_param_list_dl'])
        valid_param_length = len(d['valid_param_list'])
        print('dl_valid_param_length ', dl_valid_param_length)
        print('valid_param_length', valid_param_length)
        for i in range(valid_param_length):
            for j in range(dl_valid_param_length):
                if d['valid_param_list'][i] == d['valid_param_list_dl'][j]:
                    count += 1
                    if 'pred_loss' in d_dl:
                        corr_dl = d_dl['pred_loss'][j, 0]
                        l1_dl = d_dl['pred_loss'][j, 1]
                        ks_dl = d_dl['pred_loss'][j, 2]
                    else:
                        corr_dl = d_dl['corr_loss'][j]
                        l1_dl = d_dl['L1_loss'][j]
                        ks_dl = d_dl['ks_loss'][j]

                    corr_euler = d['corr_loss'][i]
                    mse_corr_total += (corr_dl - corr_euler)**2
                    corr_dl_list.append(corr_dl)
                    corr_euler_list.append(corr_euler)

                    l1_euler = d['l1_loss'][i]
                    mse_l1_total += (l1_dl - l1_euler)**2
                    l1_dl_list.append(l1_dl)
                    l1_euler_list.append(l1_euler)

                    ks_euler = d['ks_loss'][i]
                    mse_ks_total += (ks_dl - ks_euler)**2
                    ks_dl_list.append(ks_dl)
                    ks_euler_list.append(ks_euler)
                    # print(f"Corr dl: {corr_dl}; Corr_euler: {corr_euler}")

        print("The overlap parameter number: ", count)
        if count == 0:
            mean_mse_corr = float('nan')
            mean_mse_l1 = float('nan')
            mean_mse_ks = float('nan')
        else:
            mean_mse_corr = mse_corr_total / count
            mean_mse_l1 = mse_l1_total / count
            mean_mse_ks = mse_ks_total / count
        print("The MSE corr loss is ", mean_mse_corr)
        print("The MSE L1 loss is ", mean_mse_l1)
        print("The MSE KS loss is ", mean_mse_ks)
        res = (mean_mse_corr, mean_mse_l1, mean_mse_ks, dl_valid_param_length,
               valid_param_length, count, np.mean(corr_dl_list),
               np.mean(corr_euler_list), np.mean(l1_dl_list),
               np.mean(l1_euler_list), np.mean(ks_dl_list),
               np.mean(ks_euler_list))
    else:  # The test results path doesn't exist
        print("Hybrid Euler.")
        valid_param_length = len(d_dl['valid_param_list'])
        corr_ = torch.mean(d_dl['corr_loss']).numpy()
        l1_ = torch.mean(d_dl['L1_loss']).numpy()
        ks_ = torch.mean(d_dl['ks_loss']).numpy()
        res = (0, 0, 0, valid_param_length, valid_param_length,
               valid_param_length, corr_, corr_, l1_, l1_, ks_, ks_)
    # End if
    loss_name_list = ['corr', 'l1', 'ks']
    if save_compare_figure:
        if save_figure_dir is None:
            raise Exception(
                "You need to specify your figure saving directory.")
        x = np.arange(0, len(d['valid_param_list']), 1)
        for i in range(3):
            loss_name = loss_name_list[i]
            plt.figure()
            plt.plot(x, corr_dl_list, 'r-', label=f'{loss_name} DL loss')
            plt.plot(x, corr_euler_list, 'b-', label=f'{loss_name} Euler loss')
            plt.xlabel('parameters')
            plt.ylabel('loss')
            title = f'Comparison - {loss_name}_loss'
            plt.title(title)
            plt.legend()
            plt.savefig(
                os.path.join(save_figure_dir,
                             f'Compare_epoch{epoch}_{loss_name}' + '.png'))
            plt.close()

            print(f'Figure {i} saved.')
    return res


def check_dl_version_test_results_epochs(train_results_dir,
                                         val_results_dir,
                                         save_dir,
                                         epochs,
                                         distinct_name,
                                         need_plot_overlay_dl_euler=True,
                                         need_save_record=True,
                                         need_plot_ground_euler=False,
                                         ground_dir=None):

    record_list = np.zeros((epochs, 6))
    record_dict = {
        'mse_corr': np.zeros((epochs, ), dtype=np.float64),
        'mse_l1': np.zeros((epochs, ), dtype=np.float64),
        'mse_ks': np.zeros((epochs, ), dtype=np.float64),
        'dl_valid_param_length': np.zeros((epochs, ), dtype=np.int32),
        'valid_param_length': np.zeros((epochs, ), dtype=np.int32),
        'overlap_number': np.zeros((epochs, ), dtype=np.int32)
    }
    for i in range(epochs):
        res = check_dl_version_test_results_one_epoch(train_results_dir,
                                                      val_results_dir, i)
        mean_mse_corr, mean_mse_l1, mean_mse_ks, dl_valid_param_length, valid_param_length, count, \
            corr_dl, corr_euler, l1_dl, l1_euler, ks_dl, ks_euler = res
        record_dict['mse_corr'][i] = mean_mse_corr
        record_dict['mse_l1'][i] = mean_mse_l1
        record_dict['mse_ks'][i] = mean_mse_ks
        record_dict['dl_valid_param_length'][i] = dl_valid_param_length
        record_dict['valid_param_length'][i] = valid_param_length
        record_dict['overlap_number'][i] = count

        record_list[i] = [
            corr_dl, corr_euler, l1_dl, l1_euler, ks_dl, ks_euler
        ]
    print("Done for all epochs.")

    x = np.arange(0, epochs, 1)
    '''
    if need_plot_diff_dl_euler:
        print("Plot mean MSE...")
        plt.figure()
        plt.plot(x, record_dict['mse_corr'], 'r-', label='DL Euler mean corr loss')
        plt.plot(x, record_dict['mse_l1'], 'b-', label='DL Euler mean L1 loss')
        plt.plot(x, record_dict['mse_ks'], 'g-', label='DL Euler mean KS loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Compare DL Euler losses')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'compare_figure/Compare_DL_Euler_{distinct_name}.png'))
        plt.close()
        print("Figure saved.")
    '''
    if need_plot_overlay_dl_euler:
        tmp_dir = os.path.join(save_dir, 'pred_vs_euler')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        print("Plot Pred and Euler loss...")
        plt.figure()
        plt.plot(x,
                 record_list[:, 0],
                 color='#ff0000',
                 ls='-',
                 label='Pred corr loss')
        plt.plot(x,
                 record_list[:, 2],
                 color='#0000ff',
                 ls='-',
                 label='Pred L1 loss')
        plt.plot(x,
                 record_list[:, 4],
                 color='#00ff00',
                 ls='-',
                 label='Pred KS loss')
        plt.plot(x,
                 record_list[:, 1],
                 color='#ff6347',
                 ls=':',
                 label='Euler validated corr')
        plt.plot(x,
                 record_list[:, 3],
                 color='#4169e1',
                 ls=':',
                 label='Euler validated L1')
        plt.plot(x,
                 record_list[:, 5],
                 color='#32cd32',
                 ls=':',
                 label='Euler validated KS')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Pred and Euler validated')
        plt.legend()
        plt.savefig(
            os.path.join(tmp_dir, f'Pred_and_Euler_{distinct_name}.png'))
        plt.close()
        print(f"Figure saved to {tmp_dir}.")

    if need_save_record:
        tmp_dir = os.path.join(save_dir, 'compare_file')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        print("Saving record file...")
        dataframe = pd.DataFrame(record_dict)
        dataframe.to_csv(os.path.join(tmp_dir,
                                      f'Compare_DL_Euler_{distinct_name}.csv'),
                         sep=',')
        print(f"Record file saved to {tmp_dir}.")

    if need_plot_ground_euler:
        if ground_dir is None:
            raise Exception("Not provide ground data path.")
        tmp_dir = os.path.join(save_dir, 'ground_vs_pred')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        ground_list = np.zeros((epochs, 3))
        print("Load ground data...")
        for i in range(epochs):
            d_ground = torch.load(
                os.path.join(ground_dir, f'param_save_epoch{i}.pth'))
            ground_list[i, 0] = torch.mean(d_ground['corr_loss']).item()
            ground_list[i, 1] = torch.mean(d_ground['L1_loss']).item()
            ground_list[i, 2] = torch.mean(d_ground['ks_loss']).item()
        print("Data loaded.")
        print("Plot ground loss and Euler of DL CMA-ES loss")
        plt.figure()
        plt.plot(x,
                 ground_list[:, 0],
                 color='#ff0000',
                 ls='-',
                 label="Shaoshi's approach corr")
        plt.plot(x,
                 ground_list[:, 1],
                 color='#0000ff',
                 ls='-',
                 label="Shaoshi's approach L1")
        plt.plot(x,
                 ground_list[:, 2],
                 color='#00ff00',
                 ls='-',
                 label="Shaoshi's approach KS")
        plt.plot(x,
                 record_list[:, 1],
                 color='#ff6347',
                 ls=':',
                 label='hybrid corr validated by Euler')
        plt.plot(x,
                 record_list[:, 3],
                 color='#4169e1',
                 ls=':',
                 label='hybrid L1 validated by Euler')
        plt.plot(x,
                 record_list[:, 5],
                 color='#32cd32',
                 ls=':',
                 label='hybrid KS validated by Euler')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Ground and hybrid losses')
        plt.legend()
        plt.savefig(
            os.path.join(tmp_dir, f'Ground_and_Euler_{distinct_name}.png'))
        plt.close()
        print(f"Figure saved to {tmp_dir}.")


def compare_test_results_hybrid_normal(hybrid_test_res_path,
                                       normal_test_res_path,
                                       need_boxplot,
                                       figure_dir=None,
                                       figure_title=None):
    if need_boxplot:
        if figure_dir is None or figure_title is None:
            raise Exception(
                "You need to specify the figure saving directory and figure title."
            )
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)

    d_1 = torch.load(hybrid_test_res_path)
    d_2 = torch.load(normal_test_res_path)
    hybrid_total_loss = d_1['corr_loss'] + d_1['l1_loss'] + d_1['ks_loss']
    normal_total_loss = d_2['corr_loss'] + d_2['l1_loss'] + d_2['ks_loss']
    print('mean: ', torch.mean(hybrid_total_loss),
          torch.mean(normal_total_loss))
    print('min: ', torch.min(hybrid_total_loss), torch.min(normal_total_loss))
    statistics, p_value = stats.ttest_ind(hybrid_total_loss, normal_total_loss)
    print(f"T-test results: statistics: {statistics}; p-value: {p_value}")

    losses = torch.zeros(10, 8)
    losses[:, 0] = d_1['corr_loss']
    losses[:, 1] = d_2['corr_loss']
    losses[:, 2] = d_1['l1_loss']
    losses[:, 3] = d_2['l1_loss']
    losses[:, 4] = d_1['ks_loss']
    losses[:, 5] = d_2['ks_loss']
    losses[:, 6] = hybrid_total_loss
    losses[:, 7] = normal_total_loss

    if need_boxplot:
        print('Drawing box plot...')
        plt.figure()
        plt.rcParams['font.size'] = '8'
        plt.boxplot(
            losses.numpy(),
            labels=[
                'Hybrid\ncorr_loss', 'Shaoshi\ncorr_loss', 'Hybrid\nL1_loss',
                'Shaoshi\nL1_loss', 'Hybrid\nKS_loss', 'Shaoshi\nKS_loss',
                'Hybrid\ntotal_loss', 'Shaoshi\ntotal_loss'
            ],  # noqa
            showfliers=False,
            widths=0.3)
        plt.xlabel('Loss types')
        plt.ylabel('Loss values')
        plt.title(figure_title)
        plt.savefig(os.path.join(figure_dir, f'{figure_title}.png'))
        plt.close()
        print("Figure saved.")


def compare_train_val_test(train_dirs,
                           val_dirs,
                           test_path,
                           need_boxplot,
                           figure_dir=None,
                           figure_name=None):
    if not os.path.exists(test_path):
        print("No test path.")
        return 1

    if need_boxplot:
        if figure_dir is None or figure_name is None:
            raise Exception(
                "You must specify your figure save directory and figure name.")
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)

    epochs_in_one_trial = 100

    test_res = torch.load(test_path)

    test_corr = test_res['corr_loss']
    test_l1 = test_res['l1_loss']
    test_ks = test_res['ks_loss']
    test_total = test_corr + test_l1 + test_ks

    sorted_index = test_res['sorted_index']
    trial_nums = sorted_index / epochs_in_one_trial
    trial_nums = trial_nums.type(torch.int32)
    chosen_epochs = sorted_index % epochs_in_one_trial

    train_corr = torch.zeros_like(test_corr)
    train_l1 = torch.zeros_like(test_l1)
    train_ks = torch.zeros_like(test_ks)
    train_total = torch.zeros_like(test_total)
    val_corr = torch.zeros_like(test_corr)
    val_l1 = torch.zeros_like(test_l1)
    val_ks = torch.zeros_like(test_ks)
    val_total = torch.zeros_like(test_total)
    for i in range(len(trial_nums)):
        trial_nbr = trial_nums[i]
        epoch = chosen_epochs[i]
        # for train
        train_path = os.path.join(train_dirs[trial_nbr],
                                  f'param_save_epoch{epoch}.pth')
        train_res = torch.load(train_path)
        if 'corr_loss' in train_res:
            train_loss_tmp = train_res['corr_loss'] + train_res[
                'L1_loss'] + train_res['ks_loss']
            best_ind = torch.argmin(train_loss_tmp)
            train_corr[i] = train_res['corr_loss'][best_ind]
            train_l1[i] = train_res['L1_loss'][best_ind]
            train_ks[i] = train_res['ks_loss'][best_ind]
        else:
            print("DL")
        # for val
        val_path = os.path.join(val_dirs[trial_nbr], f'best_param{epoch}.pth')
        val_res = torch.load(val_path)
        val_corr[i] = val_res['corr_loss']
        val_l1[i] = val_res['l1_loss']
        val_ks[i] = val_res['ks_loss']

    train_total = train_corr + train_l1 + train_ks
    val_total = val_corr + val_l1 + val_ks

    print("Mean train loss: ", torch.mean(train_total).item())
    print("Mean val loss: ", torch.mean(val_total).item())
    print("Mean test loss: ", torch.mean(test_total).item())

    if need_boxplot:
        print('Drawing box plot...')
        plt.figure()
        plt.rcParams['font.size'] = '8'
        all_arr = np.hstack(
            (train_corr.unsqueeze(1).numpy(), val_corr.unsqueeze(1).numpy(),
             test_corr.unsqueeze(1).numpy(), train_l1.unsqueeze(1).numpy(),
             val_l1.unsqueeze(1).numpy(), test_l1.unsqueeze(1).numpy(),
             train_ks.unsqueeze(1).numpy(), val_ks.unsqueeze(1).numpy(),
             test_ks.unsqueeze(1).numpy(), train_total.unsqueeze(1).numpy(),
             val_total.unsqueeze(1).numpy(), test_total.unsqueeze(1).numpy()))
        plt.boxplot(all_arr,
                    labels=[
                        'Train\nco_loss', 'Val\nco_loss', 'Test\nco_loss',
                        'Train\nL1_loss', 'Val\nL1_loss', 'Test\nL1_loss',
                        'Train\nKS_loss', 'Val\nKS_loss', 'Test\nKS_loss',
                        'Train\ntotal', 'Val\ntotal', 'Test\ntotal'
                    ],
                    showfliers=False,
                    widths=0.3)
        plt.xlabel('Loss types')
        plt.ylabel('Loss values')
        plt.title(figure_name)
        plt.savefig(os.path.join(figure_dir, f'{figure_name}.png'))
        plt.close()
        print("Figure saved.")

    return train_total, val_total, test_total


def plot_EI_ratio(EI_list,
                  age_list,
                  save_fig_path,
                  xlabel='age',
                  ylabel='mean cortical E/I ratio'):
    EI_list = np.array(EI_list)
    age_list = np.array(age_list) / 12
    corr = np.corrcoef(EI_list.reshape(1, -1), age_list.reshape(1, -1))[0, 1]
    plt.figure()
    sns.regplot(x=age_list,
                y=EI_list,
                scatter_kws={'color': '#696969'},
                line_kws={'color': 'red'},
                order=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs {xlabel} curve, r={corr:.4f}')
    plt.savefig(save_fig_path)
    plt.close()
    print("Figure saved.")


def EI_age_slope_regional(n_roi, age_list, EI_regional_list):
    # EI_regional_list = np.zeros((nbr_num, n_roi))    # age_list = np.zeros((nbr_num))

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
    pvalue_fdr = stats.false_discovery_control(pvalue_arr)
    significant_num = np.sum(pvalue_fdr < 0.05)
    print(pvalue_arr, pvalue_fdr)
    print(f'Significant regions after FDR: {significant_num} / {n_roi}')
    return slope_arr, pvalue_arr, pvalue_fdr


def compare_test_results_two_lists(test_dir_1, test_dir_2,
                                   best_val_or_test_mean, prefix, nbr_range):
    """Return test results from two trials

    Args:
        test_dir_1 (str): _description_
        test_dir_2 (str): _description_
        best_val_or_test_mean (str): ['best_val', 'test_mean']
        prefix (str): ['group', 'sub']
        nbr_range (ndarray): np.arange()

    Raises:
        Exception: _description_

    Returns:
        ndarray: shape (2, len(nbr_range) (:valid_count), 4)
    """
    loss_lists = np.zeros((2, len(nbr_range), 4))
    valid_count = 0
    for nbr in nbr_range:
        # for sub_nbr in sub_list:
        path_1 = os.path.join(test_dir_1, f'{prefix}{nbr}/test_results.pth')
        if not os.path.exists(path_1):
            continue
        path_2 = os.path.join(test_dir_2, f'{prefix}{nbr}/test_results.pth')
        if not os.path.exists(path_2):
            print(nbr, end=' ')
            continue
        res_1 = torch.load(path_1)
        res_2 = torch.load(path_2)
        if best_val_or_test_mean == 'best_val':
            loss_lists[0, valid_count, 1:] = [
                res_1['corr_loss'][0], res_1['l1_loss'][0], res_1['ks_loss'][0]
            ]
            loss_lists[1, valid_count, 1:] = [
                res_2['corr_loss'][0], res_2['l1_loss'][0], res_2['ks_loss'][0]
            ]
        elif best_val_or_test_mean == 'test_mean':
            loss_lists[0, valid_count, 1:] = [
                torch.mean(res_1['corr_loss']),
                torch.mean(res_1['l1_loss']),
                torch.mean(res_1['ks_loss'])
            ]
            loss_lists[1, valid_count, 1:] = [
                torch.mean(res_2['corr_loss']),
                torch.mean(res_2['l1_loss']),
                torch.mean(res_2['ks_loss'])
            ]
        else:
            raise Exception("best val or test mean.")
        valid_count += 1
    print("Valid count: ", valid_count)
    loss_lists = loss_lists[:, :valid_count]
    loss_lists[:, :,
               0] = loss_lists[:, :, 1] + loss_lists[:, :,
                                                     2] + loss_lists[:, :, 3]

    return loss_lists


def get_test_results_from_many_dirs(test_dirs, best_val_or_test_mean, prefix,
                                    nbr_range):
    """Return test results from many trials or splits

    Args:
        test_dirs (list): [test_dir1, test_dir2, ...]
        best_val_or_test_mean (str): ['best_val', 'test_mean']
        prefix (str): ['group', 'sub']
        nbr_range (ndarray): np.arange()

    Raises:
        Exception: _description_

    Returns:
        ndarray: shape (n_dir, len(nbr_range) (:valid_count), 4)
    """
    n_dir = len(test_dirs)
    loss_lists = np.zeros((n_dir, len(nbr_range), 4))
    valid_count = 0
    for nbr in nbr_range:

        valid_flag = True
        for i_dir in range(n_dir):
            cur_path = os.path.join(test_dirs[i_dir],
                                    f'{prefix}{nbr}/test_results.pth')
            if not os.path.exists(cur_path):
                valid_flag = False
                break
        if not valid_flag:
            continue

        for i_dir in range(n_dir):
            cur_path = os.path.join(test_dirs[i_dir],
                                    f'{prefix}{nbr}/test_results.pth')
            cur_res = torch.load(cur_path)
            if best_val_or_test_mean == 'best_val':
                loss_lists[i_dir, valid_count, 1:] = [
                    cur_res['corr_loss'][0], cur_res['l1_loss'][0],
                    cur_res['ks_loss'][0]
                ]
            elif best_val_or_test_mean == 'test_mean':
                loss_lists[i_dir, valid_count, 1:] = [
                    torch.mean(cur_res['corr_loss']),
                    torch.mean(cur_res['l1_loss']),
                    torch.mean(cur_res['ks_loss'])
                ]
            else:
                raise Exception("best val or test mean.")
        valid_count += 1
    loss_lists = loss_lists[:, :valid_count]
    loss_lists[:, :,
               0] = loss_lists[:, :, 1] + loss_lists[:, :,
                                                     2] + loss_lists[:, :, 3]

    return loss_lists


def test_main():
    test_dir_1 = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/test/trial5'
    test_dir_2 = '/home/ftian/storage/projects/MFM_exploration/logs/params/PNCParams/group_by_45/test/trial6'
    best_val_or_test_mean = 'best_val'
    prefix = 'group'
    nbr_range = np.arange(1, 2)
    compare_test_results_two_lists(test_dir_1, test_dir_2,
                                   best_val_or_test_mean, prefix, nbr_range)


if __name__ == "__main__":
    test_main()
