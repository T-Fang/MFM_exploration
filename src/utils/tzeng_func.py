import numpy as np
import scipy.io as spio
import scipy.stats as stats
import scipy.special as sspec
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import seaborn as sns
import CBIG_func
import optuna
from optuna.trial import TrialState
import pickle


def tzeng_boxplot(lists, labels, save_fig_path=None, positions=None, fig_title='boxplot', xlabel='list', ylabel='y'):
    """Common used boxplot

    Args:
        lists (ndarray or lists): Different from plt.boxplot, for my convenience (directly input a list of lists), the shape of ndarray should be [n_list, features], the transpose of plt.boxplot lists.
        labels (_type_): A list of labels, must match len(lists)
        save_fig_path (_type_, optional): _description_. Defaults to None.
        positions (_type_, optional): _description_. Defaults to None.
        fig_title (str, optional): _description_. Defaults to 'boxplot'.
        xlabel (str, optional): _description_. Defaults to 'list'.
        ylabel (str, optional): _description_. Defaults to 'y'.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    if save_fig_path is None:
        raise Exception("If you need boxplot, please specify the figure save path.")
    if positions is None:
        positions = range(len(lists))
    if isinstance(lists, np.ndarray):
        lists = lists.T
    print('Drawing box plot...')
    plt.figure()
    plt.boxplot(lists,
                labels=labels,
                showfliers=False,
                positions=positions
                )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{fig_title}')
    plt.savefig(save_fig_path)
    plt.close()
    print("Boxplot figure saved.")
    return 0


def tzeng_check_logfile_empty(log_dir, log_nbr_list, logfile_prefix='', logfile_postfix='.log'):
    """Check if error logs are empty. Output the error logs not empty in shell array format (to be easily copyied to shell scripts)

    Args:
        log_dir (str): error log files directory.
        log_nbr_list (scalar or list or array-like): the log file nbrs list. If it's a scalar, will be np.arange(0, log_nbr_list, 1)
        logfile_prefix (str, optional): the log file name before the nbr. Defaults to ''.
        logfile_postfix (str, optional): the log file name after the nbr. Defaults to '.log'.
    
    Examples:
        log_dir = '/home/tzeng/storage/Matlab/HCP_Dev/sh_logs/everysub'
        loglist = np.arange(51, 100)
        tzeng_func.tzeng_check_logfile_empty(log_dir, loglist)
        # The error log file name is the default: $(nbr)_error.log
    """
    if not str.endswith(logfile_postfix, '.log'):
        logfile_postfix = logfile_postfix + '.log'
    if np.isscalar(log_nbr_list):
        log_nbr_list = np.arange(log_nbr_list)
    
    nonexist_list = []
    empty_list = []
    nonempty_list = []

    for i in log_nbr_list:
        log_filename = logfile_prefix + f'{i}' + logfile_postfix
        log_file_path = os.path.join(log_dir, log_filename)
        if not os.path.exists(log_file_path):
            nonexist_list.append(i)
            continue
        if os.stat(log_file_path).st_size == 0:
            empty_list.append(i)
        else:
            nonempty_list.append(i)
    
    return empty_list, nonempty_list, nonexist_list


def tzeng_check_n_return_mat_file(file_path, key_list):
    """Check and return .mat file. Check whether all keys in key_list exist in the mat file.

    Args:
        file_path (str): file path
        key_list (list): list of keys that should exist in this file

    Returns:
        dict: =spio.loadmat()
    """
    if not os.path.exists(file_path):
        print(f'{file_path} does not exist.')
        return None
    dict_file = spio.loadmat(file_path)
    for key in key_list:
        if key not in dict_file:
            print(f'{key} does not exist.')
            return None
    return dict_file


def tzeng_fisher_average(corr_mat):
    """Average by Fisher transformation

    Args:
        corr_mat (ndarray): [number, ...]
    Returns:
        corr_ave: [...]
    """
    z_corr_mat = CBIG_func.CBIG_StableAtanh(corr_mat)
    z_corr_ave = np.nanmean(z_corr_mat, axis=0)
    corr_ave = np.tanh(z_corr_ave)
    return corr_ave


def tzeng_group_SC_matrices(sc_mats):
    """group SC matrices like that in matlab. Neglect 'nan' entries. Those entries with lower than half sc_mats > 0 will be set to 0. Scale the final SC by log function. Set the diagonal to be 0.

    Args:
        sc_mats (np.array): [n_mats, n_roi, n_roi]

    Returns:
        grouped_sc (np.array): the group average of the input sc_mats. [n_roi, n_roi]
    """
    n_mats = sc_mats.shape[0]
    n_roi = sc_mats.shape[1]
    grouped_sc = np.zeros((n_roi, n_roi))

    for i in range(n_roi):
        for j in range(n_roi):
            
            if i == j:  #  if i == j, grouped_sc[i, j] = 0;
                continue
            
            count_non_zero = 0  #  To count the number of non_zero
            sum_temp = 0
            for mat_i in range(n_mats):  # For each SC matrix
                if np.isnan(sc_mats[mat_i, i, j]):
                    continue
                
                if sc_mats[mat_i, i, j] != 0:
                    count_non_zero += 1
                    sum_temp += sc_mats[mat_i, i, j]
            # End for mat_i
            # Only when there are over half of the SC[i, j] are not zero
            if count_non_zero >= 0.5 * n_mats:
                grouped_sc[i, j] = np.log(sum_temp / count_non_zero)
    
    return grouped_sc


def tzeng_list_difference(list_1, list_2, need_print=True, print_type=''):
    """Find values only in list_1 but not in list_2

    Args:
        list_1 (list or 1D array): list
        list_2 (list or 1D array): list
        need_print (bool, optional): whether to print list. Defaults to True.
        print_type (str, optional): 'sh' or '', 'sh' will print by end=' '. Defaults to ''.
    """
    list_1 = np.array(list_1)
    list_2 = np.array(list_2)
    list_diff = np.setdiff1d(list_1, list_2)
    n_diff = len(list_diff)
    if need_print:
        print("The number of different values (in list1 but not in list2): ", n_diff)
        if print_type == 'sh':
            for i in range(n_diff):
                print(list_diff[i], end=' ')
            print()
        else:
            print(list_diff.tolist())


def tzeng_ln_Bessel(d, k, k0=None):
    """Calculate the log_e of modified Bessel function of the first kind, e.g., log I_{d/2-1}(k)
    It can tackle the problem of k overflow in scipy.special.iv. By numerical integration.

    Args:
        d (scalar): the original feature dimension. nu = d / 2 - 1
        k (scalar or 1D array): kappa
        k0 (scalar, optional): the numerical integration starting point. Defaults to None.

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        scalar or 1D array: the same shape as k. The value of log I_{d/2-1}(k)
    """
    nu = d / 2 - 1
    if np.isscalar(k):
        k = np.ones((1,)) * k
    out = np.log(sspec.iv(nu, k))

    if k0 is None:
        invalid_mask = np.isinf(out)
    else:
        invalid_mask = k > k0
    
    if invalid_mask.any():
        if k0 is None:
            raise Exception("Need an initial scalar k0, because some values are out of range.")
        ini_out = np.log(sspec.iv(nu, k0))
        if np.isinf(ini_out):
            raise Exception("Please use a smaller initial scalar k0.")
        n_grids = 1000  # The numerical integration step size

        ofintv = (k[invalid_mask] - k0) / n_grids
        tempcnt = np.arange(0.5, n_grids + 0.5, 1)
        ks = k0 + tempcnt[np.newaxis, :] * ofintv[:, np.newaxis]
        u = (d - 1) / (2 * ks)
        adsum = np.sum(nu / ks + 1 / (u + np.sqrt(1 + u**2)), axis=1)

        out[invalid_mask] =  ini_out + ofintv * adsum

    out = np.squeeze(out)
    return out


def tzeng_map_network_roi(network_dlabel, roi_dlabel):
    """Mapping ROIs to network decided by overlapping. 0 will be excluded automatically.

    Args:
        network_dlabel (array): [vertices_num, ] with 0 ~ network_num consecutive numbers
        roi_dlabel (array): [vertices_num, ] with 0 ~ ROI_num consecutive numbers

    Returns:
        array: [roi_num, ] with 1 ~ network_num network assigning to each ROI
        dict: {f'network{i}': array([ROIs_belong_to_this_network])}
    
    Example:
        network_dlabel: [4, 3, 0, 4, 4, ...]; roi_dlabel: [43, 55, 0, 33, 43, ... ]
        
        network_dlabel = sio.loadmat('/home/tzeng/storage/Matlab/Utils/general_mats/Yeo7_fslr32.mat')
        network_dlabel = network_dlabel['dlabel']
        roi_dlabel = sio.loadmat('/home/tzeng/storage/Matlab/Utils/general_mats/Yan400_Yeo7_fslr32.mat')
        roi_dlabel = roi_dlabel['dlabel']
        _, map_dict = tzeng_map_network_roi(network_dlabel, roi_dlabel)
    """
    network_num = np.amax(network_dlabel)
    roi_num = np.amax(roi_dlabel)
    
    overlap_mat = np.zeros((roi_num, network_num))
    for i in range(roi_num):
        for j in range(network_num):
            overlap_mat[i, j] = np.sum(np.float64(roi_dlabel == i + 1) * np.float64(network_dlabel == j + 1))

    mapping_mat = np.argmax(overlap_mat, axis=1) + 1
    
    mapping_dict = {}
    roi_list = np.arange(1, roi_num + 1, 1)
    for i in range(1, network_num + 1):
        mapping_dict[f'network{i}'] = roi_list[mapping_mat == i]

    return mapping_mat, mapping_dict


def tzeng_npy2mat(path, variable_name, target_dir, target_name=None):
    """For transferring .npy file or files in a directory to .mat file or files. You can input file or directory.

    Args:
        path (str): can be file or directory
        variable_name (str): The dictionary key for .mat file
        target_dir (str): The directory for saving .mat files
        target_name (str or str list, optional): mat file name, if not specified it will keep the npy file(s)'s name. If list length is smaller than file number, then it will give name to first read-in files. Defaults to None.

    Returns:
        0 or None: 0 - succeed, None - fail
    
    Example:
        path = '/home/tzeng/storage/Python/Parcellation/Cluster/ClusterData/FC_group'
        target_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/FC_group'
        # target_name = 'split100.mat'
        variable_name = 'fc_split'
        res = tzeng_npy2mat(path, variable_name, target_dir)
    """
    if not os.path.exists(path):
        print("The input path doesn't exist.")
        return None

    if os.path.isfile(path):
        filename = path.split('/')[-1]
        
        if not filename.endswith('.npy'):
            print("The input file is not a numpy file.")
            return None
        
        print(f'Start loading from {path}...')
        try:
            f = np.load(path)
            print(f'Successfully load in {filename}.')
        except:
            print(f'Cannot load in {filename}.')
            
        if target_name is None:
            target_name = filename[:-4] + '.mat'
            
        try:
            spio.savemat(os.path.join(target_dir, target_name), {variable_name:f})
            print(f'Successfully saved to {target_name} in {target_dir}.')
        except:
            print("Cannot save mat file. Check your target directory and file name.")
            
    elif os.path.isdir(path):
        print(f'Start loading from {path}...')
        file_list = os.listdir(path)
        npy_file_count = 0
        
        for filename in file_list:
            
            if filename.endswith('.npy'):
                npy_file_count += 1
                
                try:
                    f = np.load(os.path.join(path, filename))
                    print(f'Successfully load in {filename}.')
                except:
                    print(f'Cannot load in {filename}.')
                
                if target_name is not None and len(target_name) >= npy_file_count:
                    tmp_target_name = target_name[npy_file_count]
                else:
                    tmp_target_name = filename[:-4] + '.mat'
                
                try:
                    spio.savemat(os.path.join(target_dir, tmp_target_name), {variable_name:f})
                    print(f'Successfully saved to {tmp_target_name} in {target_dir}.')
                except:
                    print("Cannot save mat file. Check your target directory.")
                    
        print(f'Done for {path}. Totally transferred {npy_file_count} files.')
    
    else:
        return None
    
    return 0


def tzeng_optuna_tune(from_sketch, study_name, store_dir, objective, direction, n_trials, timeout, ini_dict=None, n_startup_trials=5, n_warmup_steps=10):
    """optuna tune main function, TPESampler, MedianPruner

    Args:
        from_sketch (boolean): Start from scratch or load from previous study.
        study_name (str): Unique identifier of the study
        store_dir (str): storage directory. The study database and sampler will be stored under it with file name from ${study_name}.
        objective (function): the objective function
        direction (str): "minimize" or "maximize" the objective
        n_trials (int): number of trials
        timeout (int): in seconds. Terminate time.
        ini_dict (dict, optional): initial parameters for first trial. Defaults to None.
        n_startup_trials (int, optional): For Pruner. Defaults to 5.
        n_warmup_steps (int, optional): For Pruner. Defaults to 10.
    """

    storage_path = os.path.join(store_dir, study_name + '.db')
    storage_name = f"sqlite:///{storage_path}"

    if from_sketch:
        seed = np.random.randint(0, 2**32 - 1)
        print('Seed: ', seed)
        study = optuna.create_study(direction=direction, study_name=study_name, storage=storage_name, sampler=optuna.samplers.TPESampler(seed), pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps))
        if ini_dict is not None:
            study.enqueue_trial(ini_dict)
    else:
        restored_sampler = pickle.load(open(os.path.join(store_dir, f"sampler_{study_name}.pkl"), "rb"))
        study = optuna.create_study(direction="minimize", study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler)

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    with open(os.path.join(store_dir, f"sampler_{study_name}.pkl"), "wb") as fout:
        pickle.dump(study.sampler, fout)
        print("Sampler saved.")
    
    return


def tzeng_scatter_with_regress_line(x_list, y_list, save_fig_path, figure_title='Scatter plot', xlabel='x', ylabel='y'):
    """Scatter plot with regress line and correlation shown beside figure title

    Args:
        x_list (list or 1D-arraylike): x-axis values
        y_list (list or 1D-arraylike): y-axis values
        save_fig_path (str): figure saving path, should be checked by yourself.
        figure_title (str, optional): Defaults to 'Scatter plot'.
        xlabel (str, optional): Defaults to 'x'.
        ylabel (str, optional): Defaults to 'y'.

    Returns:
        0: _description_
    """
    print("Start scatterplotting with regression line ...")
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    # corr = np.corrcoef(y_list.reshape(1, -1), x_list.reshape(1, -1))[0, 1]
    corr = CBIG_func.CBIG_corr(y_list.reshape((-1, 1)), x_list.astype(np.float64).reshape((-1, 1)))[0, 0]
    plt.figure()
    sns.regplot(x=x_list, y=y_list,  scatter_kws={'color': '#696969'}, line_kws={'color': 'red'})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{figure_title}, r={corr:.4f}')
    plt.savefig(save_fig_path)
    plt.close()
    print(f"Figure saved to {save_fig_path}.")
    return 0


def tzeng_visualize_FCD(fcd_mat, save_fig_path):
    """Plot FCD matrix using CBIG FCD colormap

    Args:
        fcd_mat (array): [n_window, n_window]
        save_fig_path (str): _description_

    Returns:
        _type_: _description_
    """
    cmap = spio.loadmat('/home/shaoshi.z/storage/MFM/script/Rapaeh_color_table_FCD.mat')
    cmap = np.array(cmap['c3'])
    cmap = np.hstack((cmap, np.ones((cmap.shape[0], 1))))  # [N, 4]
    cmap = ListedColormap(cmap)
    plt.imshow(fcd_mat, cmap=cmap)
    plt.savefig(save_fig_path)
    plt.close()
    return 0


def tzeng_KS_distance(s_cdf, t_cdf=None):
    """Compute KS distance within a CDF set or between two CDF sets.

    Args:
        s_cdf (array): [s_labels, bins]
        t_cdf (array, optional): [t_labels, bins]. If None, will compute within s_cdf
    
    Returns:
        array: distance matrix, [s_labels, t_labels]
    """
    s_cdf = s_cdf / s_cdf[:, -1][:, np.newaxis]
    if t_cdf is None:
        t_cdf = s_cdf
    else:
        t_cdf = t_cdf / t_cdf[:, -1][:, np.newaxis]
    distance_matrix = np.zeros((s_cdf.shape[0], t_cdf.shape[0]))
    for k in range(distance_matrix.shape[0]):
        ks_tmp = np.abs(t_cdf - s_cdf[k][np.newaxis, :])  # [xxx, bins]
        ks_tmp = np.amax(ks_tmp, axis=1)  # [xxx]
        distance_matrix[k] = ks_tmp
    return distance_matrix


def tzeng_2_sample_t_test_n_plot(list_1, list_2, need_boxplot, need_pvalue=True, labels=['list_1', 'list_2'], save_fig_path=None, fig_title='t_test', xlabel='list', ylabel='y'):
    """Perform 2 sample T-test and plot

    Args:
        list_1 (ListLike): The first list / array
        list_2 (ListLike): The second list / array
        need_boxplot (boolean): whether to boxplot
        labels (list, optional): boxplot group labels. Defaults to ['list_1', 'list_2'].
        save_fig_path (str, optional): save figure path. Defaults to None.
        fig_title (str, optional): figure title, p-value will be added automatically. Defaults to 't_test'.
        xlabel (str, optional): x-axis label. Defaults to 'list'.
        ylabel (str, optional): y-axis label. Defaults to 'y'.

    Raises:
        Exception: Not specify save_fig_path if need_boxplot=True
    """
    if need_pvalue:
        statistics, p_value = stats.ttest_ind(list_1, list_2)
        print('Average: ', np.mean(list_1), np.mean(list_2))
        print(f"T-test results: statistics: {statistics}; p-value: {p_value}")
    
    if need_boxplot:
        if save_fig_path is None:
            raise Exception("If you need boxplot, please specify the figure save path.")
        print('Drawing box plot...')
        plt.figure()
        plt.boxplot([list_1, list_2],
                    labels=labels,
                    showfliers=False
                    )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if need_pvalue:
            tmp_title = f'{fig_title}, p={p_value:.4f}'
        else:
            tmp_title = f'{fig_title}'
        plt.title(tmp_title)
        plt.savefig(save_fig_path)
        plt.close()
        print("Boxplot figure saved.")
    return 0


def test_main():
    _, non_empty, non_exist = tzeng_check_logfile_empty('/home/tzeng/storage/Matlab/HCPS1200/logs/log_every_sub', np.arange(2, 1030), 's', '_error.log')
    print(non_empty)
    print(non_exist)
    

if __name__ == "__main__":
    test_main()
