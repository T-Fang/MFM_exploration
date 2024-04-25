import configparser
import scipy.io as spio
import torch
import numpy as np
import pandas as pd
import os
import sys
import datetime

sys.path.insert(1, '/home/ftian/storage/projects/MFM_exploration')
from src.scripts.Hybrid_CMA_ES import ModelHandler, CMAESValidator, CMAESTester, \
    simulate_fc_fcd, train_help_function, simulate_fc_fcd_mat
from src.utils import tzeng_func
from src.utils.init_utils import set_torch_default
from src.basic.constants import CONFIG_DIR, LOG_DIR
from src.utils.file_utils import get_HCPYA_group_stats, get_best_params_file_path, get_sim_res_dir, get_sim_res_path, get_run_dir, get_target_dir  # noqa
from multiprocessing import Pool, set_start_method  # noqa: F401


def apply_on_all_participants(phase, trial_idx, seed_idx):
    DS_NAME = 'HCPYA'
    set_torch_default()

    config = configparser.ConfigParser()
    config.read(os.path.join(CONFIG_DIR, 'model', 'config_hcpya.ini'))

    trial_idx = int(trial_idx)
    seed_idx = int(seed_idx)

    num_epochs = 100
    TARGET = 'all_participants'
    parent_dir = get_target_dir(DS_NAME, TARGET)
    if phase == 'train':
        group_stats = get_HCPYA_group_stats(0, 343)
        train_save_dir = get_run_dir(DS_NAME, TARGET, 'train', trial_idx,
                                     seed_idx)
        seed = None
        state = train_help_function(config=config,
                                    myelin=group_stats['myelin'],
                                    RSFC_gradient=group_stats['rsfc_gradient'],
                                    sc_euler=group_stats['sc_euler'],
                                    emp_fc=group_stats['emp_fc'],
                                    emp_fcd_cum=group_stats['emp_fcd_cum'],
                                    train_save_dir=train_save_dir,
                                    num_epochs=num_epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 100),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='Uniform',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif phase == 'validation':
        # For each training epoch, get the child (param vector) with the lowest training loss.
        # Then, validate the child on the validation set, and save the result to the val_dir.
        train_save_dir = get_run_dir(DS_NAME, TARGET, 'train', trial_idx,
                                     seed_idx)
        val_save_dir = get_run_dir(DS_NAME, TARGET, 'val', trial_idx, seed_idx)
        group_stats = get_HCPYA_group_stats(343, 686)
        mfm_validator = CMAESValidator(config, train_save_dir, val_save_dir,
                                       group_stats['sc_euler'],
                                       group_stats['emp_fc'],
                                       group_stats['emp_fcd_cum'], num_epochs)

        mfm_validator.validate()

    elif phase == 'test':
        USE_ALL_SEEDS = False

        seed_indices = np.arange(1, seed_idx +
                                 1) if USE_ALL_SEEDS else [seed_idx]
        val_save_dirs = [
            get_run_dir(DS_NAME, TARGET, 'val', trial_idx, seed_idx)
            for seed_idx in seed_indices
        ]

        print("Validation dirs: ", val_save_dirs)

        test_save_dir = get_run_dir(DS_NAME, TARGET, 'test', trial_idx,
                                    seed_idx)

        group_stats = get_HCPYA_group_stats(686, 1029)
        mfm_tester = CMAESTester(config,
                                 val_save_dirs,
                                 test_save_dir,
                                 group_stats['sc_euler'],
                                 group_stats['emp_fc'],
                                 group_stats['emp_fcd_cum'],
                                 train_num_epochs=num_epochs)
        mfm_tester.test()

    elif phase == 'best_from_train':
        # Mainly for simulating the best param with lowest train loss
        PHASE = 'train'
        sim_res_dir = get_sim_res_dir(DS_NAME, TARGET, PHASE, trial_idx,
                                      seed_idx)
        train_save_dir = get_run_dir(DS_NAME, TARGET, 'train', trial_idx,
                                     seed_idx)
        val_save_dir = get_run_dir(DS_NAME, TARGET, 'val', trial_idx, seed_idx)
        best_from_train_path = get_best_params_file_path('train', val_save_dir)
        group_stats = get_HCPYA_group_stats(0, 343)

        mfm_validator = CMAESValidator(config, train_save_dir, val_save_dir,
                                       group_stats['sc_euler'],
                                       group_stats['emp_fc'],
                                       group_stats['emp_fcd_cum'], num_epochs)
        mfm_validator.get_best_train_params()

        mfm_validator = ModelHandler(config=config,
                                     phase=PHASE,
                                     sc_euler=group_stats['sc_euler'],
                                     emp_fc=group_stats['emp_fc'],
                                     emp_fcd_cum=group_stats['emp_fcd_cum'])

        mfm_validator.sim_first_param_multi_times(best_from_train_path,
                                                  sim_res_dir,
                                                  10,
                                                  get_FCD_matrix=True,
                                                  get_bold=True,
                                                  seed=None)

    elif phase == 'val_best':
        # get the top 10 param vectors with the lowest validation loss
        # from val_dirs and save the result in the val_best_dir

        USE_ALL_SEEDS = False
        seed_indices = np.arange(1, seed_idx +
                                 1) if USE_ALL_SEEDS else [seed_idx]
        val_save_dirs = [
            get_run_dir(DS_NAME, TARGET, 'validation', trial_idx, seed_idx)
            for seed_idx in seed_indices
        ]

        val_best_dir = get_run_dir(DS_NAME, TARGET, 'val_best', trial_idx,
                                   seed_idx)
        mfm_tester = CMAESTester(config,
                                 val_save_dirs,
                                 val_best_dir,
                                 train_num_epochs=num_epochs)
        mfm_tester.select_best_from_val()

    elif phase == 'simulate_fc_fcd':
        group_stats = get_HCPYA_group_stats(686, 1029)

        val_best_results_path = os.path.join(
            parent_dir, f'val_best/trial{trial_idx}/seed{seed_idx}',
            'val_results.pth')
        val_results = torch.load(val_best_results_path)
        parameter = val_results['parameter']  # [205, param_sets]
        sim_results_dir = os.path.join(
            parent_dir, f'simulate/trial{trial_idx}/seed{seed_idx}')
        if not os.path.exists(sim_results_dir):
            os.makedirs(sim_results_dir)
        sim_results_path = os.path.join(sim_results_dir, 'sim_results.pth')
        simulate_fc_fcd(config,
                        save_path=sim_results_path,
                        parameter=parameter,
                        param_dup=3,
                        sc_euler=group_stats['sc_euler'])

    elif phase == 'simulate_fc_fcd_mat':
        group_stats = get_HCPYA_group_stats(0, 343)

        parameter_path = os.path.join(
            parent_dir, f'train/trial{trial_idx}/seed{seed_idx}',
            'param_save_epoch99.pth')
        parameter = torch.load(parameter_path)['parameter'][:, :10]
        sim_results_dir = os.path.join(
            parent_dir, f'simulate/trial{trial_idx}/seed{seed_idx}')
        if not os.path.exists(sim_results_dir):
            os.makedirs(sim_results_dir)
        sim_results_path = os.path.join(sim_results_dir,
                                        'sim_results_mats.pth')
        simulate_fc_fcd_mat(config,
                            sim_results_path,
                            parameter,
                            param_dup=3,
                            sc_euler=group_stats['sc_euler'])

    return 0


def apply_large_group(mode, trial_nbr, seed_nbr, epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/ftian/storage/projects/MFM_exploration/configs/model/config_hcpya.ini'
    )

    trial_nbr = int(trial_nbr)
    seed_nbr = int(seed_nbr)

    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    myelin_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'myelin_roi_1029.mat'))
    myelin_indi = myelin_indi['myelin_roi_1029']
    rsfc_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'rsfc_roi_1029.mat'))
    rsfc_indi = rsfc_indi['rsfc_roi_1029']
    sc_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'sc_roi_1029.mat'))
    sc_indi = sc_indi['sc_roi_1029']

    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029'
    epochs = 4

    # Always using train set myelin and RSFC gradient
    myelin = np.nanmean(myelin_indi[0:680], axis=0)
    myelin = torch.as_tensor(myelin).unsqueeze(1)
    rsfc_gradient = np.nanmean(rsfc_indi[0:680], axis=0)
    rsfc_gradient = torch.as_tensor(rsfc_gradient).unsqueeze(1)

    if mode == 'train':
        group_stats = get_HCPYA_group_stats(860, 917)

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}')
        seed = None

        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_euler=group_stats['sc_mat'],
                                    emp_fc=group_stats['emp_fc'],
                                    emp_fcd_cum=group_stats['emp_fcd_cum'],
                                    train_save_dir=save_param_dir,
                                    num_epochs=epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 4),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='Uniform',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        group_stats = get_HCPYA_group_stats(917, 973)

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}')
        val_dir = os.path.join(parent_dir,
                               f'validation/trial{trial_nbr}/seed{seed_nbr}')
        mfm_validator = CMAESValidator(config, save_param_dir, val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(group_stats['sc_euler'],
                                                  group_stats['emp_fc'],
                                                  group_stats['emp_fcd_cum'],
                                                  ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(group_stats['sc_euler'],
                                              group_stats['emp_fc'],
                                              group_stats['emp_fcd_cum'],
                                              epoch)

    elif mode == 'test':
        group_stats = get_HCPYA_group_stats(973, 1029)

        # val_dirs = [os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}') for i in np.arange(1, seed_nbr + 1)]
        val_dirs = [
            os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}')
            for i in [seed_nbr]
        ]
        print("Validation dirs: ", val_dirs)
        test_dir = os.path.join(parent_dir,
                                f'test/trial{trial_nbr}/seed{seed_nbr}')
        mfm_tester = CMAESTester(config,
                                 val_dirs,
                                 test_dir,
                                 train_num_epochs=epochs)
        mfm_tester.test_old(group_stats['sc_euler'], group_stats['emp_fc'],
                            group_stats['emp_fcd_cum'])

    elif mode == 'val_best':
        val_dirs = [
            os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}')
            for i in [seed_nbr]
        ]
        val_best_dir = os.path.join(
            parent_dir, f'val_best/trial{trial_nbr}/seed{seed_nbr}')
        mfm_tester = CMAESTester(config,
                                 val_dirs,
                                 val_best_dir,
                                 train_num_epochs=epochs)
        mfm_tester.select_best_from_val()

    elif mode == 'simulate_fc_fcd':
        sc_mat = tzeng_func.tzeng_group_SC_matrices(sc_indi[0:680])
        sc_mat = torch.as_tensor(sc_mat)
        sc_euler = sc_mat / torch.max(sc_mat) * 0.02

        val_best_results_path = os.path.join(
            parent_dir, f'val_best/trial{trial_nbr}/seed{seed_nbr}',
            'val_results.pth')
        val_results = torch.load(val_best_results_path)
        parameter = val_results['parameter']  # [205, param_sets]
        sim_results_dir = os.path.join(
            parent_dir, f'simulate/trial{trial_nbr}/seed{seed_nbr}')
        if not os.path.exists(sim_results_dir):
            os.makedirs(sim_results_dir)
        sim_results_path = os.path.join(sim_results_dir, 'sim_results.pth')
        simulate_fc_fcd(config,
                        save_path=sim_results_path,
                        parameter=parameter,
                        param_dup=3,
                        sc_euler=sc_euler)

    elif mode == 'simulate_fc_fcd_mat':
        sc_mat = tzeng_func.tzeng_group_SC_matrices(sc_indi[0:680])
        sc_mat = torch.as_tensor(sc_mat)
        sc_euler = sc_mat / torch.max(sc_mat) * 0.02

        parameter_path = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}',
            'param_save_epoch99.pth')
        parameter = torch.load(parameter_path)['parameter'][:, :10]
        sim_results_dir = os.path.join(
            parent_dir, f'simulate/trial{trial_nbr}/seed{seed_nbr}')
        if not os.path.exists(sim_results_dir):
            os.makedirs(sim_results_dir)
        sim_results_path = os.path.join(sim_results_dir,
                                        'sim_results_mats.pth')
        simulate_fc_fcd_mat(config,
                            sim_results_path,
                            parameter,
                            param_dup=3,
                            sc_euler=sc_euler)

    return 0


def apply_individual(mode, sub_nbr, trial_nbr, seed_nbr, epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/tzeng/storage/Python/MFMApplication/configs/general/config_hcpya.ini'
    )

    sub_nbr = int(sub_nbr)
    trial_nbr = int(trial_nbr)

    sublist = pd.read_csv(
        '/home/tzeng/storage/Matlab/HCPS1200/txt_files/HCP1029_sublist.txt',
        sep='\t',
        header=None,
        index_col=False)
    sublist = np.squeeze(np.array(sublist))
    subject_id = sublist[sub_nbr]
    print('subject id: ', subject_id)

    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'

    myelin_sets = spio.loadmat(
        os.path.join(individual_mats_path, 'myelin_roi_1029.mat'))
    myelin_sets = myelin_sets['myelin_roi_1029']
    # rsfc_gradient_sets = spio.loadmat(os.path.join(individual_mats_path, 'rsfc_roi_1029.mat'))
    # rsfc_gradient_sets = rsfc_gradient_sets['rsfc_roi_1029']
    sc_sets = spio.loadmat(
        os.path.join(individual_mats_path, 'sc_roi_1029.mat'))
    sc_sets = sc_sets['sc_roi_1029']
    if sub_nbr > len(myelin_sets):
        raise Exception("Subject index out of range.")
    myelin = torch.as_tensor(myelin_sets[sub_nbr]).unsqueeze(1)  # [68, 1]
    if torch.isnan(myelin).any():
        print(f"Myelin not available for subject {sub_nbr} in 1029 subjects.")
        return
    sc_mat = torch.as_tensor(
        sc_sets[sub_nbr])  # [68, 68] for DL predicting loss model
    if torch.isnan(sc_mat).any():
        print(
            f"SC matrix not available for subject {sub_nbr} in 1029 subjects.")
        return
    tmp_sc_mask = torch.eye(sc_mat.shape[0], dtype=torch.bool)
    sc_mat[tmp_sc_mask] = 0  # Set the diagonal to be zero
    tmp_sc_mask = sc_mat > 0
    sc_mat[tmp_sc_mask] = torch.log(sc_mat[tmp_sc_mask])  # Take the log
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    rsfc_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/RSFC_gradients/Desikan_1_same_direction'
    rsfc_path = os.path.join(rsfc_dir, f'{subject_id}.mat')
    rsfc_runs = tzeng_func.tzeng_check_n_return_mat_file(
        rsfc_path, ['REST1_LR', 'REST1_RL'])

    fc_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rfMRI_FC_ICAFIX_4runs'
    fc_path = os.path.join(fc_dir, f'{subject_id}.mat')
    fc_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fc_path, ['fc_REST1_LR', 'fc_REST1_RL', 'fc_REST2_LR', 'fc_REST2_RL'])

    fcd_dir = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/NeuralMass/rfMRI_FCD_CDF_ICAFIX_4runs'
    fcd_path = os.path.join(fcd_dir, f'{subject_id}.mat')
    fcd_runs = tzeng_func.tzeng_check_n_return_mat_file(
        fcd_path, [
            'fcd_cdf_REST1_LR', 'fcd_cdf_REST1_RL', 'fcd_cdf_REST2_LR',
            'fcd_cdf_REST2_RL'
        ])

    # Use same myelin, RSFC Gradient, SC across train, validation and test.
    rsfc_gradient = (rsfc_runs['REST1_LR'] + rsfc_runs['REST1_RL']) / 2
    rsfc_gradient = torch.as_tensor(rsfc_gradient)
    # rsfc_gradient = torch.as_tensor(rsfc_indi[sub_nbr]).unsqueeze(1)

    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/individual'
    epochs = 100

    if mode == 'train':
        # emp_fc = fc_runs['fc_REST1_LR']
        emp_fc = np.array([fc_runs['fc_REST1_LR'], fc_runs['fc_REST1_RL']])
        emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)
        emp_fc = torch.as_tensor(emp_fc)
        # emp_fcd_cum = fcd_runs['fcd_cdf_1']
        emp_fcd_cum = fcd_runs['fcd_cdf_REST1_LR'] + fcd_runs[
            'fcd_cdf_REST1_RL']
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}/sub{sub_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_euler=sc_mat,
                                    emp_fc=emp_fc,
                                    emp_fcd_cum=emp_fcd_cum,
                                    train_save_dir=save_param_dir,
                                    num_epochs=epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 100),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='Uniform',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    # elif mode == 'val_train_param':
    #     # rsfc_gradient = rsfc_runs['REST1_LR']
    #     # rsfc_gradient = torch.as_tensor(rsfc_gradient)
    #     rsfc_gradient = torch.as_tensor(rsfc_indi[sub_nbr]).unsqueeze(1)
    #     emp_fc = fc_runs['fc_REST1_LR']
    #     emp_fc = torch.as_tensor(emp_fc)
    #     emp_fcd_cum = fcd_runs['fcd_cdf_1']
    #     emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
    #     emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    #     trial_nbr = int(trial_nbr)
    #     epoch = int(epoch)
    #     save_param_dir = os.path.join(parent_dir, f'train/trial{trial_nbr}/split{split_nbr}/sub{sub_nbr}')
    #     val_train_param_dir = os.path.join(parent_dir, f'val_train_param/trial{trial_nbr}/split{split_nbr}/sub{sub_nbr}')
    #     mfm_validator = DLVersionCMAESValidator(save_param_dir, val_train_param_dir, dataset_name=dataset_name)
    #     mfm_validator.val_by_euler(myelin, rsfc_gradient, sc_euler, emp_fc, emp_fcd_cum, epoch)

    elif mode == 'validation':
        # emp_fc = fc_runs['fc_REST1_RL']
        emp_fc = fc_runs['fc_REST2_LR']
        emp_fc = torch.as_tensor(emp_fc)
        # emp_fcd_cum = fcd_runs['fcd_cdf_2']
        emp_fcd_cum = fcd_runs['fcd_cdf_REST2_LR']
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}')
        val_dir = os.path.join(parent_dir,
                               f'validation/trial{trial_nbr}/seed{seed_nbr}')
        mfm_validator = CMAESValidator(config, save_param_dir, val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, emp_fc,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, emp_fc, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        '''emp_fc = np.array([fc_runs['fc_REST2_LR'], fc_runs['fc_REST2_RL']])
        emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)'''
        emp_fc = fc_runs['fc_REST2_RL']
        emp_fc = torch.as_tensor(emp_fc)
        # emp_fcd_cum = (fcd_runs['fcd_cdf_3'] + fcd_runs['fcd_cdf_4'])
        emp_fcd_cum = fcd_runs['fcd_cdf_REST2_RL']
        emp_fcd_cum = torch.as_tensor(emp_fcd_cum.astype(np.float64))
        emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

        val_dirs = [
            os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(parent_dir,
                                f'test/trial{trial_nbr}/seed{seed_nbr}')
        mfm_tester = CMAESTester(config,
                                 val_dirs,
                                 test_dir,
                                 train_num_epochs=epochs)
        mfm_tester.test_old(sc_euler, emp_fc, emp_fcd_cum)

    return 0


def generate_dl_dataset_group(mode, group_nbr, trial_nbr, seed_nbr):
    # to generate deep learning dataset

    group_nbr = int(group_nbr)
    trial_nbr = int(trial_nbr)
    seed_nbr = int(seed_nbr)

    config = configparser.ConfigParser()
    config.read(os.path.join(CONFIG_DIR, 'model', 'config_hcpya.ini'))

    grouped_mats_dir = '/home/ftian/storage/projects/MFM_exploration/data/DL_group_mats/Desikan'
    grouped_mats_path = os.path.join(grouped_mats_dir, f'{mode}.mat')
    grouped_mats = spio.loadmat(grouped_mats_path)

    myelin = np.array(grouped_mats['myelin_groups'])
    myelin = torch.as_tensor(myelin[group_nbr]).unsqueeze(1)  # [n_roi, 1]
    rsfc_gradient = np.array(grouped_mats['rsfc_groups'])
    rsfc_gradient = torch.as_tensor(rsfc_gradient[group_nbr]).unsqueeze(1)
    sc_mat = np.array(grouped_mats['sc_groups'])
    sc_mat = torch.as_tensor(sc_mat[group_nbr])
    emp_fc = np.array(grouped_mats['fc_groups'])
    emp_fc = torch.as_tensor(emp_fc[group_nbr])
    emp_fcd_cum = np.array(grouped_mats['fcd_groups'])
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum[group_nbr])
    emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)  # [bins, 1]

    parent_dir = os.path.join(LOG_DIR, 'HCPYA', 'DL_dataset', 'Desikan')
    epochs = 100

    save_param_dir = os.path.join(
        parent_dir, f'{mode}/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
    seed = None
    state = train_help_function(config=config,
                                myelin=myelin,
                                RSFC_gradient=rsfc_gradient,
                                sc_euler=sc_mat,
                                emp_fc=emp_fc,
                                emp_fcd_cum=emp_fcd_cum,
                                train_save_dir=save_param_dir,
                                num_epochs=epochs,
                                dl_pfic_range=[],
                                euler_pfic_range=np.arange(0, 100),
                                dl_rfic_range=[],
                                euler_rfic_range=[],
                                query_wei_range='Uniform',
                                opportunities=10,
                                next_epoch=0,
                                seed=seed)
    return state


def apply_large_group_Yan100(mode, trial_nbr, seed_nbr, epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/tzeng/storage/Python/MFMApplication/configs/general/config_hcpya_Yan100.ini'
    )

    trial_nbr = int(trial_nbr)
    seed_nbr = int(seed_nbr)

    individual_mats_path = '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029'
    myelin_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'myelin_Yan100_1029.mat'))
    myelin_indi = myelin_indi['myelin_Yan100']
    rsfc_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'rsfc_Yan100_1029.mat'))
    rsfc_indi = rsfc_indi['rsfc_Yan100']
    sc_indi = spio.loadmat(
        os.path.join(individual_mats_path, 'sc_Yan100_1029.mat'))
    sc_indi = sc_indi['sc_Yan100']

    fc_1029 = spio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fc_Yan100_1029.mat'
    )
    fc_1029 = fc_1029['fc_Yan100']
    fcd_1029 = spio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/fcd_cdf_Yan100_1029.mat'
    )
    fcd_1029 = fcd_1029['fcd_cdf_Yan100']

    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/HCPYA/group_860_1029'
    epochs = 100

    # Always using train set myelin and RSFC gradient
    myelin = np.nanmean(myelin_indi[0:680], axis=0)
    myelin = torch.as_tensor(myelin).unsqueeze(1)
    rsfc_gradient = np.nanmean(rsfc_indi[0:680], axis=0)
    rsfc_gradient = torch.as_tensor(rsfc_gradient).unsqueeze(1)

    if mode == 'train':
        subrange = [860, 917]
        sc_mat = tzeng_func.tzeng_group_SC_matrices(
            sc_indi[subrange[0]:subrange[1]])
        sc_mat = torch.as_tensor(sc_mat)
        emp_fc = np.array(fc_1029[subrange[0]:subrange[1]])
        emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)
        emp_fc = torch.as_tensor(emp_fc)
        emp_fcd_cum = torch.as_tensor(fcd_1029[subrange[0]:subrange[1]].astype(
            np.float64))
        emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
        emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_euler=sc_mat,
                                    emp_fc=emp_fc,
                                    emp_fcd_cum=emp_fcd_cum,
                                    train_save_dir=save_param_dir,
                                    num_epochs=epochs,
                                    dl_pfic_range=np.arange(0, 100),
                                    euler_pfic_range=[],
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='Uniform',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        subrange = [917, 973]
        sc_mat = tzeng_func.tzeng_group_SC_matrices(
            sc_indi[subrange[0]:subrange[1]])
        sc_mat = torch.as_tensor(sc_mat)
        sc_euler = sc_mat / torch.max(sc_mat) * 0.02
        emp_fc = np.array(fc_1029[subrange[0]:subrange[1]])
        emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)
        emp_fc = torch.as_tensor(emp_fc)
        emp_fcd_cum = torch.as_tensor(fcd_1029[subrange[0]:subrange[1]].astype(
            np.float64))
        emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
        emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}')
        val_dir = os.path.join(parent_dir,
                               f'validation/trial{trial_nbr}/seed{seed_nbr}')
        mfm_validator = CMAESValidator(config, save_param_dir, val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, emp_fc,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, emp_fc, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        subrange = [973, 1029]
        sc_mat = tzeng_func.tzeng_group_SC_matrices(
            sc_indi[subrange[0]:subrange[1]])
        sc_mat = torch.as_tensor(sc_mat)
        sc_euler = sc_mat / torch.max(sc_mat) * 0.02
        emp_fc = np.array(fc_1029[subrange[0]:subrange[1]])
        emp_fc = tzeng_func.tzeng_fisher_average(emp_fc)
        emp_fc = torch.as_tensor(emp_fc)
        emp_fcd_cum = torch.as_tensor(fcd_1029[subrange[0]:subrange[1]].astype(
            np.float64))
        emp_fcd_cum = torch.mean(emp_fcd_cum, dim=0)
        emp_fcd_cum = (emp_fcd_cum / emp_fcd_cum[-1]).unsqueeze(1)

        # val_dirs = [os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}') for i in np.arange(1, seed_nbr + 1)]
        val_dirs = [
            os.path.join(parent_dir, f'validation/trial{trial_nbr}/seed{i}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(parent_dir,
                                f'test/trial{trial_nbr}/seed{seed_nbr}')
        mfm_tester = CMAESTester(config,
                                 val_dirs,
                                 test_dir,
                                 train_num_epochs=epochs)
        mfm_tester.test_old(sc_euler, emp_fc, emp_fcd_cum)

    return 0


if __name__ == "__main__":
    print(datetime.datetime.now(), ': main program start'.upper(), flush=True)

    if torch.cuda.is_available():
        print("Current GPU: ",
              torch.cuda.get_device_name(torch.cuda.current_device()))

    # * lightweight tasks
    # for trial_idx in [31, 32, 33, 34]:
    #     for seed_idx in [3]:
    #         apply_on_all_participants(phase='best_from_train',
    #                                   trial_idx=trial_idx,
    #                                   seed_idx=seed_idx)

    # * heavy tasks to be submitted to the scheduler
    USE_MULTIPROC = False
    phases = ['train', 'best_from_train']
    for phase in phases:
        if USE_MULTIPROC:
            set_start_method('spawn')
            num_thread = n_seeds = int(sys.argv[2])
            trial_idx = int(sys.argv[1])
            with Pool(num_thread) as p:
                p.starmap(apply_on_all_participants,
                          [(phase, trial_idx, seed_idx)
                           for seed_idx in range(1, n_seeds + 1)])
        else:
            apply_on_all_participants(phase=phase,
                                      trial_idx=sys.argv[1],
                                      seed_idx=sys.argv[2])
    # save_dir = os.path.join(LOG_DIR, 'HCPYA', 'all_participants')
    # if os.path.exists(
    #         os.path.join(save_dir, 'train', f'trial{sys.argv[1]}',
    #                      f'seed{sys.argv[2]}', 'param_save_epoch99.pth')):
    #     apply_on_all_participants(phase='validation',
    #                               trial_idx=sys.argv[1],
    #                               seed_idx=sys.argv[2])
    #     apply_on_all_participants(phase='val_best',
    #                               trial_idx=sys.argv[1],
    #                               seed_idx=sys.argv[2])
    #     apply_on_all_participants(phase='simulate_fc_fcd',
    #                               trial_idx=sys.argv[1],
    #                               seed_idx=sys.argv[2])

    # apply_large_group(mode='train',
    #                   trial_nbr=sys.argv[1],
    #                   seed_nbr=sys.argv[2],
    #                   epoch=None)
    # apply_large_group(mode='validation', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2], epoch=sys.argv[3])
    # apply_large_group(mode='test', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2])
    # apply_large_group(mode='val_best', trial_nbr=3, seed_nbr=2)
    # apply_large_group(mode='simulate_fc_fcd', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2])
    # apply_large_group(mode='simulate_fc_fcd_mat', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2])

    # generate_dl_dataset_group(mode=sys.argv[1],
    #                           group_nbr=sys.argv[2],
    #                           trial_nbr=sys.argv[3],
    #                           seed_nbr=sys.argv[4])

    # apply_large_group_Yan100(mode='train', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2], epoch=None)
    # apply_large_group_Yan100(mode='validation', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2], epoch=sys.argv[3])
    # apply_large_group_Yan100(mode='test', trial_nbr=sys.argv[1], seed_nbr=sys.argv[2])
