import configparser
import scipy.io as spio
import torch
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(1, '/home/ftian/storage/projects/MFM_exploration')
from src.scripts.Hybrid_CMA_ES import DLVersionCMAESValidator, DLVersionCMAESTester, \
    get_EI_ratio, train_help_function


def apply_individual(mode, sub_nbr, trial_nbr, seed_nbr):
    config = configparser.ConfigParser()
    config.read(
        '/home/tzeng/storage/Python/MFMApplication/configs/general/config_pnc.ini'
    )

    sub_nbr = int(sub_nbr)
    trial_nbr = int(trial_nbr)
    seed_nbr = int(seed_nbr)

    # Following age order
    subjects_id = pd.read_csv(
        '/mnt/isilon/CSC1/Yeolab/Data/PNC/documentation/rest_subject_age.csv',
        sep=',',
        header=None,
        index_col=False)
    subjects_id = np.array(subjects_id)[:, 0]
    subject_id = subjects_id[sub_nbr]
    print("Subject id: ", subject_id)

    group_mats = spio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    myelin = torch.as_tensor(group_mats['myelin_group_1029'])
    rsfc_gradient = torch.as_tensor(group_mats['rsfc_group_1029'])
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    fc_emp = pd.read_csv(
        f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_vol/rest/FC/sub-{subject_id}.csv',
        header=None,
        index_col=False)
    fc_emp = torch.as_tensor(np.array(fc_emp))
    emp_fcd_cum = spio.loadmat(
        f'/home/shaoshi.z/storage/MFM/PNC/desikan_FC_FCD_from_vol/rest/FCD/sub-{subject_id}.mat'
    )
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum['FCD_CDF'].astype(np.float64)).T
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    # Setting parent directory
    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/individual'
    epochs = 50

    # Main part, if statement for mode
    if mode == 'train':
        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}/sub{sub_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_mat=sc_mat,
                                    fc_emp=fc_emp,
                                    emp_fcd_cum=emp_fcd_cum,
                                    save_param_dir=save_param_dir,
                                    epochs=epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 50),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='PNC2',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'EI':
        epochs_range = range(0, epochs)

        save_param_dir = os.path.join(
            parent_dir, f'train/trial{trial_nbr}/seed{seed_nbr}/sub{sub_nbr}')
        if not os.path.exists(os.path.join(save_param_dir, 'final_state.pth')):
            print("This subject does not have final state file.")
            return 1
        EI_save_dir = os.path.join(
            parent_dir, f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}')
        if not os.path.exists(EI_save_dir):
            os.makedirs(EI_save_dir)

        euler_epochs = len(epochs_range)
        parameter_dim = 205
        parameter_sets = torch.zeros(parameter_dim, euler_epochs)
        loss_sets = torch.ones(euler_epochs) * 3
        for epoch in epochs_range:
            parameter_path = os.path.join(save_param_dir,
                                          f'param_save_epoch{epoch}.pth')
            d = torch.load(parameter_path)

            valid_param_list_pre = d['valid_param_list']
            parameter = d['parameter']
            parameter = parameter[:, valid_param_list_pre]
            total_loss = torch.sum(d['FC_FCD_loss'], dim=1)  # [xxx]
            best_param_ind = torch.argmin(total_loss)
            parameter = parameter[:, best_param_ind]  # [parameter_dim]

            record_ind = epoch - epochs_range[0]
            parameter_sets[:, record_ind] = parameter
            loss_sets[record_ind] = total_loss[best_param_ind]
        best_sets_ind = torch.argmin(loss_sets)
        print("Best train loss: ", loss_sets[best_sets_ind])
        best_parameter = parameter_sets[:, best_sets_ind].unsqueeze(1)

        EI_save_path = os.path.join(EI_save_dir, f'sub{sub_nbr}.pth')
        get_EI_ratio(config,
                     save_path=EI_save_path,
                     parameter=best_parameter,
                     param_dup=50,
                     sc_euler=sc_euler)
        print("Done.")

    return 0


def apply_age_group(mode, group_nbr, trial_nbr, seed_nbr, epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/ftian/storage/projects/MFM_exploration/configs/model/config_pnc.ini'
    )

    group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/age_results/input/{group_nbr}'
    postfix = 'train' if mode == 'train' else 'validation'

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
        pd.read_csv(os.path.join(group_mats_path, f'SC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    sc_mat = torch.as_tensor(sc_mat)
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    fc_emp = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'FC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    fc_emp = torch.as_tensor(fc_emp)
    emp_fcd_cum = spio.loadmat(
        os.path.join(group_mats_path, f'FCD_{postfix}.mat'))
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum[f'FCD_{postfix}'].astype(
        np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/age_group'
    epochs = 50

    if mode == 'train':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_mat=sc_mat,
                                    fc_emp=fc_emp,
                                    emp_fcd_cum=emp_fcd_cum,
                                    save_param_dir=save_param_dir,
                                    epochs=epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 50),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='PNC',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        val_dir = os.path.join(
            parent_dir,
            f'validation/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_validator = DLVersionCMAESValidator(config, save_param_dir,
                                                val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, fc_emp,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, fc_emp, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        val_dirs = [
            os.path.join(
                parent_dir,
                f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_tester = DLVersionCMAESTester(config,
                                          val_dirs,
                                          test_dir,
                                          trained_epochs=epochs)
        mfm_tester.select_best_from_val()

    elif mode == 'EI':
        param_path = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}',
            'val_results.pth')
        if not os.path.exists(param_path):
            print("This group doesn't have validation results.")
            return 1
        best_parameter = torch.load(param_path)['parameter']
        EI_save_dir = os.path.join(
            parent_dir, f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}')
        EI_save_path = os.path.join(parent_dir, EI_save_dir,
                                    f'group{group_nbr}.pth')
        if not os.path.exists(EI_save_dir):
            os.makedirs(EI_save_dir)

        get_EI_ratio(config,
                     save_path=EI_save_path,
                     parameter=best_parameter,
                     param_dup=50,
                     sc_euler=sc_euler)
        print("EI ratio Done.")

    return 0


def apply_overall_acc_group(mode,
                            group_nbr,
                            trial_nbr,
                            seed_nbr,
                            performance_group,
                            epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/ftian/storage/projects/MFM_exploration/configs/model/config_pnc.ini'
    )

    group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/behavior_results/input_overall_acc/{performance_group}/{group_nbr}'

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

    postfix = 'train' if mode == 'train' else 'validation'
    sc_mat = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'SC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    sc_mat = torch.as_tensor(sc_mat)
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    fc_emp = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'FC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    fc_emp = torch.as_tensor(fc_emp)
    emp_fcd_cum = spio.loadmat(
        os.path.join(group_mats_path, f'FCD_{postfix}.mat'))
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum[f'FCD_{postfix}'].astype(
        np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    parent_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/{performance_group}'
    epochs = 50

    if mode == 'train':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_mat=sc_mat,
                                    fc_emp=fc_emp,
                                    emp_fcd_cum=emp_fcd_cum,
                                    save_param_dir=save_param_dir,
                                    epochs=epochs,
                                    dl_pfic_range=[],
                                    euler_pfic_range=np.arange(0, 50),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='PNC',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        val_dir = os.path.join(
            parent_dir,
            f'validation/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_validator = DLVersionCMAESValidator(config, save_param_dir,
                                                val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, fc_emp,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, fc_emp, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        seed_nbr = int(seed_nbr)
        # val_dirs = [
        #     os.path.join(
        #         parent_dir,
        #         f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
        #     for i in np.arange(1, seed_nbr + 1)
        # ]
        val_dirs = [
            os.path.join(
                parent_dir,
                f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_tester = DLVersionCMAESTester(config,
                                          val_dirs,
                                          test_dir,
                                          trained_epochs=epochs)
        mfm_tester.select_best_from_val()

    elif mode == 'EI':
        param_path = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}',
            'val_results.pth')
        if not os.path.exists(param_path):
            print("This group doesn't have validation results.")
            return 1
        best_parameter = torch.load(param_path)['parameter']
        EI_save_dir = os.path.join(
            parent_dir, f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}')
        EI_save_path = os.path.join(parent_dir, EI_save_dir,
                                    f'group{group_nbr}.pth')
        if not os.path.exists(EI_save_dir):
            os.makedirs(EI_save_dir)

        get_EI_ratio(config,
                     save_path=EI_save_path,
                     parameter=best_parameter,
                     param_dup=50,
                     sc_euler=sc_euler)
        print("EI ratio Done.")

    return 0


def apply_age_group_Yan100(mode, group_nbr, trial_nbr, seed_nbr, epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/tzeng/storage/Python/MFMApplication/configs/general/config_pnc_Yan100.ini'
    )

    group_mats_path = f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/Yan/age/input/{group_nbr}'
    postfix = 'train' if mode == 'train' else 'validation'

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
        pd.read_csv(os.path.join(group_mats_path, f'SC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    sc_mat = torch.as_tensor(sc_mat)
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [n_roi, n_roi] for Euler integration

    fc_emp = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'FC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    fc_emp = torch.as_tensor(fc_emp)
    emp_fcd_cum = spio.loadmat(
        os.path.join(group_mats_path, f'FCD_{postfix}.mat'))
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum[f'FCD_{postfix}'].astype(
        np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    parent_dir = '/home/ftian/storage/projects/MFM_exploration/logs/PNC/age_group'
    epochs = 50

    if mode == 'train':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_mat=sc_mat,
                                    fc_emp=fc_emp,
                                    emp_fcd_cum=emp_fcd_cum,
                                    save_param_dir=save_param_dir,
                                    epochs=epochs,
                                    dl_pfic_range=np.arange(0, 45),
                                    euler_pfic_range=np.arange(45, 50),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='PNC',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        val_dir = os.path.join(
            parent_dir,
            f'validation/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_validator = DLVersionCMAESValidator(config, save_param_dir,
                                                val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, fc_emp,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, fc_emp, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        val_dirs = [
            os.path.join(
                parent_dir,
                f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_tester = DLVersionCMAESTester(config,
                                          val_dirs,
                                          test_dir,
                                          trained_epochs=epochs)
        mfm_tester.select_best_from_val()

    elif mode == 'EI':
        param_path = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}',
            'val_results.pth')
        if not os.path.exists(param_path):
            print("This group doesn't have validation results.")
            return 1
        best_parameter = torch.load(param_path)['parameter']
        EI_save_dir = os.path.join(
            parent_dir, f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}')
        EI_save_path = os.path.join(parent_dir, EI_save_dir,
                                    f'group{group_nbr}.pth')
        if not os.path.exists(EI_save_dir):
            os.makedirs(EI_save_dir)

        get_EI_ratio(config,
                     save_path=EI_save_path,
                     parameter=best_parameter,
                     param_dup=50,
                     sc_euler=sc_euler)
        print("EI ratio Done.")

    return 0


def apply_overall_acc_group_Yan100(mode,
                                   group_nbr,
                                   trial_nbr,
                                   seed_nbr,
                                   performance_group,
                                   epoch=None):
    config = configparser.ConfigParser()
    config.read(
        '/home/tzeng/storage/Python/MFMApplication/configs/general/config_pnc_Yan100.ini'
    )

    # performance_group: ['high', 'low']

    group_mats_path = \
        f'/home/shaoshi.z/storage/MFM/PNC/rest_from_surface/Yan/cognition/{performance_group}/input/{group_nbr}'

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

    postfix = 'train' if mode == 'train' else 'validation'
    sc_mat = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'SC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    sc_mat = torch.as_tensor(sc_mat)
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    fc_emp = np.array(
        pd.read_csv(os.path.join(group_mats_path, f'FC_{postfix}.csv'),
                    header=None,
                    index_col=False))
    fc_emp = torch.as_tensor(fc_emp)
    emp_fcd_cum = spio.loadmat(
        os.path.join(group_mats_path, f'FCD_{postfix}.mat'))
    emp_fcd_cum = torch.as_tensor(emp_fcd_cum[f'FCD_{postfix}'].astype(
        np.float64))
    emp_fcd_cum = emp_fcd_cum / emp_fcd_cum[-1, 0]

    parent_dir = f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/{performance_group}'
    epochs = 50

    if mode == 'train':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        seed = None
        state = train_help_function(config=config,
                                    myelin=myelin,
                                    RSFC_gradient=rsfc_gradient,
                                    sc_mat=sc_mat,
                                    fc_emp=fc_emp,
                                    emp_fcd_cum=emp_fcd_cum,
                                    save_param_dir=save_param_dir,
                                    epochs=epochs,
                                    dl_pfic_range=np.arange(0, 45),
                                    euler_pfic_range=np.arange(45, 50),
                                    dl_rfic_range=[],
                                    euler_rfic_range=[],
                                    query_wei_range='PNC',
                                    opportunities=10,
                                    next_epoch=0,
                                    seed=seed)
        print("Exit state: ", state)

    elif mode == 'validation':
        save_param_dir = os.path.join(
            parent_dir,
            f'train/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        val_dir = os.path.join(
            parent_dir,
            f'validation/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_validator = DLVersionCMAESValidator(config, save_param_dir,
                                                val_dir)
        if epoch is None:
            for ep in range(0, epochs):
                mfm_validator.val_best_parameters(sc_euler, fc_emp,
                                                  emp_fcd_cum, ep)
        else:
            epoch = int(epoch)
            mfm_validator.val_best_parameters(sc_euler, fc_emp, emp_fcd_cum,
                                              epoch)

    elif mode == 'test':
        seed_nbr = int(seed_nbr)
        # val_dirs = [
        #     os.path.join(
        #         parent_dir,
        #         f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
        #     for i in np.arange(1, seed_nbr + 1)
        # ]
        val_dirs = [
            os.path.join(
                parent_dir,
                f'validation/trial{trial_nbr}/seed{i}/group{group_nbr}')
            for i in [seed_nbr]
        ]
        test_dir = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}')
        mfm_tester = DLVersionCMAESTester(config,
                                          val_dirs,
                                          test_dir,
                                          trained_epochs=epochs)
        mfm_tester.select_best_from_val()

    elif mode == 'EI':
        param_path = os.path.join(
            parent_dir,
            f'test/trial{trial_nbr}/seed{seed_nbr}/group{group_nbr}',
            'val_results.pth')
        if not os.path.exists(param_path):
            print("This group doesn't have validation results.")
            return 1
        best_parameter = torch.load(param_path)['parameter']
        EI_save_dir = os.path.join(
            parent_dir, f'EI_ratio/trial{trial_nbr}/seed{seed_nbr}')
        EI_save_path = os.path.join(parent_dir, EI_save_dir,
                                    f'group{group_nbr}.pth')
        if not os.path.exists(EI_save_dir):
            os.makedirs(EI_save_dir)

        get_EI_ratio(config,
                     save_path=EI_save_path,
                     parameter=best_parameter,
                     param_dup=50,
                     sc_euler=sc_euler)
        print("EI ratio Done.")

    return 0


if __name__ == "__main__":
    # for g_nbr in np.arange(1, 15):
    #     apply_overall_acc_group(mode='test', group_nbr=g_nbr, trial_nbr=4, seed_nbr=5)
    # apply_overall_acc_group(mode='EI', group_nbr=sys.argv[1], trial_nbr=sys.argv[2], seed_nbr=sys.argv[3])

    # apply_individual(mode='train', sub_nbr=sys.argv[1], trial_nbr=sys.argv[2], seed_nbr=sys.argv[3])
    # apply_individual(mode='EI', sub_nbr=sys.argv[1], trial_nbr=sys.argv[2], seed_nbr=sys.argv[3])

    target = sys.argv[1]

    if target == "age_group":
        #  -- Age group --
        g_nbr = sys.argv[2]
        t_nbr = sys.argv[3]
        se_nbr = sys.argv[4]
        apply_age_group(mode='train',
                        group_nbr=g_nbr,
                        trial_nbr=t_nbr,
                        seed_nbr=se_nbr)
        if os.path.exists(
                '/home/ftian/storage/projects/MFM_exploration/logs/PNC/age_group/train/'
                f'trial{t_nbr}/seed{se_nbr}/group{g_nbr}/param_save_epoch49.pth'
        ):
            apply_age_group(mode='validation',
                            group_nbr=g_nbr,
                            trial_nbr=t_nbr,
                            seed_nbr=se_nbr)
            apply_age_group(mode='test',
                            group_nbr=g_nbr,
                            trial_nbr=t_nbr,
                            seed_nbr=se_nbr)
            apply_age_group(mode='EI',
                            group_nbr=g_nbr,
                            trial_nbr=t_nbr,
                            seed_nbr=se_nbr)
    elif "overall_acc_group" in target:
        # -- Overall Acc group --
        perform_group_name = 'low'
        if "high" in target:
            perform_group_name = 'high'
        g_nbr = sys.argv[2]
        t_nbr = sys.argv[3]
        se_nbr = sys.argv[4]
        apply_overall_acc_group(mode='train',
                                group_nbr=g_nbr,
                                trial_nbr=t_nbr,
                                seed_nbr=se_nbr,
                                performance_group=perform_group_name)
        if os.path.exists(
                f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/'
                f'{perform_group_name}/train/trial{t_nbr}/seed{se_nbr}/group{g_nbr}/param_save_epoch49.pth'
        ):
            apply_overall_acc_group(mode='validation',
                                    group_nbr=g_nbr,
                                    trial_nbr=t_nbr,
                                    seed_nbr=se_nbr,
                                    performance_group=perform_group_name)
            apply_overall_acc_group(mode='test',
                                    group_nbr=g_nbr,
                                    trial_nbr=t_nbr,
                                    seed_nbr=se_nbr,
                                    performance_group=perform_group_name)
            apply_overall_acc_group(mode='EI',
                                    group_nbr=g_nbr,
                                    trial_nbr=t_nbr,
                                    seed_nbr=se_nbr,
                                    performance_group=perform_group_name)
    else:
        raise Exception("Unrecognized target name!")

    # # -- Age group Yan100 --
    # g_nbr = sys.argv[2]
    # t_nbr = sys.argv[3]
    # se_nbr = sys.argv[4]
    # apply_age_group_Yan100(mode='train',
    #                        group_nbr=g_nbr,
    #                        trial_nbr=t_nbr,
    #                        seed_nbr=se_nbr)
    # if os.path.exists(
    #         '/home/ftian/storage/projects/MFM_exploration/logs/PNC/age_group/train/'
    #         f'trial{t_nbr}/seed{se_nbr}/group{g_nbr}/param_save_epoch49.pth'):
    #     apply_age_group_Yan100(mode='validation',
    #                            group_nbr=g_nbr,
    #                            trial_nbr=t_nbr,
    #                            seed_nbr=se_nbr)
    #     apply_age_group_Yan100(mode='test',
    #                            group_nbr=g_nbr,
    #                            trial_nbr=t_nbr,
    #                            seed_nbr=se_nbr)
    #     apply_age_group_Yan100(mode='EI',
    #                            group_nbr=g_nbr,
    #                            trial_nbr=t_nbr,
    #                            seed_nbr=se_nbr)
    # # -- End --

    # # -- Overall Acc Group Yan100 --
    # perform_group_name = 'low'
    # g_nbr = sys.argv[2]
    # t_nbr = sys.argv[3]
    # se_nbr = sys.argv[4]
    # apply_overall_acc_group_Yan100(mode='train',
    #                                group_nbr=g_nbr,
    #                                trial_nbr=t_nbr,
    #                                seed_nbr=se_nbr,
    #                                performance_group=perform_group_name)
    # if os.path.exists(
    #         f'/home/ftian/storage/projects/MFM_exploration/logs/PNC/overall_acc_group/{perform_group_name}/train/trial{t_nbr}/seed{se_nbr}/group{g_nbr}/param_save_epoch49.pth'  # noqa
    # ):
    #     apply_overall_acc_group_Yan100(mode='validation',
    #                                    group_nbr=g_nbr,
    #                                    trial_nbr=t_nbr,
    #                                    seed_nbr=se_nbr,
    #                                    performance_group=perform_group_name)
    #     apply_overall_acc_group_Yan100(mode='test',
    #                                    group_nbr=g_nbr,
    #                                    trial_nbr=t_nbr,
    #                                    seed_nbr=se_nbr,
    #                                    performance_group=perform_group_name)
    #     apply_overall_acc_group_Yan100(mode='EI',
    #                                    group_nbr=g_nbr,
    #                                    trial_nbr=t_nbr,
    #                                    seed_nbr=se_nbr,
    #                                    performance_group=perform_group_name)
    # # -- End --
