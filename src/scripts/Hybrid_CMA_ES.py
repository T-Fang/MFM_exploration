import pandas as pd
import numpy as np
import torch
import torch.distributions as td
import os

from src.models.model_predictor_classifier import ClassifyNanModel_2, PredictLossModel_1, ClassifyNanModel_Yan100, PredictLossModel_1_Yan100
from src.models.mfm_2014 import MfmModel2014
from src.utils.tzeng_func_torch import parameterize_myelin_rsfc
from src.utils.CBIG_func_torch import CBIG_corr


def csv2tensor(csv_path):
    content = pd.read_csv(csv_path, header=None, index_col=False)
    content = np.array(content)
    return torch.as_tensor(content, dtype=torch.double)


def regularization_loss(parameter,
                        w_wee,
                        w_wei,
                        w_sigma,
                        concat_mat,
                        pinv_concat_mat=None):
    """rFIC regularization loss

    Args:
        parameters (tensor): wEE, wEI, G, sigma. [3N+1, param_sets]
        w_wee (scalar): the weight for the regularization term for wee_regularization_loss.
        w_wei (scalar): the weight for the regularization term for wei_regularization_loss.
        w_sigma (scalar): the weight for the regularization term for sigma_regularization_loss.
        concat_mat (tensor): [bias (ones_like myelin), myelin, RSFC gradient]
        pinv_concat_mat (tensor, optional): pinv(concat_mat). Defaults to None.

    Returns:
        (tensors): [param_sets, 3]. wEE, wEI, sigma regularization loss terms
    """
    N = concat_mat.shape[0]  # ROI num
    if pinv_concat_mat is None:
        pinv_concat_mat = torch.linalg.pinv(concat_mat)
    beta_wee = torch.matmul(pinv_concat_mat, parameter[0:N])  # [3, param_sets]
    beta_wei = torch.matmul(pinv_concat_mat, parameter[N:2 * N])
    beta_sigma = torch.matmul(pinv_concat_mat, parameter[2 * N + 1:])
    wee_hat = torch.matmul(concat_mat, beta_wee)  # [N, param_sets]
    wei_hat = torch.matmul(concat_mat, beta_wei)
    sigma_hat = torch.matmul(concat_mat, beta_sigma)

    wee_regularization_loss = w_wee * torch.mean(
        torch.square(parameter[0:N] - wee_hat), dim=0)  # [param_sets]
    wei_regularization_loss = w_wei * torch.mean(
        torch.square(parameter[N:2 * N] - wei_hat), dim=0)
    sigma_regularization_loss = w_sigma * torch.mean(
        torch.square(parameter[2 * N + 1:] - sigma_hat), dim=0)

    return torch.hstack((wee_regularization_loss.unsqueeze(1),
                         wei_regularization_loss.unsqueeze(1),
                         sigma_regularization_loss.unsqueeze(1)))


def get_r_E_reg_loss(r_E, target_r_E, loss_type='L2'):
    """
    firing rate r_E regularization loss using the specified loss, the target firing rate comes from mfm_model.r_E
    Args:
        r_E: average firing rate from the simulation [N, param_sets]
        target_r_E: target firing rate from the mfm_model.r_E (scalar)
        loss_type: 'L2' or 'L1'
    Returns:
        r_E_reg_loss: r_E regularization loss [param_sets, ]
    """
    if loss_type == 'L2':
        r_E_reg_loss = torch.mean(torch.square(r_E - target_r_E), dim=0)
    elif loss_type == 'L1':
        r_E_reg_loss = torch.mean(torch.abs(r_E - target_r_E), dim=0)
    else:
        raise NotImplementedError
    return r_E_reg_loss


def select_best_parameter_from_savedict(save_dict):
    if 'corr_loss' in save_dict:
        total_loss = save_dict['corr_loss'] + save_dict['L1_loss'] + save_dict[
            'ks_loss']
    elif 'pred_loss' in save_dict:
        total_loss = torch.sum(save_dict['pred_loss'], dim=1)
    elif 'FC_FCD_loss' in save_dict:
        total_loss = torch.sum(save_dict['FC_FCD_loss'], dim=1)
    else:
        raise Exception(
            "Error in parameter saving. Key 'corr_loss' and 'pred_loss' both not exists."
        )

    index_in_valid = torch.argmin(total_loss)
    index_min = save_dict['valid_param_list'][index_in_valid]
    if 'parameter' in save_dict:
        best_parameter = save_dict['parameter'][:, index_min]
    elif 'param_10' in save_dict:
        best_parameter = parameterize_myelin_rsfc(
            save_dict['param_10'][:, index_min])
    else:
        raise Exception(
            "Error in parameter saving. Key 'parameter' and 'param_10' both not exists."
        )

    return best_parameter


# def get_euler_hyperparameters(dataset_name):
#     if dataset_name == 'HCP':
#         t_epochlong = 14.4
#         burn_in_minute = 2.4
#         t_bold = 0.72
#         window_size = 83
#     elif dataset_name == 'PNC':
#         t_epochlong = 6.2
#         burn_in_minute = 1.2
#         t_bold = 3
#         window_size = 20
#     elif dataset_name == 'HCP_Dev':
#         t_epochlong = 6.36
#         burn_in_minute = 1.06
#         t_bold = 0.8
#         window_size = 75
#     elif dataset_name == 'HCA':
#         t_epochlong = 6.36
#         burn_in_minute = 1.06
#         t_bold = 0.8
#         window_size = 75
#     elif dataset_name == 'Alprazolam':
#         t_epochlong = 10.5
#         burn_in_minute = 1.5
#         t_bold = 3
#         window_size = 20
#     elif dataset_name == 'test':
#         t_epochlong = 0.1
#         burn_in_minute = 0.1
#         t_bold = 0.1
#         window_size = 5
#     else:
#         raise Exception("Your specified dataset name doesn't exist.")
#     return t_epochlong, burn_in_minute, t_bold, window_size


@DeprecationWarning
def get_dl_model_path(dl_model_key):
    if dl_model_key == 'HCP_group' or 'PNC_group':
        classifier_path = '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/classifier/hcp_group_2_1.pth'
        predictor_path = '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/predictor/hcp_group_5_1.pth'
    elif dl_model_key == 'HCP_individual':
        classifier_path = '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/classifier/hcp_individual_2_1.pth'
        predictor_path = '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/predictor/hcp_individual_5_1.pth'
    else:
        classifier_path = None
        predictor_path = None
    # Previous
    """
    HCP individual: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/classifier/classifier2_7_hcp_individual.pth'
    PNC individual: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/classifier/classifier2_7_finetune_3.pth'
    HCP/PNC group: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/classifier/classifier2_7.pth'

    HCP individual: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/predictor/predictor5_5_hcp_individual.pth'
    PNC individual: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/predictor/predictor5_5_finetune_8.pth'
    PNC group: '/home/tzeng/storage/Python/MFMApplication/ModelSave/ModelDict/predictor/predictor5_5_finetune_6.pth' or
                not finetuned version (hcp group) (better)
    """
    return classifier_path, predictor_path


def train_help_function(config,
                        myelin,
                        RSFC_gradient,
                        sc_mat,
                        fc_emp,
                        emp_fcd_cum,
                        save_param_dir,
                        epochs,
                        dl_pfic_range,
                        euler_pfic_range,
                        dl_rfic_range,
                        euler_rfic_range,
                        query_wei_range,
                        opportunities,
                        next_epoch,
                        seed=None):
    mfm_trainer = DLVersionCMAESForward(config=config,
                                        myelin=myelin,
                                        RSFC_gradient=RSFC_gradient,
                                        sc_mat=sc_mat,
                                        fc_emp=fc_emp,
                                        emp_fcd_cum=emp_fcd_cum,
                                        save_param_dir=save_param_dir,
                                        epochs=epochs,
                                        dl_pfic_range=dl_pfic_range,
                                        euler_pfic_range=euler_pfic_range,
                                        dl_rfic_range=dl_rfic_range,
                                        euler_rfic_range=euler_rfic_range,
                                        query_wei_range=query_wei_range)
    opportunities = opportunities
    next_epoch = next_epoch
    for i in range(opportunities):
        state = mfm_trainer.train_hybrid_pFIC_rFIC(seed=seed,
                                                   next_epoch=next_epoch)
        # state: 0 success, 1 restart, 2 fail
        if state == 0:
            break
        elif state == 1:
            if i == opportunities - 1:
                print(
                    f'Having tried for {opportunities} times, but still no available parameter.'
                )
                break
            print("Restart ...")
        elif state == 2:
            print("CMA-ES broke during middle epochs. Terminate.")
            break
    print(f"In total {i + 1} random seeds.")
    print("The End.")
    return state


class DLVersionCMAESForward:

    def __init__(self, config, myelin, RSFC_gradient, sc_mat, fc_emp,
                 emp_fcd_cum, save_param_dir, epochs, dl_pfic_range,
                 euler_pfic_range, dl_rfic_range, euler_rfic_range,
                 query_wei_range):
        """Initialize Hybrid CMA-ES trainer

        Args:
            dataset_name (str): ['HCP', 'PNC', 'HCP_Dev']. The name of dataset. To get the euler hyperparameters
            myelin (tensor): [ROIs, 1]
            RSFC_gradient (tensor): [ROIs, 1]
            sc_mat (tensor): [ROIs, ROIs]
            fc_emp (tensor): [ROIs, ROIs]
            emp_fcd_cum (tensor): [bins, 1], has been normalized (largest is 1)
            save_param_dir (str): the parameters saving directory
            epochs (int): total epochs

        """

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            self.device = 'cuda'
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print(torch.cuda.get_device_name())
            myelin = myelin.cuda()
            RSFC_gradient = RSFC_gradient.cuda()
            sc_mat = sc_mat.cuda()
            fc_emp = fc_emp.cuda()
            emp_fcd_cum = emp_fcd_cum.cuda()
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.device = 'cpu'

        self.config = config

        self.N = myelin.shape[0]
        self.myelin = myelin  # [N, 1]
        self.RSFC_gradient = RSFC_gradient  # [N, 1]
        self.concat_mat = torch.hstack(
            (torch.ones_like(myelin), myelin, RSFC_gradient))  # [N, 3]
        self.pinv_concat_mat = torch.linalg.pinv(self.concat_mat)  # [3, N]

        sc_mask = torch.triu(torch.ones(self.N, self.N, dtype=torch.bool),
                             1)  # Upper triangle
        self.sc_dl = torch.as_tensor(sc_mat[sc_mask])  # [N * (N - 1) / 2]
        self.sc_euler = sc_mat / torch.max(
            sc_mat) * 0.02  # [N, N] for Euler integration
        self.fc_dl = torch.as_tensor(fc_emp[sc_mask])  # [N * (N - 1) / 2]
        self.fc_euler = torch.as_tensor(fc_emp)
        self.fcd_dl = torch.diff(emp_fcd_cum.squeeze() * 100,
                                 dim=0,
                                 prepend=torch.as_tensor([0]))  # [bins]
        self.fcd_cum_euler = emp_fcd_cum  # [bins, 1]

        self.parameter_dim = 3 * self.N + 1  # 3N + 1. The dimension of parameters in MFM model.

        # Dataset parameters
        dataset_parameters = config['Dataset Parameters']
        self.simulate_time = float(dataset_parameters['simulate_time'])
        self.burn_in_time = float(dataset_parameters['burn_in_time'])
        self.TR = float(dataset_parameters['TR'])
        self.window_size = int(dataset_parameters['window_size'])

        # Simulating parameters
        simulating_parameters = config['Simulating Parameters']
        self.param_sets = int(
            simulating_parameters['param_sets']
        )  # Every epoch the number of sampled parameter sets
        self.select_param_sets = int(
            simulating_parameters['select_param_sets']
        )  # Number of selecting parameters to update CMA-ES from ${param_sets} parameter sets.
        self.min_select_param_sets = int(
            simulating_parameters['min_select_param_sets']
        )  # If cannot get ${select_param_sets} parameters valid, the minimum number of valid parameters. Or CMA-ES will break.
        self.param_dim = 10  # The dimension of the parameters in our parameterization
        self.param_dup = int(
            simulating_parameters['param_dup']
        )  # duplicate for 3 times for each parameter in Euler integration
        self.fcd_hist_bins = int(
            simulating_parameters['fcd_hist_bins']
        )  # For reforming FCD matrix to probability distribution.
        self.dt = float(simulating_parameters['dt_train'])

        # Training and generating parameters
        self.epochs = int(epochs)
        self.dl_pfic_range = np.array(dl_pfic_range, dtype=int)  # 1D array
        self.euler_pfic_range = np.array(euler_pfic_range, dtype=int)
        self.pfic_range = np.concatenate(
            (self.dl_pfic_range, self.euler_pfic_range))
        self.dl_rfic_range = np.array(dl_rfic_range, dtype=int)
        self.euler_rfic_range = np.array(euler_rfic_range, dtype=int)
        self.rfic_range = np.concatenate(
            (self.dl_rfic_range, self.euler_rfic_range))

        # Regularization loss weights
        self.w_wee = 10
        self.w_wei = 10
        self.w_sigma = 10

        # Other
        self.query_wei_range = query_wei_range

        if len(self.dl_pfic_range) > 0:

            # if dl_model_key is None:
            #     raise Exception("Please specify the dl_model_key if you need to use deep learning model to accelerate training.")
            dl_model_path = config['Deep learning Model Path']
            classify_model_save_path, predict_model_save_path = dl_model_path[
                'classify_model_save_path'], dl_model_path[
                    'predict_model_save_path']

            print('Load classify model...')
            if self.N == 68:
                self.classify_model_pFIC = ClassifyNanModel_2()
            elif self.N == 100:
                self.classify_model_pFIC = ClassifyNanModel_Yan100()
            else:
                raise NotImplementedError
            self.classify_model_pFIC.load_state_dict(
                torch.load(classify_model_save_path,
                           map_location=torch.device(
                               self.device))['model_state_dict'])
            print('Successfully loaded.')

            print('Load predict model...')
            if self.N == 68:
                self.predict_model_pFIC = PredictLossModel_1()
            elif self.N == 100:
                self.predict_model_pFIC = PredictLossModel_1_Yan100(n_roi=100)
            else:
                raise NotImplementedError
            self.predict_model_pFIC.load_state_dict(
                torch.load(predict_model_save_path,
                           map_location=torch.device(
                               self.device))['model_state_dict'])
            print('Successfully loaded.')

        if len(self.dl_rfic_range) > 0:
            raise Exception("rFIC deep learning model has not been developed.")

        self.save_param_dir = save_param_dir
        if not os.path.exists(self.save_param_dir):
            os.makedirs(self.save_param_dir)
        print(
            f'Successfully init pMFM deep learning version CMA-ES forward trainer. The results will be saved in {self.save_param_dir}'
        )

    def get_parameters(self, param_10):
        """
        From the 10 parameterization parameters to get the 3N+1 parameters
        :param param_10: [10, param_sets]
        :return: parameters for 2014 Deco Model [3N+1, param_sets]
        """
        return parameterize_myelin_rsfc(self.myelin, self.RSFC_gradient,
                                        param_10)

    def get_wei_range(self):
        if self.query_wei_range == 'Uniform':
            wei_min = 0
            wei_max = 5
        elif self.query_wei_range == 'PNC':
            # Using mean FC to adjust range
            wei_max = 11 * torch.mean(self.fc_dl)
            # 11 from shaoshi, 10 for HCD aCompCor, 8 for HCA aCompCor, 40 for HCD GSR
            wei_max = min(max(wei_max, 3.2), 5)
            # TODO: new wEI range with range restriction
            wei_min = 2 * wei_max - 6  # Original
            # wei_min = max(wei_max - 2.2,
            #               2 * wei_max - 6)  # Restrict the wEI range to 2.2
            # TODO: new wEI range with range restriction
            wei_min = min(max(wei_min, 0), 3.5)
            print(f"wEI search range [{wei_min}, {wei_max}].")
        elif self.query_wei_range == 'PNC2':
            # For testing, enlarge the range by 1 for both
            wei_max = 11 * torch.mean(self.fc_dl) + 1
            wei_min = 2 * wei_max - 6
            wei_min = min(max(wei_min, 0), 3.5)
            wei_max = min(max(wei_max, 4), 6.5)
            print(f"wEI search range [{wei_min}, {wei_max}].")
        elif self.query_wei_range == 'mean_fc_profile':
            # Using mean FC profile to adjust range
            wei_min = 7 * torch.mean(self.fc_euler, dim=1)
            wei_max = 5
            wei_min = torch.clamp(wei_min, min=0, max=3.5)
            print("Max wEI_min: ", torch.max(wei_min))
        else:
            raise Exception("query w_EI range key should be valid.")

        return wei_min, wei_max

    def mfm_model_loss_dl(self, parameter, epoch):
        """
        Using deep learning model to predict the total loss of MFM model
        :param parameter: [3*N+1, M]
        :param param_10: 10 parameterization parameters, just for saving.
        :param epoch: current epoch, just for naming the save file.
        :return: loss [select_param_sets(10),]; index [select_param_sets,]
        """
        self.classify_model_pFIC.eval()
        self.predict_model_pFIC.eval()
        with torch.no_grad():
            parameter = parameter.T  # !!! [xxx, 3N+1] Pay attention
            sc_this = self.sc_dl.unsqueeze(0).expand(parameter.shape[0], -1)
            pred_nan = self.classify_model_pFIC(parameter, sc_this)
            # [IMPORTANT] the probability this parameter will be available.
            pred_nan = torch.squeeze(pred_nan)  # [xxx]
            valid_mask = pred_nan > 0.5
            valid_param_list = torch.arange(0, parameter.shape[0], 1)
            valid_param_list = valid_param_list[valid_mask]

            count_valid = len(valid_param_list)
            select_param_sets = self.select_param_sets
            if count_valid >= self.select_param_sets:
                pass
            else:
                print(
                    f'Valid parameter sets are not enough, only {count_valid} parameters!'
                )
                if count_valid >= self.min_select_param_sets:
                    print('Choose min parameter sets instead.')
                    select_param_sets = count_valid
                else:
                    return None, None

            valid_parameter = parameter[valid_param_list]
            sc_this = self.sc_dl.unsqueeze(0).expand(valid_parameter.shape[0],
                                                     -1)
            fc_this = self.fc_dl.unsqueeze(0).expand(valid_parameter.shape[0],
                                                     -1)
            fcd_this = self.fcd_dl.unsqueeze(0).expand(
                valid_parameter.shape[0], -1)
            pred_loss = self.predict_model_pFIC(valid_parameter, sc_this,
                                                fc_this, fcd_this)  # [xxx, 3]
            # pred_loss = self.predict_model(valid_parameter, sc_this)
            total_loss = torch.sum(pred_loss, dim=1)  # [xxx]
            loss_sorted, index_sorted_in_valid = torch.sort(total_loss,
                                                            descending=False)
            index_sorted = valid_param_list[index_sorted_in_valid]

            print("Start saving parameter, valid_param_list and loss...")
            save_dict = {
                'parameter': parameter.T,
                'valid_param_list': valid_param_list,
                'FC_FCD_loss': pred_loss
            }
            torch.save(
                save_dict,
                os.path.join(self.save_param_dir,
                             'param_save_epoch' + str(epoch) + '.pth'))
            print("Successfully saved params and losses.")

            return loss_sorted[:
                               select_param_sets], index_sorted[:
                                                                select_param_sets]

    def mfm_model_loss_euler(self, parameter, epoch, flag_pFIC_rFIC=0):
        """
        Calculate the total loss of MFM model
        :param parameter: [3*N+1, M]
        :param epoch: current epoch, just for naming the save file.
        :param flag_pFIC_rFIC: 0 stands for pFIC, 1 stands for rFIC
        :return: loss [select_param_sets(10),]; index [select_param_sets,]
        """
        parameter_repeat = parameter.repeat(
            1, self.param_dup)  # [3*N+1, param_sets * param_dup]
        mfm_model = MfmModel2014(self.config,
                                 parameter_repeat,
                                 self.sc_euler,
                                 dt=self.dt)
        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)
        # bold_signal: [N, M=param_sets * param_dup, t_for_bold]; valid_M_mask: [param_sets * param_dup]; r_E_ave: [N, param_sets * param_dup]
        # bold_signal, valid_M_mask = CBIG_mfm_single_simulation_no_input_noise(parameter_repeat, sc_mat, self.t_epochlong)

        bold_signal = bold_signal.view(
            self.N, self.param_dup, self.param_sets,
            -1).transpose(1, 2)  # [N, param_sets, param_dup, t_for_bold]
        valid_M_mask = valid_M_mask.view(
            self.param_dup, self.param_sets).T  # [param_sets, param_dup]

        valid_param_list = []  # record valid param index
        fc_sim = torch.zeros(self.param_sets, self.N, self.N)
        fcd_hist = torch.zeros(self.fcd_hist_bins, self.param_sets)

        # TODO: Save r_E
        r_E_ave = r_E_ave.view(self.N,
                               self.param_dup, self.param_sets).transpose(
                                   1, 2)  # [N, param_sets, param_dup]
        r_E_for_valid_params = torch.zeros(self.N, self.param_sets)
        # TODO: Save r_E
        count_valid = 0
        for i in range(self.param_sets):
            # for each set of parameter
            mask_this_param = valid_M_mask[i]  # [param_dup]
            if mask_this_param.any():
                valid_param_list.append(i)
                bold_this_param = bold_signal[:, i,
                                              mask_this_param, :]  # [N, 1/2/3/param_dup, t_for_bold]
                fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
                fc_this_param = torch.mean(fc_this_param, dim=0)
                _, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                    bold_this_param, self.window_size)
                fcd_hist_this_param = torch.mean(fcd_hist_this_param, dim=1)

                fc_sim[count_valid] = fc_this_param
                fcd_hist[:, count_valid] = fcd_hist_this_param
                # TODO: Save r_E
                r_E_ave_this_param = r_E_ave[:, i,
                                             mask_this_param]  # [N, 1/2/3/param_dup]
                r_E_for_valid_params[:, count_valid] = torch.mean(
                    r_E_ave_this_param, dim=1)
                # TODO: Save r_E
                count_valid += 1

        select_param_sets = self.select_param_sets
        if count_valid >= self.select_param_sets:
            pass
        else:
            print(
                f'Valid parameter sets are not enough, only {count_valid} parameters!'
            )
            if count_valid >= self.min_select_param_sets:
                print('Choose min parameter sets instead.')
                select_param_sets = count_valid
            else:
                return None, None

        fc_sim = fc_sim[:count_valid]
        fcd_hist = fcd_hist[:, :count_valid]
        # TODO: Save r_E
        r_E_for_valid_params = r_E_for_valid_params[:, :count_valid]
        # TODO: Save r_E
        total_loss, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
            fc_sim, fcd_hist, self.fc_euler,
            self.fcd_cum_euler)  # [count_valid]
        FC_FCD_loss = torch.hstack(
            (corr_loss.unsqueeze(1), L1_loss.unsqueeze(1),
             ks_loss.unsqueeze(1)))
        if flag_pFIC_rFIC == 1:
            reg_loss = regularization_loss(
                parameter[:, valid_param_list],
                w_wee=self.w_wee,
                w_wei=self.w_wei,
                w_sigma=self.w_sigma,
                concat_mat=self.concat_mat,
                pinv_concat_mat=self.pinv_concat_mat)
            # [count_valid, 3]
            total_loss = total_loss + torch.sum(reg_loss, dim=1)

        # TODO: Regularize firing rate
        # if mfm_model.r_E:
        #     r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
        #                                     mfm_model.r_E,
        #                                     loss_type='L2')
        #     # TODO: Try without any constraint on r_E
        #     total_loss = total_loss + r_E_reg_loss
        #     # TODO: Try without any constraint on r_E

        # TODO: Regularize firing rate

        loss_sorted, index_sorted_in_valid = torch.sort(total_loss,
                                                        descending=False)
        valid_param_list = torch.as_tensor(valid_param_list)
        index_sorted = valid_param_list[index_sorted_in_valid]

        print("Start saving parameter, valid_param_list and loss...")
        save_dict = {
            'parameter': parameter,
            'valid_param_list': valid_param_list,
            'FC_FCD_loss': FC_FCD_loss
        }
        if flag_pFIC_rFIC == 1:
            save_dict['reg_loss'] = reg_loss

        # TODO: Save r_E
        if mfm_model.r_E:
            # save r_E_for_valid_params
            save_dict['r_E_for_valid_params'] = r_E_for_valid_params
            # TODO: Regularize firing rate
            # save_dict['r_E_reg_loss'] = r_E_reg_loss
            # TODO: Regularize firing rate
        # TODO: Save r_E

        torch.save(
            save_dict,
            os.path.join(self.save_param_dir,
                         'param_save_epoch' + str(epoch) + '.pth'))
        print("Successfully saved params and losses.")

        return loss_sorted[:select_param_sets], index_sorted[:
                                                             select_param_sets]

    def train_hybrid_pFIC_rFIC(self, seed=None, next_epoch=0):
        if next_epoch >= self.epochs:
            raise Exception('You do not need any more epoch.')

        N = self.N
        # Define search range. Maybe later it will depend on FC, so I do not put it in __init__
        search_range = torch.zeros(self.parameter_dim, 2)
        search_range[0:N, 0] = 1  # wee_min
        search_range[0:N, 1] = 10  # wee_max
        wei_min, wei_max = self.get_wei_range()
        search_range[N:2 * N, 0] = wei_min
        search_range[N:2 * N, 1] = wei_max
        search_range[2 * N, 0] = 0  # search range for G
        search_range[2 * N, 1] = 3
        search_range[2 * N + 1:, 0] = 0.0005  # search range for sigma
        search_range[2 * N + 1:, 1] = 0.01
        self.search_range = search_range

        if next_epoch == 0:
            if seed is None:
                seed = np.random.randint(0, 1000000000000)
            rand_ge = torch.manual_seed(seed)

            # initialization, k = 0
            m_k, sigma_k, cov_k, p_sigma_k, p_c_k = self._init_CMA_ES_pFIC(
                search_range)
            print("Training parameters have been initialized...")

        elif next_epoch in self.euler_pfic_range:
            previous_final_state_path = os.path.join(self.save_param_dir,
                                                     'final_state_pFIC.pth')
            if not os.path.exists(previous_final_state_path):
                raise Exception("Previous final state path doesn't exist.")
            final_dict = torch.load(previous_final_state_path,
                                    map_location=torch.device(self.device))
            seed = final_dict['seed']
            random_state = final_dict['random_state']
            m_k = final_dict['m']
            sigma_k = final_dict['sigma']
            cov_k = final_dict['cov']
            p_sigma_k = final_dict['p_sigma']
            p_c_k = final_dict['p_c']

            rand_ge = torch.manual_seed(seed)
            rand_ge.set_state(random_state)
            print(
                "Successfully loaded previous parameters. Will start next epochs."
            )

        elif next_epoch == self.euler_rfic_range[0]:
            previous_final_state_path = os.path.join(self.save_param_dir,
                                                     'final_state_pFIC.pth')
            if not os.path.exists(previous_final_state_path):
                previous_final_state_path = os.path.join(
                    self.save_param_dir, 'final_state.pth')
                if not os.path.exists(previous_final_state_path):
                    raise Exception("Previous final state path doesn't exist.")
            print(f"Load final dict from {previous_final_state_path}...")
            final_dict = torch.load(previous_final_state_path,
                                    map_location=torch.device(self.device))
            seed = final_dict['seed']
            random_state = final_dict['random_state']
            rand_ge = torch.manual_seed(seed)
            rand_ge.set_state(random_state)
            # seed = np.random.randint(0, 1000000000000)
            # rand_ge = torch.manual_seed(seed)
            print(
                "Successfully loaded previous parameters. Will start next epochs."
            )

        else:
            raise Exception("Check next epoch.")

        print(' -- Start training by CMA-ES --')
        for k in range(next_epoch, self.epochs):
            print("Epoch: [{}/{}]".format(k + 1, self.epochs))

            if k in self.pfic_range:
                epoch_res = self._train_one_epoch_pFIC(k, m_k, sigma_k, cov_k,
                                                       p_sigma_k, p_c_k)
                if epoch_res is None:
                    return 1 if k in self.dl_pfic_range or k == self.euler_pfic_range[
                        0] else 2
                m_k, sigma_k, cov_k, p_sigma_k, p_c_k = epoch_res

                if k == self.pfic_range[-1]:
                    print(
                        "The last epoch of pFIC. Start saving random seed, random state and final CMA-ES coefs..."
                    )
                    final_dict = {
                        'seed': seed,
                        'random_state': rand_ge.get_state(),
                        'm': m_k,
                        'sigma': sigma_k,
                        'cov': cov_k,
                        'p_sigma': p_sigma_k,
                        'p_c': p_c_k,
                        'epoch': k,
                        'wEI_search_range': search_range[N:2 * N]
                    }
                    torch.save(
                        final_dict,
                        os.path.join(self.save_param_dir,
                                     'final_state_pFIC.pth'))
                    print("Successfully saved pFIC final dict.")

            elif k in self.rfic_range:
                if k == self.rfic_range[0]:
                    previous_epoch_path = os.path.join(
                        self.save_param_dir, f'param_save_epoch{k - 1}.pth')
                    if not os.path.exists(previous_epoch_path):
                        raise Exception(
                            "The previous 1 epoch parameter path doesn't exist."
                        )
                    init_rfic = torch.load(previous_epoch_path,
                                           map_location=torch.device(
                                               self.device))
                    pre_best_parameter = select_best_parameter_from_savedict(
                        init_rfic)
                    m_k, sigma_k, cov_k, p_sigma_k, p_c_k = self._init_CMA_ES_rFIC(
                        pre_best_parameter)

                epoch_res = self._train_one_epoch_rFIC(k, m_k, sigma_k, cov_k,
                                                       p_sigma_k, p_c_k)
                if epoch_res is None:
                    return 1 if k in self.dl_rfic_range or k == next_epoch else 2
                m_k, sigma_k, cov_k, p_sigma_k, p_c_k = epoch_res

        print(" -- Done training by CMA-ES --")

        print(
            "Start saving random seed, random state and final CMA-ES coefs...")
        final_dict = {
            'seed': seed,
            'random_state': rand_ge.get_state(),
            'm': m_k,
            'sigma': sigma_k,
            'cov': cov_k,
            'p_sigma': p_sigma_k,
            'p_c': p_c_k,
            'epoch': k,
            'wEI_search_range': search_range[N:2 * N]
        }
        torch.save(final_dict,
                   os.path.join(self.save_param_dir, 'final_state.pth'))
        print("Successfully saved final dict.")

        return 0

    def _train_one_epoch_pFIC(self, k, m_k, sigma_k, cov_k, p_sigma_k, p_c_k):

        # TODO: Try uniform sampling at the first iteration
        if k == 0:  # The first epoch
            param_10_k, parameter_k = self._sample_uniform_parameters()
        else:
            param_10_k, parameter_k = self._sample_valid_parameters_pFIC(
                m_k, sigma_k**2 * cov_k, self.search_range)

        # Original sampling
        # param_10_k, parameter_k = self._sample_valid_parameters_pFIC(
        #     m_k, sigma_k**2 * cov_k, self.search_range)
        # TODO: Try uniform sampling at the first iteration
        if param_10_k is None or parameter_k is None:
            print("Sampling failed!")
            return None

        if k in self.dl_pfic_range:
            loss_k, index_k = self.mfm_model_loss_dl(parameter_k, k)
        elif k in self.euler_pfic_range:
            loss_k, index_k = self.mfm_model_loss_euler(parameter_k, k)
        else:
            raise Exception("Not pFIC epoch. Break.")

        if loss_k is None:
            print("Iteration ends.")
            return None
        select_params = param_10_k[:, index_k]

        # TODO: Try uniform sampling at the first iteration
        if k == 0:  # The first epoch
            m_k = torch.mean(select_params, dim=1)
        # TODO: Try uniform sampling at the first iteration

        m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1 = self._update_CMA_ES_pFIC(
            select_params, loss_k, m_k, sigma_k, cov_k, p_sigma_k, p_c_k, k)
        return m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1

    def _train_one_epoch_rFIC(self, k, m_k, sigma_k, cov_k, p_sigma_k, p_c_k):
        parameter_k = self._sample_valid_parameters_rFIC(
            m_k, sigma_k**2 * cov_k, self.search_range)
        if parameter_k is None:
            print("Sampling failed!")
            return None

        loss_k, index_k = self.mfm_model_loss_euler(parameter_k,
                                                    k,
                                                    flag_pFIC_rFIC=1)

        if loss_k is None:
            print("Iteration ends.")
            return None
        select_params = parameter_k[:, index_k]

        m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1 = self._update_CMA_ES_rFIC(
            select_params, loss_k, m_k, sigma_k, cov_k, p_sigma_k, p_c_k, k)
        return m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1

    def _init_CMA_ES_pFIC(self, search_range):
        """
        Initialize from parameters' search range
        :return: m_0, cov_0, ... Their shape can be seen in self._update_CMA_ES
        """
        N = self.N

        # From 3N+1 parameters and concat_mat's pseudo inverse
        init_parameters = torch.rand(self.parameter_dim) * (
            search_range[:, 1] -
            search_range[:, 0]) + search_range[:, 0]  # [3*N+1]
        init_parameters = init_parameters.unsqueeze(1)  # [3*N+1, 1]
        start_point_wEE = torch.matmul(self.pinv_concat_mat,
                                       init_parameters[0:N]).squeeze()
        start_point_wEI = torch.matmul(self.pinv_concat_mat,
                                       init_parameters[N:2 * N]).squeeze()
        start_point_sigma = torch.matmul(  # noqa
            self.pinv_concat_mat, init_parameters[2 * N + 1:]).squeeze()

        # Init m_0 for CMA-ES, just by the experience of my seniors
        m_0 = torch.zeros(self.param_dim)
        m_0[0:3] = start_point_wEE
        # m_0[1] = start_point_wEE[1] / 2
        m_0[3:6] = start_point_wEI
        # m_0[4] = start_point_wEI[1] / 2
        m_0[6] = init_parameters[2 * N]  # G
        m_0[7:] = start_point_sigma

        sigma_0 = 0.2
        p_sigma_0 = torch.zeros(self.param_dim, 1)
        p_c_0 = torch.zeros(self.param_dim, 1)
        V_ini = torch.eye(self.param_dim)
        Lambda_ini = torch.ones(self.param_dim)
        # Lambda_ini[0:3] = start_point_wEE[0] / 2
        # Lambda_ini[3:6] = start_point_wEI[0] / 2
        Lambda_ini[0:3] = start_point_wEE[0]
        Lambda_ini[3:6] = start_point_wEI[0]
        Lambda_ini[6] = 0.4
        Lambda_ini[7:] = 0.0005
        cov_0 = torch.matmul(V_ini,
                             torch.matmul(torch.diag(Lambda_ini**2), V_ini.T))

        return m_0, sigma_0, cov_0, p_sigma_0, p_c_0

    def _init_CMA_ES_rFIC(self, init_parameter):
        """
        Initialize from 3*N+1 parameters' search range
        :return: m_0, cov_0, ... Their shape can be seen in self._update_CMA_ES
        """
        N = self.N

        m_0 = init_parameter
        sigma_0 = 0.2
        p_sigma_0 = torch.zeros(self.parameter_dim, 1)
        p_c_0 = torch.zeros(self.parameter_dim, 1)
        V_ini = torch.eye(self.parameter_dim)
        Lambda_ini = torch.ones(self.parameter_dim)

        Lambda_ini[0:N] = init_parameter[0:N] / 10
        Lambda_ini[N:2 * N] = init_parameter[N:2 * N] / 10
        Lambda_ini[2 * N] = 0.4
        Lambda_ini[2 * N + 1:] = 0.0005
        cov_0 = torch.matmul(V_ini,
                             torch.matmul(torch.diag(Lambda_ini**2), V_ini.T))

        return m_0, sigma_0, cov_0, p_sigma_0, p_c_0

    def _update_CMA_ES_pFIC(self, select_params, loss_k, m_k, sigma_k, cov_k,
                            p_sigma_k, p_c_k, k):
        """
        Update the parameters in CMA-ES algorithm (k -> k+1 (kp1)). Refer to wikipedia: https://en.wikipedia.org/wiki/CMA-ES
        :param select_params: parameters [param_dim, select_param_sets]
        :param loss_k: sorted loss by mfm_model_loss. [select_param_sets,]
        :param m_k: the means of multivariate gaussian, need updated. [param_dim,]
        :param sigma_k: the step sizes of CMA-ES, need updated. Double
        :param cov_k: the covariance matrix, need updated. [param_dim, param_dim]
        :param p_sigma_k: evolution path p_sigma
        :param p_c_k: evolution path p_c
        :param k: current iter
        :return: parameters in k+1 iter.
        """
        select_param_sets = select_params.shape[1]
        loss_inverse = 1 / loss_k
        weights = loss_inverse / torch.sum(
            loss_inverse)  # [select_param_sets, 1]
        # select_params = x_k[:, index_k]  # [param_dim, select_param_sets]
        m_kp1 = torch.matmul(select_params, weights)  # m_(k+1): [param_dim, 1]
        mueff = 1 / torch.sum(weights**2)  # mu_w

        # The evolution path p_sigma and p_c
        Lambda, V = torch.linalg.eigh(cov_k)  # eigen decomposition
        Lambda = torch.sqrt(Lambda)
        inv_sqrt_cov = torch.matmul(V, torch.matmul(torch.diag(Lambda**-1),
                                                    V.T))  # C^(-1/2)

        c_sigma = (mueff + 2) / (self.param_dim + mueff + 5)
        c_c = (4 + mueff / self.param_dim) / (self.param_dim + 4 +
                                              2 * mueff / self.param_dim)
        c_1 = 2 / ((self.param_dim + 1.3)**2 + mueff)
        c_mu = min(
            1 - c_1,
            2 * (mueff - 2 + 1 / mueff) / ((self.param_dim + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(
            0,
            torch.sqrt((mueff - 1) / (self.param_dim + 1)) - 1) + c_sigma
        expected_value = self.param_dim**0.5 * (1 - 1 /
                                                (4 * self.param_dim) + 1 /
                                                (21 * self.param_dim**2))

        p_sigma_kp1 = (1 - c_sigma) * p_sigma_k + torch.sqrt(
            c_sigma * (2 - c_sigma) * mueff) * torch.matmul(
                inv_sqrt_cov, (m_kp1 - m_k).unsqueeze(1) / sigma_k)
        indicator = (torch.linalg.norm(p_sigma_kp1) /
                     torch.sqrt(1 - (1 - c_sigma)**(2 * k)) / expected_value
                     < (1.4 + 2 / (self.param_dim + 1))) * 1
        p_c_kp1 = (1 - c_c) * p_c_k + indicator * torch.sqrt(
            c_c * (2 - c_c) * mueff) * (m_kp1 - m_k).unsqueeze(1) / sigma_k

        # Adapting covariance matrix C
        artmp = (1 / sigma_k) * (select_params -
                                 torch.tile(m_k, [select_param_sets, 1]).T)
        cov_kp1 = (1 - c_1 - c_mu) * cov_k + c_1 * (torch.matmul(p_c_kp1, p_c_kp1.T) + (1 - indicator) * c_c * (2 - c_c) * cov_k) + \
            c_mu * torch.matmul(artmp, torch.matmul(torch.diag(weights), artmp.T))

        # Adapting step size
        sigma_kp1 = sigma_k * torch.exp(
            (c_sigma / d_sigma) *
            (torch.linalg.norm(p_sigma_kp1) / expected_value - 1))
        return m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1

    def _update_CMA_ES_rFIC(self, select_parameters, loss_k, m_k, sigma_k,
                            cov_k, p_sigma_k, p_c_k, k):
        """
        Update the parameters in CMA-ES algorithm (k -> k+1 (kp1)). Refer to wikipedia: https://en.wikipedia.org/wiki/CMA-ES
        :param select_params: parameters [parameter_dim, select_param_sets]
        :param loss_k: sorted loss by mfm_model_loss. [select_param_sets,]
        :param m_k: the means of multivariate gaussian, need updated. [parameter_dim,]
        :param sigma_k: the step sizes of CMA-ES, need updated. Double
        :param cov_k: the covariance matrix, need updated. [parameter_dim, parameter_dim]
        :param p_sigma_k: evolution path p_sigma
        :param p_c_k: evolution path p_c
        :param k: current iter
        :return: parameters in k+1 iter.
        """
        select_param_sets = select_parameters.shape[1]
        loss_inverse = 1 / loss_k
        weights = loss_inverse / torch.sum(
            loss_inverse)  # [select_param_sets, 1]
        # select_params = x_k[:, index_k]  # [param_dim, select_param_sets]
        m_kp1 = torch.matmul(select_parameters,
                             weights)  # m_(k+1): [param_dim, 1]
        mueff = 1 / torch.sum(weights**2)  # mu_w

        # The evolution path p_sigma and p_c
        Lambda, V = torch.linalg.eigh(cov_k)  # eigen decomposition
        Lambda = torch.sqrt(Lambda)
        inv_sqrt_cov = torch.matmul(V, torch.matmul(torch.diag(Lambda**-1),
                                                    V.T))  # C^(-1/2)

        c_sigma = (mueff + 2) / (self.parameter_dim + mueff + 5)
        c_c = (4 + mueff / self.parameter_dim) / (
            self.parameter_dim + 4 + 2 * mueff / self.parameter_dim)
        c_1 = 2 / ((self.parameter_dim + 1.3)**2 + mueff)
        c_mu = min(
            1 - c_1, 2 * (mueff - 2 + 1 / mueff) /
            ((self.parameter_dim + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(
            0,
            torch.sqrt((mueff - 1) / (self.parameter_dim + 1)) - 1) + c_sigma
        expected_value = self.parameter_dim**0.5 * (
            1 - 1 / (4 * self.parameter_dim) + 1 /
            (21 * self.parameter_dim**2))

        p_sigma_kp1 = (1 - c_sigma) * p_sigma_k + torch.sqrt(
            c_sigma * (2 - c_sigma) * mueff) * torch.matmul(
                inv_sqrt_cov, (m_kp1 - m_k).unsqueeze(1) / sigma_k)
        indicator = (torch.linalg.norm(p_sigma_kp1) /
                     torch.sqrt(1 - (1 - c_sigma)**(2 * k)) / expected_value
                     < (1.4 + 2 / (self.parameter_dim + 1))) * 1
        p_c_kp1 = (1 - c_c) * p_c_k + indicator * torch.sqrt(
            c_c * (2 - c_c) * mueff) * (m_kp1 - m_k).unsqueeze(1) / sigma_k

        # Adapting covariance matrix C
        artmp = (1 / sigma_k) * (select_parameters -
                                 torch.tile(m_k, [select_param_sets, 1]).T)
        cov_kp1 = (1 - c_1 - c_mu) * cov_k + c_1 * (torch.matmul(p_c_kp1, p_c_kp1.T) + (1 - indicator) * c_c * (2 - c_c) * cov_k) + \
            c_mu * torch.matmul(artmp, torch.matmul(torch.diag(weights), artmp.T))

        # Adapting step size
        sigma_kp1 = sigma_k * torch.exp(
            (c_sigma / d_sigma) *
            (torch.linalg.norm(p_sigma_kp1) / expected_value - 1))
        return m_kp1, sigma_kp1, cov_kp1, p_sigma_kp1, p_c_kp1

    # TODO: Try uniform sampling at the first iteration
    def _sample_uniform_parameters(self):
        """
        Sample parameters from uniform distribution
        :return:
            sampled_params: [10, param_sets]
            sampled_parameters: [3*N+1, param_sets]
        """
        N = self.N
        search_range = self.search_range

        sampled_params = torch.zeros(self.param_dim, self.param_sets)
        sampled_parameters = torch.zeros(
            self.parameter_dim, self.param_sets)  # [3*N+1, param_sets]
        for i in range(self.param_sets):
            sampled_parameters[:, i] = torch.rand(self.parameter_dim) * (
                search_range[:, 1] -
                search_range[:, 0]) + search_range[:, 0]  # [3*N+1]
            sampled_params[0:3, i] = torch.matmul(self.pinv_concat_mat,
                                                  sampled_parameters[0:N, i])
            sampled_params[3:6,
                           i] = torch.matmul(self.pinv_concat_mat,
                                             sampled_parameters[N:2 * N, i])
            sampled_params[6, i] = sampled_parameters[2 * N, i]
            sampled_params[7:,
                           i] = torch.matmul(self.pinv_concat_mat,
                                             sampled_parameters[2 * N + 1:, i])
        return sampled_params, sampled_parameters

    # TODO: Try uniform sampling at the first iteration

    def _sample_valid_parameters_pFIC(self, mean, cov, search_range):
        # TODO: Try without parameterizing sigma
        # mean[7] = 0.005
        # mean[8] = 0
        # mean[9] = 0
        # TODO: Try without parameterizing sigma
        multivariate_normal = td.MultivariateNormal(mean, cov)
        valid_count = 0
        total_count = 0
        total_threshold = 20000 * self.param_sets
        sampled_params = torch.zeros(self.param_dim, self.param_sets)
        sampled_parameters = torch.zeros(
            self.parameter_dim, self.param_sets)  # [3*N+1, param_sets]
        while valid_count < self.param_sets:
            sampled_params[:, valid_count] = multivariate_normal.sample(
            )  # [10, param_sets]
            # TODO: Try without parameterizing sigma
            # sampled_params[7, valid_count] = 0.005
            # sampled_params[8, valid_count] = 0
            # sampled_params[9, valid_count] = 0
            # TODO: Try without parameterizing sigma
            sampled_parameters[:, valid_count] = self.get_parameters(
                sampled_params[:, valid_count]).squeeze()
            wEI = sampled_parameters[self.N:2 * self.N,
                                     valid_count].unsqueeze(1)
            wEI_myelin_corr = torch.squeeze(CBIG_corr(wEI, self.myelin))
            wEI_rsfc_corr = torch.squeeze(CBIG_corr(wEI, self.RSFC_gradient))
            if (sampled_parameters[:, valid_count] < search_range[:, 0]).any() \
                    or (sampled_parameters[:, valid_count] > search_range[:, 1]).any() or (wEI_myelin_corr > 0) or (wEI_rsfc_corr < 0):
                valid_count -= 1
            valid_count += 1
            total_count += 1
            if total_count >= total_threshold:
                print(
                    f"Not enough valid sampled parameters! Only sample {valid_count} parameters!"
                )
                return None, None
        return sampled_params, sampled_parameters

    def _sample_valid_parameters_rFIC(self, mean, cov, search_range):
        multivariate_normal = td.MultivariateNormal(mean, cov)
        valid_count = 0
        total_count = 0
        total_threshold = 20000 * self.param_sets
        sampled_parameters = torch.zeros(
            self.parameter_dim, self.param_sets)  # [3*N+1, param_sets]
        while valid_count < self.param_sets:
            sampled_parameters[:, valid_count] = multivariate_normal.sample()
            wEI = sampled_parameters[self.N:2 * self.N,
                                     valid_count].unsqueeze(1)
            wEI_myelin_corr = torch.squeeze(CBIG_corr(wEI, self.myelin))
            wEI_rsfc_corr = torch.squeeze(CBIG_corr(wEI, self.RSFC_gradient))
            if (sampled_parameters[:, valid_count] < search_range[:, 0]).any() \
                    or (sampled_parameters[:, valid_count] > search_range[:, 1]).any() or (wEI_myelin_corr > 0) or (wEI_rsfc_corr < 0):
                valid_count -= 1
            valid_count += 1
            total_count += 1
            if total_count >= total_threshold:
                print(
                    f"Not enough valid sampled parameters! Only sample {valid_count} parameters!"
                )
                return None
        return sampled_parameters


class DLVersionCMAESValidator:
    """The validator for Hybrid version CMA-ES

    Functions:
    val_by_euler: use Euler integration to validate those epochs trained by Deep learning model
    val_best_parameters: Choose those best parameters in train set and apply them to validation set
    """

    def __init__(self, config, save_param_dir, val_save_dir):

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            self.device = 'cuda'
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.device = 'cpu'

        self.config = config

        # Dataset parameters
        dataset_parameters = config['Dataset Parameters']
        self.simulate_time = float(dataset_parameters['simulate_time'])
        self.burn_in_time = float(dataset_parameters['burn_in_time'])
        self.TR = float(dataset_parameters['TR'])
        self.window_size = int(dataset_parameters['window_size'])

        # Simulating parameters
        simulating_parameters = config['Simulating Parameters']
        self.N = int(simulating_parameters['n_ROI'])
        self.parameters_dim = 3 * self.N + 1  # The dimension of parameters in MFM model.
        self.param_sets = int(simulating_parameters['param_sets'])
        self.param_dup = int(simulating_parameters['param_dup'])
        self.fcd_hist_bins = int(simulating_parameters['fcd_hist_bins'])
        self.dt = float(simulating_parameters['dt_val'])

        self.save_param_dir = save_param_dir
        self.val_save_dir = val_save_dir

        if not os.path.exists(self.val_save_dir):
            os.makedirs(self.val_save_dir)
        print(
            f"DL Version CMA-ES validator has been successfully initialized. Results will be saved in {self.val_save_dir}"
        )

    @DeprecationWarning  # Need modify
    def val_by_euler(self, myelin, rsfc_gradient, sc_euler, fc_emp,
                     emp_fcd_cum, epoch):
        print(" -- Start validating -- ")

        self.dt = 0.006
        # Set random seed
        seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)

        param_10_path = os.path.join(self.save_param_dir,
                                     f'param_save_epoch{epoch}.pth')
        if not os.path.exists(param_10_path):
            raise Exception("Path doesn't exist.")
        d = torch.load(param_10_path)
        param_10 = d['param_10']
        valid_param_list_dl = d['valid_param_list']
        parameter = parameterize_myelin_rsfc(myelin, rsfc_gradient, param_10)
        parameter_repeat = parameter.repeat(
            1, self.param_dup)  # [3*N+1, param_sets * param_dup]
        mfm_model = MfmModel2014(parameter_repeat, sc_euler, dt=self.dt)
        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)

        bold_signal = bold_signal.view(
            self.N, self.param_dup, self.param_sets,
            -1).transpose(1, 2)  # [N, param_sets, param_dup, t_for_bold]
        valid_M_mask = valid_M_mask.view(
            self.param_dup, self.param_sets).T  # [param_sets, param_dup]

        valid_param_list = []  # record valid param index
        fc_sim = torch.zeros(self.param_sets, self.N, self.N)
        fcd_hist = torch.zeros(self.fcd_hist_bins, self.param_sets)
        count_valid = 0
        for i in range(self.param_sets):
            # for each set of parameter
            mask_this_param = valid_M_mask[i]  # [param_dup]
            if mask_this_param.any():
                valid_param_list.append(i)
                bold_this_param = bold_signal[:, i,
                                              mask_this_param, :]  # [N, 1/2/3/param_dup, t_for_bold]
                fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
                fc_this_param = torch.mean(fc_this_param, dim=0)
                _, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                    bold_this_param, self.window_size)
                fcd_hist_this_param = torch.mean(fcd_hist_this_param, dim=1)

                fc_sim[count_valid] = fc_this_param
                fcd_hist[:, count_valid] = fcd_hist_this_param
                count_valid += 1
        fc_sim = fc_sim[:count_valid]
        fcd_hist = fcd_hist[:, :count_valid]
        total_loss, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
            fc_sim, fcd_hist, fc_emp, emp_fcd_cum)  # [count_valid]
        valid_param_list = torch.as_tensor(valid_param_list)

        print('Start saving results...')
        save_dict = {
            'valid_param_list_dl': valid_param_list_dl,
            'valid_param_list': valid_param_list,
            'corr_loss': corr_loss,
            'l1_loss': L1_loss,
            'ks_loss': ks_loss,
            'seed': seed
        }
        torch.save(
            save_dict,
            os.path.join(self.val_save_dir, f'Euler_epoch{epoch}' + '.pth'))
        print("Successfully saved valid param lists and losses.")

        print(" -- Done validating -- ")
        return 0

    def val_best_parameters(self,
                            sc_euler,
                            fc_emp,
                            emp_fcd_cum,
                            epoch,
                            seed=None):
        print(" -- Start validating -- ")
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)

        parameter_path = os.path.join(self.save_param_dir,
                                      f'param_save_epoch{epoch}.pth')
        if not os.path.exists(parameter_path):
            raise Exception("Path doesn't exist.")
        d = torch.load(parameter_path, map_location=torch.device(self.device))

        valid_param_list_pre = d['valid_param_list']
        parameter = d['parameter']
        parameter = parameter[:, valid_param_list_pre]
        if 'FC_FCD_loss' in d:
            # TODO: Try without FCD KS loss
            # FC_FCD_loss is [param_sets, 3]
            total_loss = torch.sum(d['FC_FCD_loss'], dim=1)
            # total_loss = torch.sum(d['FC_FCD_loss'][:, :2], dim=1)
            # TODO: Try without FCD KS loss

            # TODO: Regularize firing rate
            # TODO: Try without any constraint on r_E
            # total_loss += d['r_E_reg_loss']  # r_E_reg_loss is [param_sets]
            # TODO: Try without any constraint on r_E
            # TODO: Regularize firing rate
        else:
            raise Exception("Check the dictionary keys.")
        best_param_ind = torch.argmin(total_loss)
        parameter = parameter[:, best_param_ind].unsqueeze(1)

        parameter_repeat = parameter.repeat(
            1, self.param_dup)  # [3*N+1, param_sets * param_dup]
        mfm_model = MfmModel2014(self.config,
                                 parameter_repeat,
                                 sc_euler,
                                 dt=self.dt)
        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)
        # bold_signal: [ROIs, param_dup, t]; valid_M_mask: [param_dup]
        fc_this_param = None
        if valid_M_mask.any():
            bold_this_param = bold_signal[:,
                                          valid_M_mask, :]  # [N, 1/2/3/param_dup, t_for_bold]
            fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
            fc_this_param = torch.mean(fc_this_param, dim=0)
            fcd_mat_this_param, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                bold_this_param, self.window_size)
            # You should not average across different FCD matrix
            fcd_hist_this_param = torch.mean(fcd_hist_this_param,
                                             dim=1).unsqueeze(1)  # [10000, 1]

        if fc_this_param is None:
            print("This chosen parameter fails Euler.")
            return 1

        _, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
            fc_this_param.unsqueeze(0), fcd_hist_this_param, fc_emp,
            emp_fcd_cum)  # [1]
        print('Start saving results...')
        save_dict = {
            'parameter': parameter,
            'corr_loss': corr_loss,
            'l1_loss': L1_loss,
            'ks_loss': ks_loss,
            'seed': seed
        }
        # TODO: Save r_E
        if mfm_model.r_E and valid_M_mask.any():
            # save r_E_for_valid_params
            r_E_for_valid_params = r_E_ave[:, valid_M_mask]
            r_E_for_valid_params = torch.mean(r_E_for_valid_params,
                                              dim=1,
                                              keepdim=True)
            save_dict['r_E_for_valid_params'] = r_E_for_valid_params
            # TODO: Regularize firing rate
            # # get and save r_E_reg_loss
            # r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
            #                                 mfm_model.r_E,
            #                                 loss_type='L2')
            # save_dict['r_E_reg_loss'] = r_E_reg_loss
            # TODO: Regularize firing rate

        # TODO: Save r_E
        torch.save(
            save_dict,
            os.path.join(self.val_save_dir, f'best_param{epoch}' + '.pth'))
        print("Successfully saved.")

        print(" -- Done validating -- ")
        return 0


class DLVersionCMAESTester:

    def __init__(self, config, val_dirs, test_dir, trained_epochs):

        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
            self.device = 'cuda'
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')
            self.device = 'cpu'

        self.config = config

        # Dataset parameters
        dataset_parameters = config['Dataset Parameters']
        self.simulate_time = float(dataset_parameters['simulate_time'])
        self.burn_in_time = float(dataset_parameters['burn_in_time'])
        self.TR = float(dataset_parameters['TR'])
        self.window_size = int(dataset_parameters['window_size'])

        # Simulating parameters
        simulating_parameters = config['Simulating Parameters']
        self.N = int(simulating_parameters['n_ROI'])
        self.parameters_dim = 3 * self.N + 1  # The dimension of parameters in MFM model.
        self.param_sets = int(
            simulating_parameters['test_param_sets']
        )  # select number of best parameters from val_dirs
        # [NOTE] here param_sets is not identical to the definition in Validator and Trainer
        self.param_dup = int(simulating_parameters['param_dup'])
        self.fcd_hist_bins = int(simulating_parameters['fcd_hist_bins'])
        self.dt = float(simulating_parameters['dt_test'])

        self.trained_epochs = trained_epochs  # The number of param files in one validation directory

        self.val_dirs = val_dirs
        self.test_dir = test_dir
        self.val_dirs_len = len(self.val_dirs)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        print("Validation dirs: ", val_dirs)
        print(
            f"DL Version CMA-ES tester has been successfully initialized. Results will be stored in {self.test_dir}"
        )

    # TODO: Modify this as well if we want to regularize firing rate for HCPYA
    def test(self, sc_euler, fc_emp, emp_fcd_cum, seed=None):
        print(" -- Start testing -- ")

        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)

        parameter_sets = torch.zeros(self.parameters_dim, self.val_dirs_len *
                                     self.trained_epochs)  # [10, 500]
        val_loss_sets = torch.ones(self.val_dirs_len * self.trained_epochs) * 3
        valid_val_dir_count = 0
        valid_val_epoch_count = 0
        for val_dir_i in range(self.val_dirs_len):
            val_dir = self.val_dirs[val_dir_i]
            if not os.path.exists(val_dir):
                print(f"{val_dir} doesn't exist.")
                continue
            valid_val_dir_count += 1
            for epoch in range(self.trained_epochs):
                param_val_path = os.path.join(val_dir,
                                              f'best_param{epoch}.pth')
                if not os.path.exists(param_val_path):
                    print(f"{param_val_path} doesn't exist.")
                    continue
                valid_val_epoch_count += 1
                d = torch.load(param_val_path,
                               map_location=torch.device(self.device))
                parameter_sets[:, val_dir_i * self.trained_epochs +
                               epoch] = torch.squeeze(d['parameter'])
                val_loss_sets[
                    val_dir_i * self.trained_epochs +
                    epoch] = d['corr_loss'] + d['l1_loss'] + d['ks_loss']
        if valid_val_dir_count == 0:
            print("No valid validated directories.")
            return 1
        if valid_val_epoch_count == 0:
            print("No valid epoch.")
            return 1
        # Record all param_10 and loss
        val_losses, sorted_index = torch.sort(val_loss_sets, descending=False)
        val_losses = val_losses[:self.param_sets]
        sorted_index = sorted_index[:self.param_sets]
        parameter = parameter_sets[:, sorted_index]

        parameter_repeat = parameter.repeat(
            1, self.param_dup)  # [3*N+1, param_sets * param_dup]
        mfm_model = MfmModel2014(self.config,
                                 parameter_repeat,
                                 sc_euler,
                                 dt=self.dt)
        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)

        bold_signal = bold_signal.view(
            self.N, self.param_dup, self.param_sets,
            -1).transpose(1, 2)  # [N, param_sets, param_dup, t_for_bold]
        valid_M_mask = valid_M_mask.view(
            self.param_dup, self.param_sets).T  # [param_sets, param_dup]

        valid_param_list = []  # record valid param index
        fc_sim = torch.zeros(self.param_sets, self.N, self.N)
        fcd_hist = torch.zeros(self.fcd_hist_bins, self.param_sets)
        count_valid = 0
        for i in range(self.param_sets):
            # for each set of parameter
            mask_this_param = valid_M_mask[i]  # [param_dup]
            if mask_this_param.any():
                valid_param_list.append(i)
                bold_this_param = bold_signal[:, i,
                                              mask_this_param, :]  # [N, 1/2/3/param_dup, t_for_bold]
                fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
                fc_this_param = torch.mean(fc_this_param, dim=0)
                _, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                    bold_this_param, self.window_size)
                fcd_hist_this_param = torch.mean(fcd_hist_this_param, dim=1)

                fc_sim[count_valid] = fc_this_param
                fcd_hist[:, count_valid] = fcd_hist_this_param
                count_valid += 1
        fc_sim = fc_sim[:count_valid]
        fcd_hist = fcd_hist[:, :count_valid]
        total_loss, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
            fc_sim, fcd_hist, fc_emp, emp_fcd_cum)  # [count_valid]
        valid_param_list = torch.as_tensor(valid_param_list)

        print('Start saving results...')
        save_dict = {
            'parameter': parameter,
            'val_losses': val_losses,
            'sorted_index': sorted_index,
            'valid_param_list': valid_param_list,
            'corr_loss': corr_loss,
            'l1_loss': L1_loss,
            'ks_loss': ks_loss,
            'seed': seed
        }
        torch.save(save_dict, os.path.join(self.test_dir, 'test_results.pth'))
        print("Successfully saved test results.")

        print(" -- Done testing -- ")
        return 0

    def select_best_from_val(self):
        print(" -- Start testing -- ")

        parameter_sets = torch.zeros(
            self.parameters_dim, self.val_dirs_len *
            self.trained_epochs)  # [205, 50 * num_of_tried_seeds]
        # TODO: Regularize firing rate
        num_of_losses = 4  # was 4 without r_E_reg_loss
        # TODO: Regularize firing rate
        val_loss_sets = torch.ones(self.val_dirs_len * self.trained_epochs,
                                   num_of_losses) * 3
        # [total, corr, L1, ks]
        valid_val_dir_count = 0
        for val_dir_i in range(self.val_dirs_len):
            val_dir = self.val_dirs[val_dir_i]
            if not os.path.exists(val_dir):
                print(f"{val_dir} doesn't exist.")
                continue
            valid_val_dir_count += 1
            for epoch in range(self.trained_epochs):
                param_val_path = os.path.join(val_dir,
                                              f'best_param{epoch}.pth')
                if not os.path.exists(param_val_path):
                    print(f"{param_val_path} doesn't exist.")
                    continue
                d = torch.load(param_val_path,
                               map_location=torch.device(self.device))
                parameter_sets[:, val_dir_i * self.trained_epochs +
                               epoch] = torch.squeeze(d['parameter'])
                val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                              1] = d['corr_loss']
                val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                              2] = d['l1_loss']
                val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                              3] = d['ks_loss']
                # TODO: Regularize firing rate
                # # key for the r_E regularization loss is 'r_E_reg_loss'
                # if 'r_E_reg_loss' in d:
                #     # TODO: Try without any constraint on r_E, can change the following line to assign zero values
                #     val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                #                   4] = d['r_E_reg_loss']
                #     # val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                #     #               4] = 0
                #     # TODO: Try without any constraint on r_E
                # TODO: Regularize firing rate
        if valid_val_dir_count == 0:
            print("No valid validated directories.")
            return 1
        # TODO: Try without FCD KS loss
        val_loss_sets[:, 0] = torch.sum(val_loss_sets[:, 1:], dim=1)
        # val_loss_sets[:, 0] = torch.sum(val_loss_sets[:, 1:3], dim=1)
        # TODO: Try without FCD KS loss
        # Record all param_10 and loss
        val_total_loss, sorted_index = torch.sort(val_loss_sets[:, 0],
                                                  descending=False)
        val_total_loss = val_total_loss[:self.param_sets]
        sorted_index = sorted_index[:self.param_sets]
        all_loss = val_loss_sets[sorted_index]
        parameter = parameter_sets[:, sorted_index]

        print('Start saving results...')
        save_dict = {
            'parameter': parameter,
            'val_total_loss': val_total_loss,
            'sorted_index': sorted_index,
            'corr_loss': all_loss[:, 1],
            'l1_loss': all_loss[:, 2],
            'ks_loss': all_loss[:, 3]
        }
        # TODO: Regularize firing rate
        # if all_loss.shape[1] > 4:
        #     save_dict['r_E_reg_loss'] = all_loss[:, 4]
        # TODO: Regularize firing rate
        torch.save(save_dict, os.path.join(self.test_dir, 'val_results.pth'))
        print("Successfully saved test results.")

        print(" -- Done testing -- ")
        return 0

    def select_best_from_train(self):
        print(" -- Start testing -- ")

        print(" -- Done testing -- ")
        return 0


def simulate_fc_fcd(config,
                    save_path,
                    parameter,
                    param_dup,
                    sc_euler,
                    fc_emp=None,
                    emp_fcd_cdf=None,
                    seed=None):
    # Save FC, FCD for each param_sets, Average across param_dup.
    dataset_parameters = config['Dataset Parameters']
    simulate_time = float(dataset_parameters['simulate_time'])
    burn_in_time = float(dataset_parameters['burn_in_time'])
    TR = float(dataset_parameters['TR'])
    window_size = int(dataset_parameters['window_size'])

    # Simulating parameters
    simulating_parameters = config['Simulating Parameters']
    N = int(simulating_parameters['n_ROI'])
    fcd_hist_bins = int(simulating_parameters['fcd_hist_bins'])
    param_sets = parameter.shape[1]
    # param_dup = int(simulating_parameters['param_dup'])
    euler_dt = float(simulating_parameters['dt_test'])

    print(" -- Start generating -- ")
    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 1000000000000)
    torch.manual_seed(seed)

    parameter_repeat = parameter.repeat(1, param_dup)  # [3*N+1, param_dup]
    mfm_model = MfmModel2014(config, parameter_repeat, sc_euler, dt=euler_dt)
    bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
        simulate_time=simulate_time, burn_in_time=burn_in_time, TR=TR)
    # bold_signal: [ROIs, param_dup, t]; valid_M_mask: [param_dup]
    bold_signal = bold_signal.view(N, param_dup, param_sets, -1).transpose(
        1, 2)  # [N, param_sets, param_dup, t_for_bold]
    valid_M_mask = valid_M_mask.view(param_dup,
                                     param_sets).T  # [param_sets, param_dup]

    valid_param_list = []  # record valid param index
    fc_sim = torch.zeros(param_sets, N, N)
    fcd_hist = torch.zeros(fcd_hist_bins, param_sets)
    count_valid = 0
    for i in range(param_sets):
        # for each set of parameter
        mask_this_param = valid_M_mask[i]  # [param_dup]
        if mask_this_param.any():
            valid_param_list.append(i)
            bold_this_param = bold_signal[:, i,
                                          mask_this_param, :]  # [N, 1/2/3/param_dup, t_for_bold]
            fc_this_param = MfmModel2014.FC_calculate(
                bold_this_param)  # [param_dup, roi, roi]
            fc_this_param = torch.mean(fc_this_param, dim=0)
            # fcd_mat_this_param, fcd_hist_this_param = MfmModel2014.FCD_calculate(bold_this_param, window_size)
            # You should not average across different FCD matrix
            # fcd_mat_this_param = torch.squeeze(fcd_mat_this_param)
            _, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                bold_this_param, window_size)
            fcd_hist_this_param = torch.mean(fcd_hist_this_param,
                                             dim=1)  # [bins]

            fc_sim[count_valid] = fc_this_param
            fcd_hist[:, count_valid] = fcd_hist_this_param
            count_valid += 1

    fc_sim = fc_sim[:count_valid]
    fcd_hist = fcd_hist[:, :count_valid]

    corr_loss = None
    L1_loss = None
    ks_loss = None
    if fc_emp is not None and emp_fcd_cdf is not None:
        _, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
            fc_sim, fcd_hist, fc_emp, emp_fcd_cdf)  # [param_dup]
    valid_param_list = torch.as_tensor(valid_param_list)

    print('Start saving results...')
    # save_dict = {'fc': fc_this_param, 'fcd': fcd_mat_this_param, 'corr_loss': corr_loss,
    #              'l1_loss': L1_loss, 'ks_loss': ks_loss, 'seed': seed, 'param_10': param_10,    'parameter': parameter}
    save_dict = {
        'valid_param_list': valid_param_list,
        'fc': fc_sim,
        'fcd_pdf': fcd_hist,
        'corr_loss': corr_loss,
        'l1_loss': L1_loss,
        'ks_loss': ks_loss,
        'seed': seed,
        'parameter': parameter
    }
    torch.save(save_dict, save_path)
    print("Successfully saved FC and FCD and losses.")

    print(" -- Done generating -- ")
    return 0


# TODO: Modify this as well if we want to regularize firing rate for HCPYA
def simulate_fc_fcd_mat(config,
                        save_path,
                        parameter,
                        param_dup,
                        sc_euler,
                        seed=None):
    # Save FC, FCD for each param_dup, rather than average.
    dataset_parameters = config['Dataset Parameters']
    simulate_time = float(dataset_parameters['simulate_time'])
    burn_in_time = float(dataset_parameters['burn_in_time'])
    TR = float(dataset_parameters['TR'])
    window_size = int(dataset_parameters['window_size'])

    # Simulating parameters
    simulating_parameters = config['Simulating Parameters']
    N = int(simulating_parameters['n_ROI'])
    fcd_hist_bins = int(simulating_parameters['fcd_hist_bins'])
    param_sets = parameter.shape[1]
    # param_dup = int(simulating_parameters['param_dup'])
    euler_dt = float(simulating_parameters['dt_test'])

    print(" -- Start generating -- ")
    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 1000000000000)
    torch.manual_seed(seed)

    parameter_repeat = parameter.repeat(1, param_dup)  # [3*N+1, param_dup]
    mfm_model = MfmModel2014(config, parameter_repeat, sc_euler, dt=euler_dt)
    bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
        simulate_time=simulate_time, burn_in_time=burn_in_time, TR=TR)
    # bold_signal: [ROIs, param_dup, t]; valid_M_mask: [param_dup]
    bold_signal = bold_signal.view(N, param_dup, param_sets, -1).transpose(
        1, 2)  # [N, param_sets, param_dup, t_for_bold]
    valid_M_mask = valid_M_mask.view(param_dup,
                                     param_sets).T  # [param_sets, param_dup]

    valid_param_list = []  # record valid param index
    fc_sim = torch.zeros(param_sets, param_dup, N, N)
    window_num = bold_signal.shape[3] - window_size + 1
    fcd_mats = torch.zeros(param_sets, param_dup, window_num, window_num)
    fcd_hist = torch.zeros(param_sets, fcd_hist_bins, param_dup)
    count_valid = 0
    for i in range(param_sets):
        # for each set of parameter
        mask_this_param = valid_M_mask[i]  # [param_dup]
        if mask_this_param.all():  # Must all be true
            valid_param_list.append(i)
            bold_this_param = bold_signal[:, i,
                                          mask_this_param, :]  # [N, param_dup, t_for_bold]
            fc_sim[count_valid] = MfmModel2014.FC_calculate(
                bold_this_param)  # [param_dup, roi, roi]
            fcd_mats[count_valid], fcd_hist[
                count_valid] = MfmModel2014.FCD_calculate(
                    bold_this_param,
                    window_size)  # [M, window_num, window_num], [bins, M]
            # You should not average across different FCD matrix
            # fcd_mat_this_param = torch.squeeze(fcd_mat_this_param)
            count_valid += 1

    fc_sim = fc_sim[:count_valid]
    fcd_mats = fcd_mats[:count_valid]
    fcd_hist = fcd_hist[:count_valid]

    valid_param_list = torch.as_tensor(valid_param_list)

    print('Start saving results...')
    # save_dict = {'fc': fc_this_param, 'fcd': fcd_mat_this_param, 'corr_loss': corr_loss,
    #              'l1_loss': L1_loss, 'ks_loss': ks_loss, 'seed': seed, 'param_10': param_10,    'parameter': parameter}
    save_dict = {
        'valid_param_list': valid_param_list,
        'fc': fc_sim,
        'fcd_mats': fcd_mats,
        'fcd_pdf': fcd_hist,
        'seed': seed,
        'parameter': parameter
    }
    torch.save(save_dict, save_path)
    print("Successfully saved FC and FCD and losses.")

    print(" -- Done generating -- ")
    return 0


def simulate_fc(config,
                save_path,
                parameter,
                param_dup,
                sc_euler,
                fc_emp,
                emp_fcd_cdf,
                seed=None):
    # Set MFM hyper-parameters
    dataset_parameters = config['Dataset Parameters']
    simulate_time = float(dataset_parameters['simulate_time'])
    burn_in_time = float(dataset_parameters['burn_in_time'])
    TR = float(dataset_parameters['TR'])
    window_size = int(dataset_parameters['window_size'])

    # Simulating parameters
    simulating_parameters = config['Simulating Parameters']
    # param_dup = int(simulating_parameters['param_dup'])
    euler_dt = float(simulating_parameters['dt_test'])

    print(" -- Start simulating -- ")
    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 1000000000000)
    torch.manual_seed(seed)

    parameter_repeat = parameter.repeat(
        1, param_dup)  # [3*N+1, param_sets * param_dup]
    mfm_model = MfmModel2014(config, parameter_repeat, sc_euler, dt=euler_dt)
    bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
        simulate_time=simulate_time, burn_in_time=burn_in_time, TR=TR)
    # bold_signal: [ROIs, param_dup, t]; valid_M_mask: [param_dup]
    if valid_M_mask.any():
        bold_this_param = bold_signal[:,
                                      valid_M_mask, :]  # [N, 1/2/3/param_dup, t_for_bold]
        fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
        fc_this_param = torch.mean(fc_this_param, dim=0)
        fcd_mat_this_param, fcd_hist_this_param = MfmModel2014.FCD_calculate(
            bold_this_param, window_size)
        # You should not average across different FCD matrix
        fcd_mat_this_param = torch.squeeze(fcd_mat_this_param)
        fcd_hist_this_param = torch.mean(fcd_hist_this_param,
                                         dim=1,
                                         keepdim=True)  # [10000, 1]

    else:
        print("The chosen parameter fails Euler.")
        return 1

    _, corr_loss, L1_loss, ks_loss = MfmModel2014.all_loss_calculate_from_fc_fcd(
        fc_this_param.unsqueeze(0), fcd_hist_this_param, fc_emp,
        emp_fcd_cdf)  # [1]
    print('Start saving results...')

    save_dict = {
        'fc': fc_this_param,
        'corr_loss': corr_loss,
        'l1_loss': L1_loss,
        'ks_loss': ks_loss,
        'parameter': parameter,
        'seed': seed
    }
    torch.save(save_dict, save_path)
    print("Successfully saved FC and losses.")

    print(" -- Done Simulating -- ")
    return 0


def get_EI_ratio(config, save_path, parameter, param_dup, sc_euler, seed=None):
    """_summary_

    Args:
        save_path (_type_): _description_
        parameter (tensor): [205, param_sets]
        param_dup (_type_): _description_
        sc_euler (_type_): _description_
    """
    # Set MFM hyper-parameters
    dataset_parameters = config['Dataset Parameters']
    simulate_time = float(dataset_parameters['simulate_time'])
    burn_in_time = float(dataset_parameters['burn_in_time'])
    TR = float(dataset_parameters['TR'])

    # Simulating parameters
    simulating_parameters = config['Simulating Parameters']
    # param_dup = int(simulating_parameters['param_dup'])
    euler_dt = float(simulating_parameters['dt_test'])
    # euler_dt = 0.006

    print(" -- Start simulating -- ")
    # Set random seed
    if seed is None:
        seed = np.random.randint(0, 1000000000000)
    torch.manual_seed(seed)

    parameter_repeat = parameter.repeat(
        1, param_dup)  # [3*N+1, param_sets * param_dup]
    mfm_model = MfmModel2014(config, parameter_repeat, sc_euler, dt=euler_dt)
    bold_signal, valid_M_mask, s_e_ave, s_i_ave, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
        simulate_time=simulate_time,
        burn_in_time=burn_in_time,
        TR=TR,
        need_EI=True)
    print(f"Bold {bold_signal.shape}; S_E {s_e_ave.shape}")
    # bold_signal = bold_signal.view(roi_num, param_dup, param_sets, -1).transpose(1, 2)  # [N, param_sets, param_dup, t_for_bold]
    # valid_M_mask = valid_M_mask.view(param_dup, param_sets).T  # [param_sets, param_dup]
    # S_E: [n_roi, param_dup]

    # for i in range(param_sets):
    # for each set of parameter
    # mask_this_param = valid_M_mask[i]  # [param_dup]
    if valid_M_mask.any():
        s_e_this_param = s_e_ave[:, valid_M_mask]
        s_i_this_param = s_i_ave[:, valid_M_mask]
        ei_ratio = s_e_this_param / s_i_this_param  # [roi, valid_number]
        ei_ratio = torch.mean(ei_ratio, dim=1, keepdim=True)
        s_e_ave_ = torch.mean(s_e_this_param, dim=1, keepdim=True)  # [roi, 1]
        s_i_ave_ = torch.mean(s_i_this_param, dim=1, keepdim=True)
    else:
        print("No valid run.")
        return 1
    print("Start saving results...")
    save_dict = {
        'ei_ratio': ei_ratio,
        's_e_ave': s_e_ave_,
        's_i_ave': s_i_ave_,
        'parameter': parameter,
        'seed': seed
    }
    # TODO: Save r_E
    if mfm_model.r_E and valid_M_mask.any():
        # save r_E_for_valid_params
        r_E_for_valid_params = r_E_ave[:, valid_M_mask]
        save_dict['r_E_for_valid_params'] = r_E_for_valid_params
        # TODO: Regularize firing rate
        # # get and save r_E_reg_loss
        # r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
        #                                 mfm_model.r_E,
        #                                 loss_type='L2')
        # save_dict['r_E_reg_loss'] = r_E_reg_loss
        # TODO: Regularize firing rate
    # TODO: Save r_E
    torch.save(save_dict, save_path)
    print("Successfully saved EI ratio.")
    print(" -- Done simulating -- ")
    return 0
