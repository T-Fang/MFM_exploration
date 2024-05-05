import pandas as pd
import numpy as np
import torch
import torch.distributions as td
import os
import datetime

from src.models.model_predictor_classifier import ClassifyNanModel_2, PredictLossModel_1, ClassifyNanModel_Yan100, PredictLossModel_1_Yan100
from src.models.mfm_2014 import MfmModel2014
from src.basic.constants import DESIKAN_NEUROMAPS_DIR  # noqa: F401
from src.utils.file_utils import combine_all_param_dicts, get_best_params_file_path, get_best_params_sim_res_path, get_train_file_path
from src.utils.tzeng_func_torch import parameterize_myelin_rsfc
from src.utils.CBIG_func_torch import CBIG_corr
from src.utils.neuromaps_utils import get_concat_matrix  # noqa: F401
from src.utils.init_utils import set_torch_default


def csv2tensor(csv_path):
    content = pd.read_csv(csv_path, header=None, index_col=False)
    content = np.array(content)
    return torch.as_tensor(content, dtype=torch.double)


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
    index_min = save_dict['valid_param_indices'][index_in_valid]
    if 'parameter' in save_dict:
        best_parameter = save_dict['parameter'][:, index_min]
    elif 'param_coef' in save_dict:
        best_parameter = parameterize_myelin_rsfc(
            save_dict['param_coef'][:, index_min])
    else:
        raise Exception(
            "Error in parameter saving. Key 'parameter' and 'param_coef' both not exists."
        )

    return best_parameter


def train_help_function(config,
                        emp_stats: dict[str, torch.Tensor],
                        train_save_dir: str,
                        num_epochs,
                        dl_pfic_range,
                        euler_pfic_range,
                        dl_rfic_range,
                        euler_rfic_range,
                        query_wei_range,
                        opportunities,
                        next_epoch,
                        seed=None,
                        other_parameterization=None):
    # * TODO: comment out codes if we want to use the original parameterization
    other_parameterization = get_concat_matrix(
        DESIKAN_NEUROMAPS_DIR,
        PCs=3,
        use_mean_map=True,
        use_standardizing=True)  # shape: (N, num_of_PCs + 2)
    # * TODO: comment out codes if we want to use the original parameterization
    mfm_trainer = CMAESTrainer(config=config,
                               emp_stats=emp_stats,
                               train_save_dir=train_save_dir,
                               num_epochs=num_epochs,
                               dl_pfic_range=dl_pfic_range,
                               euler_pfic_range=euler_pfic_range,
                               dl_rfic_range=dl_rfic_range,
                               euler_rfic_range=euler_rfic_range,
                               query_wei_range=query_wei_range,
                               other_parameterization=other_parameterization)
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


class ModelHandler:
    """
    Parent class for Trainer, Validator, and Tester
    """

    def __init__(self, config, phase, emp_stats: dict[str, torch.Tensor],
                 prev_phase_best_params_path: str, curr_phase_save_dir: str):
        """
        Initialize the ModelHandler

        Args:
            config (dict): the configuration dictionary
            phase (str): the phase of the model handler, must be one of ['train', 'val', 'test']
            emp_stats (dict): the empirical statistics, which contains:
                sc_mat (tensor): [N, N] the structural connectivity matrix
                sc_euler (tensor): [N, N] the structural connectivity matrix for Euler integration
                emp_fc (tensor): [N, N] the empirical functional connectivity matrix
                emp_fcd_cum (tensor): [bins, 1] the empirical FCD cumulative histogram
            prev_phase_best_params_path (str): path to the best params from the previous phase
            curr_phase_save_dir (str): the save directory of the current phase
        """

        assert phase in ['train', 'val', 'test'
                         ], "phase must be one of ['train', 'val', 'test']"
        self.device, self.dtype = set_torch_default()
        self.config = config
        self.phase = phase
        self.curr_phase_save_dir = curr_phase_save_dir
        self.prev_phase_best_params_path = prev_phase_best_params_path
        os.makedirs(self.curr_phase_save_dir, exist_ok=True)

        # # Check the config file for descriptions of the parameters
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
        self.test_param_sets = int(simulating_parameters['test_param_sets'])
        self.select_param_sets = int(
            simulating_parameters['select_param_sets'])
        self.min_select_param_sets = int(
            simulating_parameters['min_select_param_sets'])
        self.param_dup = int(simulating_parameters['param_dup'])
        self.fcd_hist_bins = int(simulating_parameters['fcd_hist_bins'])
        self.dt = float(simulating_parameters[f'dt_{phase}'])

        # TODO: Regularize firing rate
        self.use_r_E_reg_loss = False
        # TODO: Regularize firing rate
        # TODO: Save r_E
        self.save_r_E = True
        # TODO: Save r_E

        # Empirical data
        self.sc_euler = emp_stats['sc_euler']  # [N, N] for Euler integration
        self.fc_euler = emp_stats['emp_fc']  # [N, N]
        self.fcd_cum_euler = emp_stats['emp_fcd_cum']  # [bins, 1]

        if self.phase == 'train':
            sc_mask = torch.triu(torch.ones(self.N, self.N, dtype=torch.bool),
                                 1)  # Upper triangle
            self.sc_dl = torch.as_tensor(
                self.sc_euler[sc_mask])  # [N * (N - 1) / 2]
            self.fc_dl = self.fc_euler[sc_mask]  # [N * (N - 1) / 2]
            self.fcd_dl = torch.diff(self.fcd_cum_euler.squeeze() * 100,
                                     dim=0,
                                     prepend=torch.as_tensor([0]))  # [bins]
        print(self.post_init_message)

    @property
    def post_init_message(self):
        return f"Successfully init CMAES ModelHandler at phase {self.phase}. The results will be saved in {self.curr_phase_save_dir}"

    @property
    def prev_phase(self):
        return self.get_prev_phase(self.phase)

    @staticmethod
    def get_prev_phase(phase: str):
        if phase == 'train':
            return 'train'
        elif phase == 'val':
            return 'train'
        elif phase == 'test':
            return 'val'

    def sim_param_with_dup(self, param_vectors, reshape_res=True):
        """
        Simulate the MFM model with duplicated parameters

        Args:
            param_vectors (tensor): [3N+1, param_sets]

        Returns:
            bold_signal (tensor): (N, param_sets, param_dup, #time_points_in_BOLD)
            valid_M_mask (tensor): (param_sets, param_dup)
            r_E_ave (tensor): (N, param_sets, param_dup)
        """
        parameter_repeat = param_vectors.repeat(
            1, self.param_dup)  # [3*N+1, param_sets * param_dup]
        mfm_model = MfmModel2014(self.config,
                                 parameter_repeat,
                                 self.sc_euler,
                                 dt=self.dt)
        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)
        # bold_signal, valid_M_mask = CBIG_mfm_single_simulation_no_input_noise(parameter_repeat, sc_mat, self.t_epochlong)

        if reshape_res:
            bold_signal, valid_M_mask, r_E_ave = self.reshape_sim_res(
                bold_signal, valid_M_mask, r_E_ave)
        return bold_signal, valid_M_mask, r_E_ave

    def reshape_sim_res(self, bold_signal, valid_M_mask, r_E_ave):
        """
        Reshape the simulation results

        Args:
            bold_signal (tensor): (N, param_sets * param_dup, #time_points_in_BOLD)
            valid_M_mask (tensor): (param_sets * aram_dup, )
            r_E_ave (tensor): (N, param_sets * param_dup)

        Returns:
            bold_signal (tensor): (N, param_sets, param_dup, #time_points_in_BOLD)
            valid_M_mask (tensor): (param_sets, param_dup)
            r_E_ave (tensor): (N, param_sets, param_dup)
        """
        param_sets = bold_signal.shape[1] // self.param_dup
        bold_signal = bold_signal.view(
            self.N, self.param_dup, param_sets, -1).transpose(
                1, 2)  # [N, param_sets, param_dup, #time_points_in_BOLD]
        valid_M_mask = valid_M_mask.view(
            self.param_dup, param_sets).T  # [param_sets, param_dup]
        r_E_ave = r_E_ave.view(self.N, self.param_dup, param_sets).transpose(
            1, 2)  # [N, param_sets, param_dup]
        return bold_signal, valid_M_mask, r_E_ave

    def get_valid_params(self, valid_M_mask):
        """
        Get the valid param_vectors

        Args:
            valid_M_mask (tensor): (param_sets, param_dup)

        Returns:
            valid_param_indices (list): the list of valid parameter indices
        """
        if valid_M_mask.ndim == 1:
            valid_M_mask = valid_M_mask.unsqueeze(0)
        valid_param_indices = []
        for i in range(valid_M_mask.shape[0]):
            mask_this_param = valid_M_mask[i]
            if mask_this_param.any():
                valid_param_indices.append(i)
        return valid_param_indices

    def get_fc_fcd_for_valid_params(self,
                                    valid_param_indices,
                                    bold_signal,
                                    valid_M_mask,
                                    r_E_ave,
                                    get_FCD_matrix=False,
                                    get_mean_bold=False):
        """
        Get the FC and FCD for the valid param_vectors

        Args:
            valid_param_indices (list): the list of valid parameter indices
            bold_signal (tensor): (N, param_sets, param_dup, #time_points_in_BOLD)
            valid_M_mask (tensor): (param_sets, param_dup)
            r_E_ave (tensor): (N, param_sets, param_dup)
        Returns:
            fc_sim (tensor): (#valid_params, N, N)
            fcd_sim (tensor): (fcd_hist_bins, #valid_params) or (#valid_params, window_num, window_num) if get_FCD_matrix is True
            r_E_for_valid_params (tensor): (N, #valid_params)
            mean_bold (tensor): (N, #valid_params, #time_points_in_BOLD) if get_mean_bold is True
        """
        num_valid = len(valid_param_indices)
        t_len = bold_signal.shape[3]
        fc_sim = torch.zeros(num_valid, self.N, self.N)
        fcd_hist = torch.zeros(self.fcd_hist_bins, num_valid)
        mean_bold = torch.zeros(self.N, num_valid, t_len)

        if get_FCD_matrix:
            window_num = t_len - self.window_size + 1
            fcd_matrices = torch.zeros(num_valid, window_num, window_num)

        if self.save_r_E:
            r_E_for_valid_params = torch.zeros(self.N, num_valid)
        else:
            r_E_for_valid_params = None

        for i, idx in enumerate(valid_param_indices):
            # for each set of parameter
            mask_this_param = valid_M_mask[idx]  # (param_dup, )
            bold_this_param = bold_signal[:, idx,
                                          mask_this_param, :]  # [N, 1/2/3/param_dup, #time_points_in_BOLD]
            if get_mean_bold:
                mean_bold[:, i] = torch.mean(bold_this_param, dim=1)

            fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
            fc_this_param = torch.mean(fc_this_param, dim=0)
            fcd_mat_this_param, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                bold_this_param,
                self.window_size,
                get_FCD_matrix=get_FCD_matrix)
            fcd_hist_this_param = torch.mean(fcd_hist_this_param, dim=1)

            fc_sim[i] = fc_this_param
            fcd_hist[:, i] = fcd_hist_this_param
            if get_FCD_matrix:
                # ! we should't directly average fcd matrices
                fcd_matrices[i] = fcd_mat_this_param[0]
            if self.save_r_E:
                r_E_ave_this_param = r_E_ave[:, idx,
                                             mask_this_param]  # [N, 1/2/3/param_dup]
                r_E_for_valid_params[:, i] = torch.mean(r_E_ave_this_param,
                                                        dim=1)

        extra = {}
        if get_FCD_matrix:
            extra['fcd_sim'] = fcd_matrices
            extra['fc_sim'] = fc_sim
        if get_mean_bold:
            extra['bold_TC_sim'] = mean_bold

        return fc_sim, fcd_hist, r_E_for_valid_params, extra

    def calc_losses(self,
                    fc_sim,
                    fcd_hist_sim,
                    reg_loss=None,
                    apply_sort=True):
        """
        Calculate the losses for the simulated FC and FCD

        Args:
            fc_sim (tensor): (#valid_params, N, N)
            fcd_hist_sim (tensor): (fcd_hist_bins, #valid_params)
            reg_loss (dict): the regularization loss
            apply_sort (bool): whether to sort the losses
        """
        losses = MfmModel2014.calc_all_loss_from_fc_fcd(
            fc_sim, fcd_hist_sim, self.fc_euler,
            self.fcd_cum_euler)  # [num_valid]

        # * TODO: Modify losses
        total_loss = losses['corr_loss'] + losses['l1_loss'] + losses['ks_loss']
        # total_loss = losses['corr_loss'] + losses['l1_loss']
        # total_loss = losses['l1_loss'] + losses['ks_loss']
        # * TODO: Modify losses

        if reg_loss is not None:
            for loss in reg_loss.values():
                total_loss = total_loss + loss

        index_sorted_in_valid = None
        if apply_sort:
            total_loss, index_sorted_in_valid = torch.sort(total_loss,
                                                           descending=False)
            for key in losses.keys():
                losses[key] = losses[key][index_sorted_in_valid]

            if reg_loss is not None:
                for key in reg_loss.keys():
                    losses[key] = reg_loss[key][index_sorted_in_valid]

        losses['total_loss'] = total_loss

        return losses, index_sorted_in_valid

    def get_reg_loss(self,
                     param_vectors,
                     use_rFIC=False,
                     r_E_for_valid_params=None):
        """
        Get the regularization loss
        :param param_vectors: [3*N+1, num_valid]
        :param use_rFIC: False stands for pFIC, True stands for rFIC
        :param r_E_for_valid_params: [N, num_valid]

        :return: reg_loss: {'rFIC_reg_loss': [num_valid, ], 'r_E_reg_loss': [num_valid, ]}
        """
        reg_loss = {}
        if r_E_for_valid_params is not None and self.use_r_E_reg_loss:
            r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
                                            3,
                                            loss_type='L2')
            reg_loss['r_E_reg_loss'] = r_E_reg_loss

        return reg_loss

    def sim_best_param_multi_times(self,
                                   sim_res_dir: str,
                                   sim_times: int = 10,
                                   get_FCD_matrix: bool = True,
                                   get_bold: bool = True,
                                   seed=None):
        """
        Simulate the MFM model multiple times with the best param vector.
        Then save the simulation result to sim_res_dir

        Args:
            sim_res_dir (str): the dir to save the simulation results
            sim_times (int): the number of times to simulate
            get_FCD_matrix (bool): whether to save the entire FCD matrix instead of the histogram of the FCD
            get_bold (bool): whether to get the mean bold signal
        """

        if seed is None:
            seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)
        best_params_path = get_best_params_file_path(self.phase,
                                                     self.curr_phase_save_dir)
        saved_params = torch.load(best_params_path,
                                  map_location=self.device)['parameter']
        param_vector = saved_params[:, 0:1]  # use the best param_vector
        param_vectors = param_vector.repeat(
            1, sim_times)  # [3*N+1, sim_times * param_dup]
        print(
            f'Simulating the MFM model {sim_times} times with the best parameter vector from {best_params_path}...'
        )
        # print(f'param_vectors shape: {param_vectors.shape}')

        save_dict = self.sim_n_get_losses(param_vectors,
                                          get_FCD_matrix=get_FCD_matrix,
                                          get_bold=get_bold)
        save_dict['seed'] = seed

        os.makedirs(sim_res_dir, exist_ok=True)
        save_path = get_best_params_sim_res_path(self.phase, self.phase,
                                                 sim_res_dir)

        torch.save(save_dict, save_path)
        print(f'Successfully saved the simulation results to {save_path}')

    def sim_n_get_fc_fcd(self,
                         param_vectors,
                         get_FCD_matrix: bool = False,
                         get_bold: bool = False):
        """
        Simulate the MFM using the given param_vectors and get the corresponding FC and FCD for valid param_vectors.

        Args:
            param_vectors (tensor): (3N+1, param_sets)
            save_FCD_matrix (bool): whether to save the entire FCD matrix instead of the histogram of the FCD
            get_bold (bool): whether to get the mean bold signal

        Returns:
            fc_sim (tensor): (#valid_params, N, N)
            fcd_sim (tensor): (fcd_hist_bins, #valid_params) or (#valid_params, window_num, window_num) if get_FCD_matrix is True
            r_E_for_valid_params (tensor): (N, #valid_params)
            valid_param_indices (list[int]): the list of valid parameter indices
        """

        bold_signal, valid_M_mask, r_E_ave = self.sim_param_with_dup(
            param_vectors, reshape_res=True)
        valid_param_indices = self.get_valid_params(valid_M_mask)
        fc_sim, fcd_sim, r_E_for_valid_params, extra = self.get_fc_fcd_for_valid_params(
            valid_param_indices,
            bold_signal,
            valid_M_mask,
            r_E_ave,
            get_FCD_matrix,
            get_mean_bold=get_bold)

        return fc_sim, fcd_sim, r_E_for_valid_params, valid_param_indices, extra

    def sim_n_get_losses(self,
                         param_vectors,
                         use_rFIC=False,
                         get_FCD_matrix: bool = False,
                         get_bold: bool = False):
        """
        Simulate the MFM using the given param_vectors and get their corresponding losses.
        The param_vectors, valid_param_indices, losses for valid param vectors are returned in a dictionary.
        Additionally, if self.save_r_E is True, r_E_for_valid_params is also returned.

        Args:
            param_vectors (tensor): [3N+1, param_sets]
            use_rFIC (bool): whether to use rFIC or not
            get_bold (bool): whether to get the mean bold signal

        Returns:
            a dictionary containing the following keys:
                'total_loss': (#valid_params, )
                'corr_loss': (#valid_params, )
                'l1_loss': (#valid_params, )
                'ks_loss': (#valid_params, )
                'rFIC_reg_loss': (#valid_params, ) if use_rFIC is True
                'r_E_reg_loss': (#valid_params, ) if self.save_r_E is True
                'parameter': (3N+1, param_sets)
                'valid_param_indices': (#valid_params, )
                'r_E_for_valid_params': (N, #valid_params) if self.save_r_E is True
        """
        fc_sim, fcd_hist_sim, r_E_for_valid_params, valid_param_indices, extra = self.sim_n_get_fc_fcd(
            param_vectors, get_FCD_matrix=get_FCD_matrix, get_bold=get_bold)

        reg_loss = self.get_reg_loss(param_vectors[:, valid_param_indices],
                                     use_rFIC, r_E_for_valid_params)

        losses, index_sorted_in_valid = self.calc_losses(fc_sim,
                                                         fcd_hist_sim,
                                                         reg_loss,
                                                         apply_sort=True)

        valid_param_indices = torch.as_tensor(valid_param_indices)
        valid_param_indices = valid_param_indices[index_sorted_in_valid]

        save_dict = losses
        save_dict['valid_param_indices'] = valid_param_indices
        save_dict['parameter'] = param_vectors[:, valid_param_indices]

        if self.save_r_E:
            save_dict[
                'r_E_for_valid_params'] = r_E_for_valid_params[:,
                                                               index_sorted_in_valid]
        if 'fcd_sim' in extra:
            save_dict['fcd_sim'] = extra['fcd_sim'][index_sorted_in_valid]
        if 'fc_sim' in extra:
            save_dict['fc_sim'] = extra['fc_sim'][index_sorted_in_valid]
        if 'bold_TC_sim' in extra:
            save_dict['bold_TC_sim'] = extra[
                'bold_TC_sim'][:, index_sorted_in_valid]

        return save_dict


class CMAESTrainer(ModelHandler):

    def __init__(self,
                 config,
                 emp_stats: dict[str, torch.Tensor],
                 train_save_dir,
                 num_epochs: int = 100,
                 dl_pfic_range=[],
                 euler_pfic_range=np.arange(0, 100),
                 dl_rfic_range=[],
                 euler_rfic_range=[],
                 query_wei_range: str = 'Uniform',
                 other_parameterization=None):
        """Initialize Hybrid CMA-ES trainer

        Args:
            dataset_name (str): ['HCP', 'PNC', 'HCP_Dev']. The name of dataset. To get the euler hyperparameters
            emp_stats (dict): the empirical statistics, which contains:
                myelin (tensor): [ROIs, 1]
                rsfc_gradient (tensor): [ROIs, 1]
                sc_mat (tensor): [ROIs, ROIs]
                sc_euler (tensor): [ROIs, ROIs]
                emp_fc (tensor): [ROIs, ROIs]
                emp_fcd_cum (tensor): [bins, 1], has been normalized (largest is 1)
            save_param_dir (str): the parameters saving directory
            epochs (int): total epochs
            other_parameterization (np.ndarray): if we want to use another parameterization,
                we will feed in the new concat matrix of shape (N, p) here,
                where p is the number of parameterization variables
        """

        super().__init__(config, 'train', emp_stats, train_save_dir,
                         train_save_dir)

        self.myelin = emp_stats['myelin']  # [N, 1]
        self.rsfc_gradient = emp_stats['rsfc_gradient']  # [N, 1]

        # TODO: Use PCs and mean of neuromaps to parameterize wEE and wEI
        self.other_parameterization = other_parameterization
        if other_parameterization is None:
            self.concat_mat = torch.hstack(
                (torch.ones_like(self.myelin), self.myelin,
                 self.rsfc_gradient))  # (N, 3)
            self.p = 3
        else:
            self.concat_mat = other_parameterization  # (N, p), where p = num_of_PCs + 2
            self.p = other_parameterization.shape[1]
        self.pinv_concat_mat = torch.linalg.pinv(self.concat_mat)
        # TODO: Use PCs and mean of neuromaps to parameterize wEE and wEI

        self.parameter_dim = 3 * self.N + 1  # 3N + 1. The dimension of parameters in MFM model.

        # TODO: change to 2 * self.p + 1, if we want to totally remove parameterization for sigma
        self.param_dim = 3 * self.p + 1  # The dimension of the parameters in our parameterization

        # Training and generating parameters
        self.num_epochs = int(num_epochs)
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
                           map_location=self.device)['model_state_dict'])
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
                           map_location=self.device)['model_state_dict'])
            print('Successfully loaded.')

        if len(self.dl_rfic_range) > 0:
            raise Exception("rFIC deep learning model has not been developed.")

    def get_parameters(self, param_coef):
        """
        From the parameterization coefficients to get the 3N+1 parameters
        :param param_coef: [3 * self.p + 1, param_sets]
        :return: parameters for 2014 Deco Model [3N+1, param_sets]
        """
        w_EE = torch.matmul(self.concat_mat,
                            param_coef[0:self.p])  # shape: (N, param_sets)
        w_EI = torch.matmul(self.concat_mat,
                            param_coef[self.p:2 *
                                       self.p])  # shape: (N, param_sets)
        G = param_coef[2 * self.p].unsqueeze(0)  # shape: (param_sets, )
        sigma = torch.matmul(self.concat_mat,
                             param_coef[2 * self.p +
                                        1:])  # shape: (N, param_sets)

        return torch.cat((w_EE, w_EI, G, sigma),
                         dim=0)  # shape: (3N+1, param_sets)

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
        :param param_coef: 3 * self.p + parameterization parameters, just for saving.
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
            valid_param_indices = torch.arange(0, parameter.shape[0], 1)
            valid_param_indices = valid_param_indices[valid_mask]

            count_valid = len(valid_param_indices)
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

            valid_parameter = parameter[valid_param_indices]
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
            index_sorted = valid_param_indices[index_sorted_in_valid]

            print("Start saving parameter, valid_param_indices and loss...")
            save_dict = {
                'parameter': parameter.T,
                'valid_param_indices': valid_param_indices,
                'FC_FCD_loss': pred_loss
            }
            train_file_path = get_train_file_path(self.curr_phase_save_dir,
                                                  epoch)
            torch.save(save_dict, train_file_path)
            print("Successfully saved params and losses.")

            return loss_sorted[:
                               select_param_sets], index_sorted[:
                                                                select_param_sets]

    def get_select_param_sets(self, num_valid):
        if num_valid >= self.select_param_sets:
            return self.select_param_sets
        else:
            print(
                f'Valid parameter sets are not enough, only {num_valid} parameters!'
            )
            if num_valid >= self.min_select_param_sets:
                print('Choose min parameter sets instead.')
                return num_valid
            else:
                return None

    def get_rFIC_reg_loss(self, param_vectors):
        """
        rFIC regularization loss
        w_wee (scalar): the weight for the regularization term for wee_regularization_loss.
        w_wei (scalar): the weight for the regularization term for wei_regularization_loss.
        w_sigma (scalar): the weight for the regularization term for sigma_regularization_loss.
        concat_mat (tensor): [bias (ones_like myelin), myelin, RSFC gradient]
        pinv_concat_mat (tensor, optional): pinv(concat_mat). Defaults to None.

        Args:
            parameters (tensor): wEE, wEI, G, sigma. [3N+1, param_sets]

        Returns:
            (tensors): [param_sets, 3]. wEE, wEI, sigma regularization loss terms
        """
        if self.pinv_concat_mat is None:
            self.pinv_concat_mat = torch.linalg.pinv(self.concat_mat)
        beta_wee = torch.matmul(self.pinv_concat_mat,
                                param_vectors[0:self.N])  # [3, param_sets]
        beta_wei = torch.matmul(self.pinv_concat_mat,
                                param_vectors[self.N:2 * self.N])
        beta_sigma = torch.matmul(self.pinv_concat_mat,
                                  param_vectors[2 * self.N + 1:])
        wee_hat = torch.matmul(self.concat_mat, beta_wee)  # [N, param_sets]
        wei_hat = torch.matmul(self.concat_mat, beta_wei)
        sigma_hat = torch.matmul(self.concat_mat, beta_sigma)

        wee_regularization_loss = self.w_wee * torch.mean(
            torch.square(param_vectors[0:self.N] - wee_hat),
            dim=0)  # [param_sets]
        wei_regularization_loss = self.w_wei * torch.mean(
            torch.square(param_vectors[self.N:2 * self.N] - wei_hat), dim=0)
        sigma_regularization_loss = self.w_sigma * torch.mean(
            torch.square(param_vectors[2 * self.N + 1:] - sigma_hat), dim=0)

        return torch.hstack((wee_regularization_loss.unsqueeze(1),
                             wei_regularization_loss.unsqueeze(1),
                             sigma_regularization_loss.unsqueeze(1)))

    def get_reg_loss(self,
                     param_vectors,
                     use_rFIC=False,
                     r_E_for_valid_params=None):
        """
        Get the regularization loss
        :param param_vectors: [3*N+1, num_valid]
        :param use_rFIC: False stands for pFIC, True stands for rFIC
        :param r_E_for_valid_params: [N, num_valid]

        :return: reg_loss: {'rFIC_reg_loss': [num_valid, ], 'r_E_reg_loss': [num_valid, ]}
        """
        reg_loss = super().get_reg_loss(param_vectors, use_rFIC,
                                        r_E_for_valid_params)
        if use_rFIC:
            rFIC_reg_loss = self.get_rFIC_reg_loss(
                param_vectors)  # shape: (num_valid, 3)
            rFIC_reg_loss = torch.sum(rFIC_reg_loss, dim=1)
            reg_loss['rFIC_reg_loss'] = rFIC_reg_loss

        return reg_loss

    def mfm_model_loss_euler(self, param_vectors, epoch, use_rFIC=False):
        """
        Calculate the total loss of MFM model
        :param parameter: [3*N+1, M]
        :param epoch: current epoch, just for naming the save file.
        :param use_rFIC: False stands for pFIC, True stands for rFIC
        :return: loss [select_param_sets(10),]; index [select_param_sets,]
        """
        save_dict = self.sim_n_get_losses(param_vectors, use_rFIC)

        valid_param_indices = save_dict['valid_param_indices']
        num_valid = len(valid_param_indices)
        select_param_sets = self.get_select_param_sets(num_valid)
        if select_param_sets is None:
            return None, None

        print("Start saving parameter, valid_param_indices and losses...")
        train_file_path = get_train_file_path(self.curr_phase_save_dir, epoch)
        torch.save(save_dict, train_file_path)
        print("Successfully saved params and losses to:", train_file_path)

        return save_dict['total_loss'][:select_param_sets], \
            valid_param_indices[:select_param_sets]

    def train_hybrid_pFIC_rFIC(self, seed=None, next_epoch=0):
        if next_epoch >= self.num_epochs:
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
            previous_final_state_path = os.path.join(self.curr_phase_save_dir,
                                                     'final_state_pFIC.pth')
            if not os.path.exists(previous_final_state_path):
                raise Exception("Previous final state path doesn't exist.")
            final_dict = torch.load(previous_final_state_path,
                                    map_location=self.device)
            seed = final_dict['seed']
            random_state = final_dict['random_state'].to(
                torch.ByteTensor(device=torch.device('cpu')))
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
            previous_final_state_path = os.path.join(self.curr_phase_save_dir,
                                                     'final_state_pFIC.pth')
            if not os.path.exists(previous_final_state_path):
                previous_final_state_path = os.path.join(
                    self.curr_phase_save_dir, 'final_state.pth')
                if not os.path.exists(previous_final_state_path):
                    raise Exception("Previous final state path doesn't exist.")
            print(f"Load final dict from {previous_final_state_path}...")
            final_dict = torch.load(previous_final_state_path,
                                    map_location=self.device)
            seed = final_dict['seed']
            random_state = final_dict['random_state'].to(
                torch.ByteTensor(device=torch.device('cpu')))
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
        for k in range(next_epoch, self.num_epochs):
            print("Epoch: [{}/{}]".format(k + 1, self.num_epochs))

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
                        os.path.join(self.curr_phase_save_dir,
                                     'final_state_pFIC.pth'))
                    print("Successfully saved pFIC final dict.")

            elif k in self.rfic_range:
                if k == self.rfic_range[0]:
                    previous_epoch_path = get_train_file_path(
                        self.curr_phase_save_dir, k - 1)
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
                   os.path.join(self.curr_phase_save_dir, 'final_state.pth'))
        print("Successfully saved final dict.")

        return 0

    def _train_one_epoch_pFIC(self, k, m_k, sigma_k, cov_k, p_sigma_k, p_c_k):

        # TODO: Try uniform sampling at the first iteration
        # if k == 0:  # The first epoch
        #     param_coef_k, parameter_k = self._sample_uniform_parameters()
        # else:
        #     param_coef_k, parameter_k = self._sample_valid_parameters_pFIC(
        #         m_k, sigma_k**2 * cov_k, self.search_range)

        # Original sampling
        cov_for_gaussian = sigma_k**2 * cov_k
        # check if cov_next contain nan
        if torch.isnan(cov_for_gaussian).any():
            print("cov_for_gaussian contains nan!")
            print("sigma_k: ", sigma_k)
            print("cov_k: ", cov_k)
            return None
        param_coef_k, parameter_k = self._sample_valid_parameters_pFIC(
            m_k, cov_for_gaussian, self.search_range)
        # TODO: Try uniform sampling at the first iteration
        if param_coef_k is None or parameter_k is None:
            print("Sampling failed!")
            return None

        print(f"Successfully sampled {param_coef_k.shape[1]} parameters.",
              flush=True)

        if k in self.dl_pfic_range:
            loss_k, index_k = self.mfm_model_loss_dl(parameter_k, k)
        elif k in self.euler_pfic_range:
            loss_k, index_k = self.mfm_model_loss_euler(parameter_k, k)
        else:
            raise Exception("Not pFIC epoch. Break.")

        if loss_k is None:
            print("Iteration ends.")
            return None
        select_params = param_coef_k[:, index_k]

        # TODO: Try uniform sampling at the first iteration
        # if k == 0:  # The first epoch
        #     m_k = torch.mean(select_params, dim=1)
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
                                                    use_rFIC=True)

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
        m_0[0:self.p] = start_point_wEE
        # m_0[1] = start_point_wEE[1] / 2
        m_0[self.p:2 * self.p] = start_point_wEI
        # m_0[self.p + 1] = start_point_wEI[1] / 2
        m_0[2 * self.p] = init_parameters[2 * N]  # G
        m_0[2 * self.p + 1:] = start_point_sigma

        sigma_0 = 0.2
        p_sigma_0 = torch.zeros(self.param_dim, 1)
        p_c_0 = torch.zeros(self.param_dim, 1)
        V_ini = torch.eye(self.param_dim)
        Lambda_ini = torch.ones(self.param_dim)
        # Lambda_ini[0:self.p] = start_point_wEE[0] / 2
        # Lambda_ini[self.p:2 * self.p] = start_point_wEI[0] / 2
        Lambda_ini[0:self.p] = start_point_wEE[0]
        Lambda_ini[self.p:2 * self.p] = start_point_wEI[0]
        Lambda_ini[2 * self.p] = 0.4
        Lambda_ini[2 * self.p + 1:] = 0.0005
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
        # TODO: remove all prints
        # print("Start updating parameters in CMA-ES algorithm...")
        # # print all inputs
        # print("select_params: ", select_params)
        # print("loss_k: ", loss_k)
        # print("m_k: ", m_k)
        # print("sigma_k: ", sigma_k)
        # print("cov_k: ", cov_k)

        select_param_sets = select_params.shape[1]
        loss_inverse = 1 / loss_k
        weights = loss_inverse / torch.sum(
            loss_inverse)  # [select_param_sets, 1]
        # select_params = x_k[:, index_k]  # [param_dim, select_param_sets]
        m_kp1 = torch.matmul(select_params, weights)  # m_(k+1): [param_dim, 1]
        mueff = 1 / torch.sum(weights**2)  # mu_w

        # print("select_param_sets: ", select_param_sets)
        # print("loss_inverse: ", loss_inverse)
        # print("weights: ", weights)
        # print("m_kp1: ", m_kp1)
        # print("mueff: ", mueff)
        # The evolution path p_sigma and p_c
        Lambda, V = torch.linalg.eigh(cov_k)  # eigen decomposition
        Lambda = torch.sqrt(Lambda)
        inv_sqrt_cov = torch.matmul(V, torch.matmul(torch.diag(Lambda**-1),
                                                    V.T))  # C^(-1/2)
        # print('Lambda: ', Lambda)
        # print('V: ', V)
        # print('inv_sqrt_cov: ', inv_sqrt_cov)
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
        # print("c_sigma: ", c_sigma)
        # print("c_c: ", c_c)
        # print("c_1: ", c_1)
        # print("c_mu: ", c_mu)
        # print("d_sigma: ", d_sigma)
        # print("expected_value: ", expected_value)

        p_sigma_kp1 = (1 - c_sigma) * p_sigma_k + torch.sqrt(
            c_sigma * (2 - c_sigma) * mueff) * torch.matmul(
                inv_sqrt_cov, (m_kp1 - m_k).unsqueeze(1) / sigma_k)
        indicator = (torch.linalg.norm(p_sigma_kp1) /
                     torch.sqrt(1 - (1 - c_sigma)**(2 * k)) / expected_value
                     < (1.4 + 2 / (self.param_dim + 1))) * 1
        p_c_kp1 = (1 - c_c) * p_c_k + indicator * torch.sqrt(
            c_c * (2 - c_c) * mueff) * (m_kp1 - m_k).unsqueeze(1) / sigma_k

        # print("p_sigma_kp1: ", p_sigma_kp1)
        # print("indicator: ", indicator)
        # print("p_c_kp1: ", p_c_kp1)

        # Adapting covariance matrix C
        artmp = (1 / sigma_k) * (select_params -
                                 torch.tile(m_k, [select_param_sets, 1]).T)
        cov_kp1 = (1 - c_1 - c_mu) * cov_k + c_1 * (torch.matmul(p_c_kp1, p_c_kp1.T) + (1 - indicator) * c_c * (2 - c_c) * cov_k) + \
            c_mu * torch.matmul(artmp, torch.matmul(torch.diag(weights), artmp.T))

        # print("artmp: ", artmp)
        # print("cov_kp1: ", cov_kp1)
        # return None
        # Adapting step size
        sigma_kp1 = sigma_k * torch.exp(
            (c_sigma / d_sigma) *
            (torch.linalg.norm(p_sigma_kp1) / expected_value - 1))
        # print("sigma_kp1: ", sigma_kp1)

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
            sampled_params: [3 * self.p + 1, param_sets]
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
            sampled_params[0:self.p,
                           i] = torch.matmul(self.pinv_concat_mat,
                                             sampled_parameters[0:N, i])
            sampled_params[self.p:2 * self.p,
                           i] = torch.matmul(self.pinv_concat_mat,
                                             sampled_parameters[N:2 * N, i])
            sampled_params[2 * self.p, i] = sampled_parameters[2 * N, i]
            sampled_params[2 * self.p + 1:,
                           i] = torch.matmul(self.pinv_concat_mat,
                                             sampled_parameters[2 * N + 1:, i])
        return sampled_params, sampled_parameters

    # TODO: Try uniform sampling at the first iteration

    def _sample_valid_parameters_pFIC(self, mean, cov, search_range):
        # * TODO: Try without parameterizing sigma
        # mean[2 * self.p + 1] = 0.005
        # mean[2 * self.p + 2:] = 0
        # * TODO: Try without parameterizing sigma
        multivariate_normal = td.MultivariateNormal(mean, cov)
        valid_count = 0
        total_count = 0
        # TODO: change multiplier to the self.param_sets to a smaller number (originally 20000)
        total_threshold = 20000 * self.param_sets
        sampled_params = torch.zeros(self.param_dim, self.param_sets)
        sampled_parameters = torch.zeros(
            self.parameter_dim, self.param_sets)  # [3*N+1, param_sets]
        while valid_count < self.param_sets:
            # print('current valid count: ', valid_count)
            sampled_params[:, valid_count] = multivariate_normal.sample(
            )  # [3 * self.p + 1, param_sets]
            # * TODO: Try without parameterizing sigma
            # sampled_params[2 * self.p + 1, valid_count] = 0.005
            # sampled_params[2 * self.p + 2:, valid_count] = 0
            # * TODO: Try without parameterizing sigma
            sampled_parameters[:, valid_count] = self.get_parameters(
                sampled_params[:, valid_count]).squeeze()

            if self.other_parameterization is not None:
                if (sampled_parameters[:, valid_count] < search_range[:, 0]
                    ).any() or (sampled_parameters[:, valid_count]
                                > search_range[:, 1]).any():
                    valid_count -= 1
            else:
                wEI = sampled_parameters[self.N:2 * self.N,
                                         valid_count].unsqueeze(1)
                wEI_myelin_corr = torch.squeeze(CBIG_corr(wEI, self.myelin))
                wEI_rsfc_corr = torch.squeeze(
                    CBIG_corr(wEI, self.rsfc_gradient))
                # print('wEE: ', sampled_parameters[0:self.N, valid_count])
                # print('wEI: ', wEI)
                # print('G: ', sampled_parameters[2 * self.N, valid_count])
                # print('sigma: ', sampled_parameters[2 * self.N + 1:,
                #                                     valid_count])

                if (sampled_parameters[:, valid_count] < search_range[:, 0]).any() \
                        or (sampled_parameters[:, valid_count] > search_range[:, 1]).any() or (wEI_myelin_corr > 0) or (wEI_rsfc_corr < 0):
                    # print("Out of search range!")

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
            wEI_rsfc_corr = torch.squeeze(CBIG_corr(wEI, self.rsfc_gradient))
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

    def get_best_train_params(self, top_k_for_each_epoch: int = 1):
        """
        Firstly, load the dict for each epoch under 'self.curr_phase_save_dir'.
        Afterwards, get the best param vector along with their costs for each train epoch.
        Then combine them into a dict with the same structure.
        Finally, save the dict as 'best_from_train.pth' under 'self.curr_phase_save_dir'
        """

        train_save_files = [
            get_train_file_path(self.curr_phase_save_dir, ep)
            for ep in range(self.num_epochs)
        ]

        # save the top few param vectors with the lowest validation loss
        best_from_train_file_path = get_best_params_file_path(
            self.phase, self.curr_phase_save_dir)
        best_from_train = combine_all_param_dicts(
            paths_to_dicts=train_save_files,
            top_k_per_dict=top_k_for_each_epoch,
            combined_dict_save_path=best_from_train_file_path)

        print(
            f"Successfully saved the top {top_k_for_each_epoch} parameters from each train epoch to: {best_from_train_file_path}"
        )

        return best_from_train


class CMAESValidator(ModelHandler):
    """
    The validator for Hybrid version CMA-ES
    """

    def __init__(self, config, emp_stats: dict[str, torch.Tensor],
                 prev_phase_best_params_path: str, curr_phase_save_dir: str):
        super().__init__(config, 'val', emp_stats, prev_phase_best_params_path,
                         curr_phase_save_dir)

    def validate(self, use_top_k=None, seed=None):
        print(f" -- Start {self.phase} phase -- ")
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)

        # load the best_from_prev_phase
        best_from_prev_phase = torch.load(self.prev_phase_best_params_path,
                                          map_location=self.device)

        # simulate on the current dataset
        param_vectors = best_from_prev_phase['parameter']
        if use_top_k is not None:
            param_vectors = param_vectors[:, :use_top_k]
        save_dict = self.sim_n_get_losses(param_vectors)

        save_dict['seed'] = seed
        # save the losses from previous phase
        valid_param_indices = save_dict['valid_param_indices']
        for key, value in best_from_prev_phase.items():
            if key.endswith('loss'):
                value = value[valid_param_indices]
                if key.startswith('train'):
                    save_dict[key] = value
                else:
                    save_dict[f"{self.prev_phase}_{key}"] = value

        # save the simulation results
        print(datetime.datetime.now(), 'Start saving results...')

        sim_on_curr_phase_file_path = get_best_params_file_path(
            self.phase, self.curr_phase_save_dir)
        torch.save(save_dict, sim_on_curr_phase_file_path)
        print("Successfully saved to:", sim_on_curr_phase_file_path)

        print(datetime.datetime.now(), f" -- Done {self.phase} phase -- ")
        return 0

    @DeprecationWarning
    def val_best_parameters(self,
                            sc_euler,
                            emp_fc,
                            emp_fcd_cum,
                            epoch,
                            seed=None):
        print(" -- Start validating -- ")
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 1000000000000)
        torch.manual_seed(seed)

        parameter_path = get_train_file_path(self.prev_phase_best_params_path,
                                             epoch)
        if not os.path.exists(parameter_path):
            raise Exception("Path doesn't exist.")
        d = torch.load(parameter_path, map_location=self.device)

        valid_param_indices_pre = d['valid_param_indices']
        parameter = d['parameter']
        parameter = parameter[:, valid_param_indices_pre]
        if 'FC_FCD_loss' in d:
            # TODO_old: Try without FCD KS loss
            # FC_FCD_loss is [param_sets, 3]
            total_loss = torch.sum(d['FC_FCD_loss'], dim=1)
            # total_loss = torch.sum(d['FC_FCD_loss'][:, :2], dim=1)
            # TODO_old: Try without FCD KS loss

            # TODO_old: Regularize firing rate
            # TODO_old: Try without any constraint on r_E
            # total_loss += d['r_E_reg_loss']  # r_E_reg_loss is [param_sets]
            # TODO_old: Try without any constraint on r_E
            # TODO_old: Regularize firing rate
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

        print(datetime.datetime.now(), 'start simulation'.upper())

        bold_signal, valid_M_mask, r_E_ave = mfm_model.CBIG_2014_mfm_simulation(
            simulate_time=self.simulate_time,
            burn_in_time=self.burn_in_time,
            TR=self.TR)
        # bold_signal: [ROIs, param_dup, t]; valid_M_mask: [param_dup]
        fc_this_param = None
        if valid_M_mask.any():
            bold_this_param = bold_signal[:,
                                          valid_M_mask, :]  # [N, 1/2/3/param_dup, #time_points_in_BOLD]
            fc_this_param = MfmModel2014.FC_calculate(bold_this_param)
            fc_this_param = torch.mean(fc_this_param, dim=0)
            fcd_mat_this_param, fcd_hist_this_param = MfmModel2014.FCD_calculate(
                bold_this_param, self.window_size)
            # You should not average across different FCD matrix
            fcd_hist_this_param = torch.mean(fcd_hist_this_param,
                                             dim=1).unsqueeze(1)  # [10000, 1]

        if fc_this_param is None:
            print(datetime.datetime.now(),
                  "This chosen parameter fails Euler.")
            return 1

        losses = MfmModel2014.calc_all_loss_from_fc_fcd(
            fc_this_param.unsqueeze(0), fcd_hist_this_param, emp_fc,
            emp_fcd_cum)  # [1]
        corr_loss, l1_loss, ks_loss = losses['corr_loss'], losses[
            'l1_loss'], losses['ks_loss']
        print(datetime.datetime.now(), 'Start saving results...')
        save_dict = {
            'parameter': parameter,
            'corr_loss': corr_loss,
            'l1_loss': l1_loss,
            'ks_loss': ks_loss,
            'seed': seed
        }
        # TODO_old: Save r_E
        if mfm_model.r_E and valid_M_mask.any():
            # save r_E_for_valid_params
            r_E_for_valid_params = r_E_ave[:, valid_M_mask]
            r_E_for_valid_params = torch.mean(r_E_for_valid_params,
                                              dim=1,
                                              keepdim=True)
            save_dict['r_E_for_valid_params'] = r_E_for_valid_params
            # TODO_old: Regularize firing rate
            # # get and save r_E_reg_loss
            # r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
            #                                 mfm_model.r_E,
            #                                 loss_type='L2')
            # save_dict['r_E_reg_loss'] = r_E_reg_loss
            # TODO_old: Regularize firing rate

        # TODO_old: Save r_E
        torch.save(
            save_dict,
            os.path.join(self.curr_phase_save_dir,
                         f'best_param{epoch}' + '.pth'))
        print("Successfully saved.")

        print(datetime.datetime.now(), " -- Done validating -- ")
        return 0


class CMAESTester(CMAESValidator):

    def __init__(self, config, emp_stats: dict[str, torch.Tensor],
                 prev_phase_best_params_path: str, curr_phase_save_dir: str):
        super().__init__(config, emp_stats, prev_phase_best_params_path,
                         curr_phase_save_dir)
        self.phase = 'test'
        print(self.post_init_message)

    def test(self, use_top_k=None, seed=None):
        if use_top_k is None:
            use_top_k = self.test_param_sets
        return self.validate(use_top_k=use_top_k, seed=seed)

    @DeprecationWarning
    def select_best_from_val(self):
        print(" -- Start testing -- ")
        train_num_epochs = 100
        val_save_dirs_len = len(self.prev_phase_best_params_path)
        parameter_sets = torch.zeros(
            self.parameters_dim, val_save_dirs_len *
            train_num_epochs)  # [205, 50 * num_of_tried_seeds]
        # TODO_old: Regularize firing rate
        num_of_losses = 4  # was 4 without r_E_reg_loss
        # TODO_old: Regularize firing rate
        val_loss_sets = torch.ones(val_save_dirs_len * train_num_epochs,
                                   num_of_losses) * 3
        # [total, corr, L1, ks]
        valid_val_dir_count = 0
        for val_dir_i in range(val_save_dirs_len):
            val_dir = self.prev_phase_best_params_path[val_dir_i]
            if not os.path.exists(val_dir):
                print(f"{val_dir} doesn't exist.")
                continue
            valid_val_dir_count += 1
            for epoch in range(train_num_epochs):
                param_val_path = os.path.join(val_dir,
                                              f'best_param{epoch}.pth')
                if not os.path.exists(param_val_path):
                    print(f"{param_val_path} doesn't exist.")
                    continue
                d = torch.load(param_val_path, map_location=self.device)
                parameter_sets[:, val_dir_i * train_num_epochs +
                               epoch] = torch.squeeze(d['parameter'])
                val_loss_sets[val_dir_i * train_num_epochs + epoch,
                              1] = d['corr_loss']
                val_loss_sets[val_dir_i * train_num_epochs + epoch,
                              2] = d['l1_loss']
                val_loss_sets[val_dir_i * train_num_epochs + epoch,
                              3] = d['ks_loss']
                # TODO_old: Regularize firing rate
                # # key for the r_E regularization loss is 'r_E_reg_loss'
                # if 'r_E_reg_loss' in d:
                #     # TODO_old: Try without any constraint on r_E, can change the following line to assign zero values
                #     val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                #                   4] = d['r_E_reg_loss']
                #     # val_loss_sets[val_dir_i * self.trained_epochs + epoch,
                #     #               4] = 0
                #     # TODO_old: Try without any constraint on r_E
                # TODO_old: Regularize firing rate
        if valid_val_dir_count == 0:
            print("No valid validated directories.")
            return 1
        # TODO_old: Try without FCD KS loss
        val_loss_sets[:, 0] = torch.sum(val_loss_sets[:, 1:], dim=1)
        # val_loss_sets[:, 0] = torch.sum(val_loss_sets[:, 1:3], dim=1)
        # TODO_old: Try without FCD KS loss
        # Record all param_coef and loss
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
        # TODO_old: Regularize firing rate
        # if all_loss.shape[1] > 4:
        #     save_dict['r_E_reg_loss'] = all_loss[:, 4]
        # TODO_old: Regularize firing rate
        torch.save(save_dict,
                   os.path.join(self.curr_phase_save_dir, 'val_results.pth'))
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
                    emp_fc=None,
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
        1, 2)  # [N, param_sets, param_dup, #time_points_in_BOLD]
    valid_M_mask = valid_M_mask.view(param_dup,
                                     param_sets).T  # [param_sets, param_dup]

    valid_param_indices = []  # record valid param index
    fc_sim = torch.zeros(param_sets, N, N)
    fcd_hist = torch.zeros(fcd_hist_bins, param_sets)
    count_valid = 0
    for i in range(param_sets):
        # for each set of parameter
        mask_this_param = valid_M_mask[i]  # [param_dup]
        if mask_this_param.any():
            valid_param_indices.append(i)
            bold_this_param = bold_signal[:, i,
                                          mask_this_param, :]  # [N, 1/2/3/param_dup, #time_points_in_BOLD]
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
    l1_loss = None
    ks_loss = None
    if emp_fc is not None and emp_fcd_cdf is not None:
        losses = MfmModel2014.calc_all_loss_from_fc_fcd(
            fc_sim, fcd_hist, emp_fc, emp_fcd_cdf)  # [param_dup]
        corr_loss, l1_loss, ks_loss = losses['corr_loss'], losses[
            'l1_loss'], losses['ks_loss']
    valid_param_indices = torch.as_tensor(valid_param_indices)

    print('Start saving results...')
    # save_dict = {'fc': fc_this_param, 'fcd': fcd_mat_this_param, 'corr_loss': corr_loss,
    #              'l1_loss': L1_loss, 'ks_loss': ks_loss, 'seed': seed, 'param_coef': param_coef,    'parameter': parameter}
    save_dict = {
        'valid_param_indices': valid_param_indices,
        'fc': fc_sim,
        'fcd_pdf': fcd_hist,
        'corr_loss': corr_loss,
        'l1_loss': l1_loss,
        'ks_loss': ks_loss,
        'seed': seed,
        'parameter': parameter
    }
    torch.save(save_dict, save_path)
    print("Successfully saved FC and FCD and losses.")

    print(" -- Done generating -- ")
    return 0


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
        1, 2)  # [N, param_sets, param_dup, #time_points_in_BOLD]
    valid_M_mask = valid_M_mask.view(param_dup,
                                     param_sets).T  # [param_sets, param_dup]

    valid_param_indices = []  # record valid param index
    fc_sim = torch.zeros(param_sets, param_dup, N, N)
    window_num = bold_signal.shape[3] - window_size + 1
    fcd_mats = torch.zeros(param_sets, param_dup, window_num, window_num)
    fcd_hist = torch.zeros(param_sets, fcd_hist_bins, param_dup)
    count_valid = 0
    for i in range(param_sets):
        # for each set of parameter
        mask_this_param = valid_M_mask[i]  # [param_dup]
        if mask_this_param.all():  # Must all be true
            valid_param_indices.append(i)
            bold_this_param = bold_signal[:, i,
                                          mask_this_param, :]  # [N, param_dup, #time_points_in_BOLD]
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

    valid_param_indices = torch.as_tensor(valid_param_indices)

    print('Start saving results...')
    # save_dict = {'fc': fc_this_param, 'fcd': fcd_mat_this_param, 'corr_loss': corr_loss,
    #              'l1_loss': L1_loss, 'ks_loss': ks_loss, 'seed': seed, 'param_coef': param_coef,    'parameter': parameter}
    save_dict = {
        'valid_param_indices': valid_param_indices,
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
                emp_fc,
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
                                      valid_M_mask, :]  # [N, 1/2/3/param_dup, #time_points_in_BOLD]
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

    losses = MfmModel2014.calc_all_loss_from_fc_fcd(fc_this_param.unsqueeze(0),
                                                    fcd_hist_this_param,
                                                    emp_fc, emp_fcd_cdf)  # [1]
    corr_loss, l1_loss, ks_loss = losses['corr_loss'], losses[
        'l1_loss'], losses['ks_loss']
    print('Start saving results...')

    save_dict = {
        'fc': fc_this_param,
        'corr_loss': corr_loss,
        'l1_loss': l1_loss,
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
    # bold_signal = bold_signal.view(roi_num, param_dup, param_sets, -1).transpose(1, 2)  # [N, param_sets, param_dup, #time_points_in_BOLD]
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
    # TODO_old: Save r_E
    if mfm_model.r_E and valid_M_mask.any():
        # save r_E_for_valid_params
        r_E_for_valid_params = r_E_ave[:, valid_M_mask]
        save_dict['r_E_for_valid_params'] = r_E_for_valid_params
        # TODO_old: Regularize firing rate
        # # get and save r_E_reg_loss
        # r_E_reg_loss = get_r_E_reg_loss(r_E_for_valid_params,
        #                                 mfm_model.r_E,
        #                                 loss_type='L2')
        # save_dict['r_E_reg_loss'] = r_E_reg_loss
        # TODO_old: Regularize firing rate
    # TODO_old: Save r_E
    torch.save(save_dict, save_path)
    print("Successfully saved EI ratio.")
    print(" -- Done simulating -- ")
    return 0
