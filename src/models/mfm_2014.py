import math
import torch
import numpy as np
import datetime
import time
import configparser
import scipy.io as spio
from scipy.optimize import fsolve
from tqdm import tqdm
from src.utils.init_utils import set_torch_default
from src.basic.constants import DEFAULT_DTYPE


def torch_corr_3D(vec_3d):
    """
    Compute the correlation coefficient parallel for 3D vector. And the middle dimension must be 2, standing for X and Y.
    :param vec_3d: [M, 2, len]
    :return: The corr_coef for X and Y. corr_coef: [M,]
    """
    ex_ey = torch.mean(vec_3d, dim=-1)
    std = torch.std(vec_3d, dim=-1, unbiased=False)
    # unbiased is set to False because in torch.mean, the denominator is N. If unbiased is true, then the denominator is N-1, not matched.
    xy = vec_3d[:, 0, :] * vec_3d[:, 1, :]
    e_xy = torch.mean(xy, dim=-1)
    cov = e_xy - ex_ey[:, 0] * ex_ey[:, 1]
    corr = cov / (std[:, 0] * std[:, 1])
    return corr


class MfmModel2014:

    def __init__(self, config, parameter, sc_euler, dt):
        """
        Deco 2014
        :param parameter: (N*3+1)*M matrix.
                    N is the number of ROI
                    M is the number of candidate parameter sets.
                    Each column of matrix presents a parameter set, where:
                    parameter[0:N]: recurrent strength within excitatory population (wEE)
                    parameter[N:2*N]: connection strength from excitatory population to inhibitory population (wEI)
                    parameter[2*N]: Global constant G
                    parameter[2*N+1:3*N+1]: noise amplitude sigma
        :param sc_mat: N*N structural connectivity matrix, should be sc_euler
        :param dt: time interval
        """
        self.device, self.tensor_type = set_torch_default()

        # Model Adaptation
        self.is_w_IE_fixed = True

        self.N = (parameter.shape[0] - 1) // 3  # N = 68 ROIs
        self.M = parameter.shape[1]  # num of parameter sets
        self.parameter = parameter
        self.sc_euler = sc_euler
        self.dt = dt
        self.w_EE = parameter[0:self.N]
        self.w_EI = parameter[self.N:2 * self.N]
        self.G = parameter[2 * self.N]
        self.sigma = parameter[2 * self.N + 1:3 * self.N + 1]

        # Synaptic Dynamical Equations Constants
        synaptic_constants = config['Synaptic Dynamical Equations Constants']
        self.I_0 = float(synaptic_constants['I_0'])
        self.a_E = float(synaptic_constants['a_E'])
        self.b_E = float(synaptic_constants['b_E'])
        self.d_E = float(synaptic_constants['d_E'])
        self.tau_E = float(synaptic_constants['tau_E'])
        self.W_E = float(synaptic_constants['W_E'])
        self.a_I = float(synaptic_constants['a_I'])
        self.b_I = float(synaptic_constants['b_I'])
        self.d_I = float(synaptic_constants['d_I'])
        self.tau_I = float(synaptic_constants['tau_I'])
        self.W_I = float(synaptic_constants['W_I'])
        self.J_NMDA = float(synaptic_constants['J_NMDA'])
        self.gamma_kin = float(synaptic_constants['gamma_kin'])
        # self.r_E_min = float(synaptic_constants['r_E_min'])
        # self.r_E_max = float(synaptic_constants['r_E_max'])
        # self.S_E_ave = float(synaptic_constants['S_E_ave'])

        # self.r_E = self.S_E_ave / (self.tau_E * self.gamma_kin * (1 - self.S_E_ave))
        self.r_E = 3
        print(f"Excitatory firing rate will be around {self.r_E}.")
        self.r_E_min = self.r_E - 0.3
        self.r_E_max = self.r_E + 0.3

        # Hemodynamic Model Constants
        hemodynamic_constants = config['Hemodynamic Model Constants']
        self.V0 = float(
            hemodynamic_constants['V0'])  # resting blood volume fraction
        self.kappa = float(
            hemodynamic_constants['kappa'])  # [s^-1] rate of signal decay
        self.gamma_hemo = float(hemodynamic_constants['gamma_hemo']
                                )  # [s^-1] rate of flow-dependent elimination
        self.tau = float(
            hemodynamic_constants['tau'])  # [s] hemodynamic transit time
        self.alpha = float(hemodynamic_constants['alpha'])  # Grubb's exponent
        self.rho = float(
            hemodynamic_constants['rho'])  # resting oxygen extraction fraction
        self.B0 = float(
            hemodynamic_constants['B0'])  # magnetic field strength, T
        self.TE = float(hemodynamic_constants['TE'])  # TE echo time, s
        self.r0 = float(hemodynamic_constants['r0']
                        )  # the intravascular relaxation rate, Hz
        self.epsilon_hemo = float(
            hemodynamic_constants['epsilon_hemo']
        )  # the ratio between intravascular and extravascular MR signal
        self.k1 = 4.3 * 28.265 * self.B0 * self.TE * self.rho
        self.k2 = self.epsilon_hemo * self.r0 * self.TE * self.rho
        self.k3 = 1 - self.epsilon_hemo

        # if self.is_w_IE_fixed:
        # Calculating w_IE

        # Calculate S_E average
        S_E_ini = 0.1641205151
        self.S_E_ave, info_dict, ier, message = fsolve(self._solve_S_E_ave,
                                                       S_E_ini,
                                                       full_output=True)
        if ier == 0:
            print(message)
        self.S_E_ave = self.S_E_ave[0]  # convert 1 element array to float

        # Calculate I_E average
        I_E_ini = 0.3772259651
        I_E_ave, info_dict, ier, message = fsolve(self._solve_I_E_ave,
                                                  I_E_ini,
                                                  full_output=True)
        if ier == 0:
            print(message)
        I_E_ave = I_E_ave[0]

        # Calculate I_I fixed point
        I_I_ini = 0.296385800197336 * np.ones((self.N, self.M))
        I_I_ave = np.ones_like(I_I_ini)
        for m in range(self.M):
            I_I_func_args = self.S_E_ave, self.w_EI[:, m]
            I_I_ave[:, m], info_dict, ier, message = fsolve(self._solve_I_I,
                                                            I_I_ini[:, m],
                                                            args=I_I_func_args,
                                                            full_output=True)
            if ier == 0:
                print(message)

        S_I_ave = self.tau_I * (self.a_I * I_I_ave - self.b_I) / (
            1 - np.exp(-self.d_I * (self.a_I * I_I_ave - self.b_I)))
        S_I_ave = torch.as_tensor(S_I_ave).to(DEFAULT_DTYPE)
        self.w_IE = (
            self.W_E * self.I_0 + self.w_EE * self.J_NMDA * self.S_E_ave +
            self.G * self.J_NMDA * torch.matmul(
                self.sc_euler, self.S_E_ave * torch.ones(self.N, self.M)) -
            I_E_ave) / S_I_ave
        # w_IE: [N, M]

    def CBIG_2014_mfm_simulation(self,
                                 simulate_time,
                                 burn_in_time,
                                 TR,
                                 warm_up_t=5000,
                                 use_tqdm=False,
                                 need_EI=False):
        """
        For M sets of parameters. Do not store the whole process like S_E, S_I. Save memory.
        :param warm_up_t: The loop for warm up S_E and S_I
        :param burn_in_minute: The burn-in time (can be seen as the second warm up) in [min]
        :param simulate_time: total simulated time, in [min]
        :param t_bold: simulated time interval
        :param use_tqdm:
        :param need_EI: return S_E_ave and S_I_ave
        :return: BOLD: [N, M, t]; valid_M_mask: [M], a boolean mask to indicate whether this parameter is valid
        """

        N = self.N
        M = self.M
        dt = self.dt

        # Set time
        t_start = 0  # seconds
        burn_in_t = 60 * burn_in_time
        t_end = t_start + burn_in_t + 60 * simulate_time
        t_p = torch.arange(t_start, t_end + dt, dt)
        t_len = len(t_p)
        t_inter = int(round(TR / dt))

        # Set initial values
        r_E = torch.zeros(N, M)
        # v_noise = torch.randn((N, M, t_len, 2))
        z = torch.zeros(N, M)
        f = torch.ones(N, M)
        v_volume = torch.ones(N, M)
        q = torch.ones(N, M)
        bold = torch.zeros(N, M, t_len // t_inter + 1)
        count_bold = 0

        S_E = torch.ones(N, M) * self.S_E_ave
        S_I = torch.ones(N, M) * 0.1433408985
        # v_noise_ini = torch.randn((N, M, warm_up_t, 2))

        bold_time_start = time.time()
        print(datetime.datetime.now(), ": Start BOLD calculating...")
        if use_tqdm:
            warm_loop = tqdm(range(warm_up_t), position=0, leave=True)
            main_loop = tqdm(range(t_len), position=0, leave=True)
        else:
            warm_loop = range(warm_up_t)
            main_loop = range(t_len)

        for t1 in warm_loop:
            dSE_dt, dSI_dt, _ = self._synaptic_dynamical_equations_2014(
                S_E, S_I)
            S_E = S_E + dSE_dt * dt + self.sigma * torch.randn(
                N, M) * math.sqrt(dt)
            S_I = S_I + dSI_dt * dt + self.sigma * torch.randn(
                N, M) * math.sqrt(dt)
        # warm_loop.close()
        print(datetime.datetime.now(),
              ": End warming up and start main body...")

        S_E_ave = torch.zeros(N, M)
        S_I_ave = torch.zeros(N, M)
        r_E_ave = torch.zeros(N, M)

        # Start calculating
        for t in main_loop:
            # print t at the interval of 100
            # if t % 10000 == 0:
            #     print(datetime.datetime.now(), ": t = ", t, flush=True)
            dSE_dt, dSI_dt, r_E = self._synaptic_dynamical_equations_2014(
                S_E, S_I)
            S_E = S_E + dSE_dt * dt + self.sigma * torch.randn(
                N, M) * math.sqrt(dt)
            # here math.sqrt(dt) is to make noise equivalent under different time interval, see notes.
            S_I = S_I + dSI_dt * dt + self.sigma * torch.randn(
                N, M) * math.sqrt(dt)
            dz_dt, df_dt, dv_dt, dq_dt = self._hemodynamic_equations(
                S_E, z, f, v_volume, q)
            z = z + dz_dt * dt
            f = f + df_dt * dt
            v_volume = v_volume + dv_dt * dt
            q = q + dq_dt * dt

            S_E_ave = S_E_ave + S_E
            S_I_ave = S_I_ave + S_I
            r_E_ave = r_E_ave + r_E

            if (t + 2) % t_inter == 0:
                bold[:, :, count_bold] = 100 / self.rho * self.V0 * (
                    self.k1 * (1 - q) + self.k2 *
                    (1 - q / v_volume) + self.k3 * (1 - v_volume))
                # here 100 / rho comes from Xiaolu's code, just a nonsense multiplication :.:
                count_bold += 1
                '''
                if not use_tqdm:
                    print("Steps: [{}/{}]".format(t, t_len))
                '''
        bold[:, :,
             count_bold] = 100 / self.rho * self.V0 * (self.k1 *
                                                       (1 - q) + self.k2 *
                                                       (1 - q / v_volume) +
                                                       self.k3 *
                                                       (1 - v_volume))
        # end for loop
        burn_in_bold = int(burn_in_t / TR)

        S_E_ave = S_E_ave / t_len
        S_I_ave = S_I_ave / t_len
        r_E_ave = r_E_ave / t_len

        valid_M_mask = torch.ones(M, dtype=torch.bool)
        for i in range(self.M):
            if torch.isnan(r_E_ave[:, i]).any():
                print("r_E exploded!")
                valid_M_mask[i] = False
            # ** TODO: Try without r_E constraints
            # elif (r_E_ave[:, i]
            #       < self.r_E_min).any() or (r_E_ave[:, i]
            #                                 > self.r_E_max).any():
            #     bold[:, i, :] = float('nan')
            #     valid_M_mask[i] = False
            # ** TODO: Try without r_E constraints
            elif torch.isnan(bold[:, i, :]).any():
                valid_M_mask[i] = False

        bold_elapsed = time.time() - bold_time_start
        print('Time using for calculating BOLD signals cost: ', bold_elapsed)
        print("BOLD shape with burn-in: ", bold.shape)

        if need_EI:
            return bold[:, :, burn_in_bold +
                        1:], valid_M_mask, r_E_ave, S_E_ave, S_I_ave
        else:
            return bold[:, :, burn_in_bold + 1:], valid_M_mask, r_E_ave

    @staticmethod
    def calc_all_loss_from_fc_fcd(fc_sim, fcd_hist, emp_fc, emp_fcd_cum):
        """
        Calculate corr_loss, l1_loss and KS_loss from simulated FC matrix and FCD histogram
        :param fc_sim: [M, N, N]
        :param fcd_hist: [10000, M], no need to cumsum or normalize. Will do automatically in ks_loss
        :param emp_fc: [N, N]
        :param emp_fcd_cum: [10000, 1]
        :return: [M]
        """
        losses = MfmModel2014.calc_FC_losses(fc_sim, emp_fc)
        losses.update(MfmModel2014.calc_FCD_losses(fcd_hist, emp_fcd_cum))
        return losses

    @staticmethod
    def FC_calculate(bold):
        """
        Calculate FC matrix for M sets of BOLD signals
        :param bold: [N, M, t_len]
        :return: FC matrix [M, N, N]
        """
        N = bold.shape[0]
        M = bold.shape[1]
        fc_mat = torch.zeros((M, N, N))
        for i in range(M):
            fc_mat[i] = torch.corrcoef(bold[:, i, :])
        return fc_mat

    @staticmethod
    def calc_FC_losses(fc_sim, emp_fc):
        """
        Compute the FC correlation and L1 cost for all M sets.
        [ATTENTION] here L1 is not the real L1 (subtraction of values and get the mean), but the subtraction of two mean values.
        :param fc_sim: [M, N, N]
        :param emp_fc: [N, N]
        :return: corr: [M,]; L1: [M,]
        """
        M = fc_sim.shape[0]
        N = fc_sim.shape[1]

        # Extract the upper triangular part
        mask = torch.ones(N, N, dtype=torch.bool)
        mask = torch.triu(mask, 1)
        vec_emp = emp_fc[mask]
        vec_emp = vec_emp.unsqueeze(0).expand(M, -1)  # [M, len]
        vec_sim = fc_sim[:, mask]

        # ** TODO: experiment with different l1 cost

        # L1 version 1: abs(mean)
        old_l1_loss = torch.abs(
            torch.mean(vec_emp, dim=1) - torch.mean(vec_sim, dim=1))

        # L1 version 2: mean(abs) or MAE
        MAE_l1_loss = torch.mean(torch.abs(vec_emp - vec_sim), dim=1)

        l1_loss = old_l1_loss
        # l1_loss = MAE_l1_loss
        # ** TODO: experiment with different l1 cost
        # L2 or MSE
        # l1_loss = torch.mean(torch.square(vec_emp - vec_sim), dim=1)

        # sqrt(L2) or RMSE
        # l1_loss = torch.sqrt(torch.mean(torch.square(vec_emp - vec_sim), dim=1))

        vec_3d = torch.zeros(M, 2, vec_emp.shape[1])
        vec_3d[:, 0, :] = vec_sim
        vec_3d[:, 1, :] = vec_emp

        corr = torch_corr_3D(vec_3d)
        corr_loss = 1 - corr

        losses = {
            'corr_loss': corr_loss,
            'l1_loss': l1_loss,
            'old_l1_loss': old_l1_loss,
            'MAE_l1_loss': MAE_l1_loss
        }
        return losses

    @staticmethod
    def FC_correlation_single(fc_1, fc_2):
        """
        Compute the correlation between two FC matrix. By flatten their upright part to two vectors and use Pearson's correlation
        :param fc_1: [N, N]
        :param fc_2: [N, N]
        :return:
        """
        N = fc_1.shape[0]
        mask = np.ones((N, N)).astype(bool)
        mask = np.triu(mask, 1)
        vec = np.zeros((2, (N * N - N) // 2))
        vec[0] = fc_1[mask]
        vec[1] = fc_2[mask]
        cor_fc = np.corrcoef(vec)
        return cor_fc[0, 1]

    @staticmethod
    def FCD_calculate(bold, window_size=83, bins=10000, get_FCD_matrix=False):
        """
        Moving windows and calculating the FC matrix for every window_size, then calculating the correlation between these FC matrix.
        :param bold: [N, M, t_len]
        :param window_size:
        :param bins: The histogram bins
        :param get_FCD_matrix: whether return the FCD matrix
        :return: FCD matrix [M, window_num, window_num]. window_num = t_len - window_size + 1
                FCD histogram: [10000, M]
        """
        set_torch_default()
        # start_time = time.time()
        N = bold.shape[0]
        M = bold.shape[1]
        t_len = bold.shape[2]
        window_num = t_len - window_size + 1
        if t_len < window_size:
            raise Exception(
                "The length of bold signal is shorter than the window size!")
        fc_list = torch.zeros(M, window_num, N, N)
        for t in range(0, window_num):
            bold_single = bold[:, :, t:t + window_size]
            fc_list[:, t, :, :] = MfmModel2014.FC_calculate(bold_single)
        fcd_mat = torch.zeros(M, window_num, window_num)
        # fc_hist = torch.zeros()
        # calculate the up triangle
        fc_mask = torch.ones(N, N, dtype=torch.bool)
        fc_mask = torch.triu(fc_mask, 1)
        '''
        vec_len = (N * N - N) // 2
        for i in range(0, window_num - 1):
            for j in range(i + 1, window_num):
                vec = torch.zeros(M, 2, vec_len)
                vec[:, 0] = fc_list[:, i, fc_mask]
                vec[:, 1] = fc_list[:, j, fc_mask]
                fcd_mat[:, i, j] = torch_corr_3D(vec)
        '''
        for m in range(M):
            fcd_mat[m] = torch.corrcoef(fc_list[m, :, fc_mask])
        # Calculate the FCD histogram
        fcd_mask = torch.ones(window_num, window_num, dtype=torch.bool)
        fcd_mask = torch.triu(fcd_mask, 1)
        fcd_vec = fcd_mat[:, fcd_mask]
        fcd_hist = torch.ones(bins, M)

        for hist_i in range(M):
            fcd_hist[:, hist_i] = torch.histc(fcd_vec[hist_i],
                                              bins=bins,
                                              min=-1.,
                                              max=1.)

        # if get_FCD_matrix:
        #     # Calculate whole FCD matrix
        #     fcd_mat = fcd_mat + fcd_mat.transpose(
        #         1, 2) + torch.eye(window_num).expand(M, -1, -1)

        # time_elapse = time.time() - start_time
        # print("The time for FCD matrix and FCD histogram is: ", time_elapse)
        return fcd_mat, fcd_hist

    @staticmethod
    def calc_FCD_losses(sim_fcd_hist, emp_fcd_cum):
        """
        Calculate the KS cost between simulated FCD matrix and empirical FCD matrix
        :param sim_fcd_hist: (10000, M)
        :param emp_fcd_cum: (10000, 1). Has been done cumulative summation and normalization (divided by emp_fcd_cum[-1:, :])
        :return: a loss dict containing the ks_loss: (M, )
        """
        M = sim_fcd_hist.shape[1]
        sim_fcd_cum = torch.cumsum(sim_fcd_hist, dim=0)
        sim_fcd_cum = sim_fcd_cum / sim_fcd_cum[-1:, :]
        emp_fcd_cum_expand = emp_fcd_cum.expand(-1, M)
        ks_dif = torch.abs(sim_fcd_cum - emp_fcd_cum_expand)
        ks_loss = torch.max(ks_dif, dim=0)[0]
        losses = {'ks_loss': ks_loss}
        return losses

    def _synaptic_dynamical_equations_2014(self, S_E_t, S_I_t):
        """
        The equations of Deco 2014 model
        :param S_E_t: [N, M]
        :param S_I_t: [N, M]
        :return: dS^E/dt, dS^I/dt
        """
        # Equations
        '''
        if self.is_w_IE_fixed:
            w_IE = self.w_IE
        else:
            w_IE = 0
        '''
        I_E_t = self.W_E * self.I_0 + self.w_EE * self.J_NMDA * S_E_t + self.G * self.J_NMDA * torch.matmul(
            self.sc_euler, S_E_t) - self.w_IE * S_I_t
        I_I_t = self.W_I * self.I_0 + self.w_EI * self.J_NMDA * S_E_t - S_I_t
        r_E = (self.a_E * I_E_t -
               self.b_E) / (1 - torch.exp(-self.d_E *
                                          (self.a_E * I_E_t - self.b_E)))
        r_I = (self.a_I * I_I_t -
               self.b_I) / (1 - torch.exp(-self.d_I *
                                          (self.a_I * I_I_t - self.b_I)))
        dSE_dt = -S_E_t / self.tau_E + (1 - S_E_t) * self.gamma_kin * r_E
        dSI_dt = -S_I_t / self.tau_I + r_I
        return dSE_dt, dSI_dt, r_E

    def _hemodynamic_equations(self, S_E_t, z_t, f_t, v_t, q_t):
        dz_dt = S_E_t - self.kappa * z_t - self.gamma_hemo * (f_t - 1)
        df_dt = z_t
        dv_dt = (f_t - v_t**(1 / self.alpha)) / self.tau
        dq_dt = (f_t / self.rho * (1 - (1 - self.rho)**(1 / f_t)) -
                 q_t * v_t**(1 / self.alpha - 1)) / self.tau
        return dz_dt, df_dt, dv_dt, dq_dt

    def _solve_I_I(self, I_I_ave, S_E_ave, w_EI_m):
        if torch.cuda.is_available():
            w_EI_m = w_EI_m.cpu().numpy()
        phi_I_I_ave = (self.a_I * I_I_ave - self.b_I) / (
            1 - np.exp(-self.d_I * (self.a_I * I_I_ave - self.b_I)))
        res = self.W_I * self.I_0 + self.J_NMDA * w_EI_m * S_E_ave - phi_I_I_ave * self.tau_I - I_I_ave
        return res

    def _solve_S_E_ave(self, S_E_ave):
        res = S_E_ave / (self.tau_E * self.gamma_kin *
                         (1 - S_E_ave)) - self.r_E
        return res

    def _solve_I_E_ave(self, I_E_ave):
        tmp = self.a_E * I_E_ave - self.b_E
        res = tmp / (1 - np.exp(-self.d_E * tmp)) - self.r_E
        return res


if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read(
        '/home/ftian/storage/projects/MFM_exploration/configs/model/config_pnc.ini'
    )
    parameter = torch.rand(205, 10)
    print("Parameter's shape: ", parameter.shape)

    # Get sc matrix
    group_mats = spio.loadmat(
        '/home/tzeng/storage/Matlab/HCPS1200/matfiles/all_mats_1029/group_all.mat'
    )
    sc_mat = torch.as_tensor(group_mats['sc_group_1029'])
    sc_euler = sc_mat / torch.max(
        sc_mat) * 0.02  # [68, 68] for Euler integration

    # Start
    mfm_model = MfmModel2014(config, parameter, sc_euler, dt=0.006)
    bold, valid_M_mask, _ = mfm_model.CBIG_2014_mfm_simulation(
        simulate_time=0.1, burn_in_time=0.1, use_tqdm=True)

    print("bold shape: ", bold.shape)  # [N, M, t_len]
    print("valid M mask: ", valid_M_mask)
    if valid_M_mask.any():
        print("Good!")
    else:
        print("None!")
