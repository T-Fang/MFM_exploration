import torch.nn as nn
import torch
from src.utils.tzeng_class_torch import Bottleneck


class ClassifyNanModel_2(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        param_layers = []
        in_features_param = 205
        # param_hidden_dim = [230, 63, 72]
        param_hidden_dim = [88]
        param_n_layers = len(param_hidden_dim)
        for i in range(param_n_layers):
            out_features_param = param_hidden_dim[i]
            param_layers.append(
                nn.Linear(in_features_param, out_features_param))
            param_layers.append(nn.ReLU())
            param_layers.append(nn.BatchNorm1d(out_features_param))
            in_features_param = out_features_param

        sc_layers = []
        in_features_sc = 2278
        # sc_hidden_dim = [2349, 2855]
        sc_hidden_dim = [984]
        sc_n_layers = len(sc_hidden_dim)
        for i in range(sc_n_layers):
            out_features_sc = sc_hidden_dim[i]
            sc_layers.append(nn.Linear(in_features_sc, out_features_sc))
            sc_layers.append(nn.ReLU())
            sc_layers.append(nn.BatchNorm1d(out_features_sc))
            in_features_sc = out_features_sc

        # out_features_both = 974
        out_features_both = 111

        param_layers.append(nn.Linear(in_features_param, out_features_both))
        param_layers.append(nn.ReLU())
        param_layers.append(nn.BatchNorm1d(out_features_both))
        sc_layers.append(nn.Linear(in_features_sc, out_features_both))
        sc_layers.append(nn.ReLU())
        sc_layers.append(nn.BatchNorm1d(out_features_both))

        self.param_emb = nn.Sequential(*param_layers)
        self.sc_emb = nn.Sequential(*sc_layers)

        self.output_layer = nn.Sequential(nn.Linear(out_features_both, 1),
                                          nn.Sigmoid())

    def forward(self, parameter, sc_mat):
        param_emb = self.param_emb(parameter)
        sc_emb = self.sc_emb(sc_mat)
        output_score = self.output_layer(param_emb + sc_emb)
        return output_score


class ClassifyNanModel_Yan100(nn.Module):

    def __init__(self, n_roi=100) -> None:
        super().__init__()

        param_layers = []
        in_features_param = int(3 * n_roi + 1)
        param_hidden_dim = [284]
        param_n_layers = len(param_hidden_dim)
        for i in range(param_n_layers):
            out_features_param = param_hidden_dim[i]
            param_layers.append(
                nn.Linear(in_features_param, out_features_param))
            param_layers.append(nn.ReLU())
            param_layers.append(nn.BatchNorm1d(out_features_param))
            in_features_param = out_features_param

        sc_layers = []
        in_features_sc = int(n_roi * (n_roi - 1) / 2)
        sc_hidden_dim = [4737]
        sc_n_layers = len(sc_hidden_dim)
        for i in range(sc_n_layers):
            out_features_sc = sc_hidden_dim[i]
            sc_layers.append(nn.Linear(in_features_sc, out_features_sc))
            sc_layers.append(nn.ReLU())
            sc_layers.append(nn.BatchNorm1d(out_features_sc))
            in_features_sc = out_features_sc

        out_features_both = 63

        param_layers.append(nn.Linear(in_features_param, out_features_both))
        param_layers.append(nn.ReLU())
        param_layers.append(nn.BatchNorm1d(out_features_both))
        sc_layers.append(nn.Linear(in_features_sc, out_features_both))
        sc_layers.append(nn.ReLU())
        sc_layers.append(nn.BatchNorm1d(out_features_both))

        self.param_emb = nn.Sequential(*param_layers)
        self.sc_emb = nn.Sequential(*sc_layers)

        self.output_layer = nn.Sequential(nn.Linear(out_features_both, 1),
                                          nn.Sigmoid())

    def forward(self, parameter, sc_mat):
        param_emb = self.param_emb(parameter)
        sc_emb = self.sc_emb(sc_mat)
        output_score = self.output_layer(param_emb + sc_emb)
        return output_score


class CoefModel_2(nn.Module):
    """
    Follow the thoughts of previous Tian Fang's work.
    Input parameters and sc_matrix. Same in MFMDeepLearning
    """

    def __init__(self, param_dim=205):
        super(CoefModel_2, self).__init__()

        self.param_dim = param_dim

        param_hidden_dim = [88, 111]
        param_layers = []
        in_features = 205
        for i in range(len(param_hidden_dim)):
            out_features = param_hidden_dim[i]
            param_layers.append(nn.Linear(in_features, out_features))
            param_layers.append(nn.ReLU())
            param_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.param_emb = nn.Sequential(*param_layers)

        sc_hidden_dim = [984]
        in_features = 2278
        sc_layers = []
        for i in range(len(sc_hidden_dim)):
            out_features = sc_hidden_dim[i]
            sc_layers.append(nn.Linear(in_features, out_features))
            sc_layers.append(nn.ReLU())
            sc_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features

        sc_layers.append(nn.Linear(in_features, 111))
        sc_layers.append(nn.ReLU())
        sc_layers.append(nn.BatchNorm1d(111))
        self.sc_emb = nn.Sequential(*sc_layers)

        self.output_score = nn.Sequential(nn.Linear(111, 3), nn.Sigmoid())
        '''
        self.block1 = nn.Sequential(nn.Linear(param_dim, 100), nn.ReLU(), nn.BatchNorm1d(100))
        self.block2 = nn.Sequential(nn.Linear(2278, 100), nn.ReLU(), nn.BatchNorm1d(100))
                                    # nn.Linear(1000, 100), nn.ReLU(), nn.BatchNorm1d(100))
        '''

    def forward(self, parameter, sc_mat):
        """
        Forward
        :param parameter: [bs, 205]
        :param sc_mat: [bs, 67 + ... + 1] = [bs, 2278]
        :return:
        """
        param_emb = self.param_emb(parameter)
        sc_emb = self.sc_emb(sc_mat)

        output_score = self.output_score(param_emb + sc_emb)

        return output_score


class PredictLossModel_1(nn.Module):
    """
    Same in MFMDeepLearning model nbr 5
    Input parameter, SC, FC_emp, FCD_emp
    """

    def __init__(self, param_dim=205) -> None:
        super().__init__()
        self.param_dim = param_dim
        last_dim = 111

        # For parameter
        param_hidden_dim = [88, last_dim]
        param_layers = []
        in_features = 205
        for i in range(len(param_hidden_dim)):
            out_features = param_hidden_dim[i]
            param_layers.append(nn.Linear(in_features, out_features))
            param_layers.append(nn.ReLU())
            param_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.param_emb = nn.Sequential(*param_layers)

        # For SC matrix
        sc_hidden_dim = [984, last_dim]
        in_features = 2278
        sc_layers = []
        for i in range(len(sc_hidden_dim)):
            out_features = sc_hidden_dim[i]
            sc_layers.append(nn.Linear(in_features, out_features))
            sc_layers.append(nn.ReLU())
            sc_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.sc_emb = nn.Sequential(*sc_layers)

        # For empirical FC
        fc_hidden_dim = [899, last_dim]
        in_features = 2278
        fc_layers = []
        for i in range(len(fc_hidden_dim)):
            out_features = fc_hidden_dim[i]
            if i == 0:
                fc_layers.append(Bottleneck(in_features, out_features, 313))
            else:
                fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.fc_emb = nn.Sequential(*fc_layers)

        # For empirical FCD
        fcd_hidden_dim = [5835, last_dim]
        in_features = 10000
        fcd_layers = []
        for i in range(len(fcd_hidden_dim)):
            out_features = fcd_hidden_dim[i]
            if i == 0:
                fcd_layers.append(Bottleneck(in_features, out_features, 1334))
            else:
                fcd_layers.append(nn.Linear(in_features, out_features))
            fcd_layers.append(nn.ReLU())
            fcd_layers.append(nn.BatchNorm1d(out_features))
            '''
            if i == 0:
                fcd_layers.append(nn.Dropout(p=0.33))
            '''
            in_features = out_features
        self.fcd_emb = nn.Sequential(*fcd_layers)

        self.output_fc_score = nn.Sequential(nn.Linear(last_dim, 2),
                                             nn.Sigmoid())
        self.output_fcd_score = nn.Sequential(nn.Linear(last_dim, 1),
                                              nn.Sigmoid())

    def forward(self, parameter, sc_mat, emp_fc, emp_fcd):
        """
        Forward
        :param parameter: [bs, 205]
        :param sc_mat: [bs, 67 + ... + 1] = [bs, 2278]
        :param emp_fc: [bs, 2278]
        :param emp_fcd: [bs, 10000]
        :return:
        """
        param_emb = self.param_emb(parameter)
        sc_emb = self.sc_emb(sc_mat)
        input_emb = param_emb + sc_emb

        fc_emb = self.fc_emb(emp_fc)
        fcd_emb = self.fcd_emb(emp_fcd)

        corr_l1 = self.output_fc_score(input_emb + fc_emb)
        ks = self.output_fcd_score(input_emb + fcd_emb)

        return torch.cat((corr_l1, ks), dim=1)  # [xxx, 3]


class PredictLossModel_1_Yan100(nn.Module):
    """
    Same in MFMDeepLearning model nbr 5
    Input parameter, SC, FC_emp, FCD_emp
    """

    def __init__(self, n_roi, bins=10000) -> None:
        super().__init__()
        self.param_dim = 3 * n_roi + 1
        unify_dim = 63

        # For parameter
        param_hidden_dim = [284, unify_dim]
        param_layers = []
        in_features = self.param_dim
        for i in range(len(param_hidden_dim)):
            out_features = param_hidden_dim[i]
            param_layers.append(nn.Linear(in_features, out_features))
            param_layers.append(nn.ReLU())
            param_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.param_emb = nn.Sequential(*param_layers)

        # For SC matrix
        sc_hidden_dim = [4737, unify_dim]
        in_features = int(n_roi * (n_roi - 1) / 2)
        sc_layers = []
        for i in range(len(sc_hidden_dim)):
            out_features = sc_hidden_dim[i]
            sc_layers.append(nn.Linear(in_features, out_features))
            sc_layers.append(nn.ReLU())
            sc_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.sc_emb = nn.Sequential(*sc_layers)

        # For empirical FC
        fc_hidden_dim = [1877, unify_dim]
        in_features = int(n_roi * (n_roi - 1) / 2)
        fc_layers = []
        for i in range(len(fc_hidden_dim)):
            out_features = fc_hidden_dim[i]
            if i == 0:
                fc_layers.append(Bottleneck(in_features, out_features, 627))
            else:
                fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.BatchNorm1d(out_features))
            in_features = out_features
        self.fc_emb = nn.Sequential(*fc_layers)

        # For empirical FCD
        fcd_hidden_dim = [5820, unify_dim]
        in_features = bins

        fcd_layers = []
        for i in range(len(fcd_hidden_dim)):
            out_features = fcd_hidden_dim[i]
            if i == 0:
                fcd_layers.append(Bottleneck(in_features, out_features, 1998))
            else:
                fcd_layers.append(nn.Linear(in_features, out_features))
            fcd_layers.append(nn.ReLU())
            fcd_layers.append(nn.BatchNorm1d(out_features))
            '''
            if i == 0:
                fcd_layers.append(nn.Dropout(p=0.33))
            '''
            in_features = out_features
        self.fcd_emb = nn.Sequential(*fcd_layers)

        self.output_fc_score = nn.Sequential(nn.Linear(unify_dim, 2),
                                             nn.Sigmoid())
        self.output_fcd_score = nn.Sequential(nn.Linear(unify_dim, 1),
                                              nn.Sigmoid())

    def forward(self, parameter, sc_mat, emp_fc, emp_fcd):
        """
        Forward
        :param parameter: [bs, 205]
        :param sc_mat: [bs, 67 + ... + 1] = [bs, 2278]
        :param emp_fc: [bs, 2278]
        :param emp_fcd: [bs, 10000]
        :return:
        """
        param_emb = self.param_emb(parameter)
        sc_emb = self.sc_emb(sc_mat)
        input_emb = param_emb + sc_emb

        fc_emb = self.fc_emb(emp_fc)
        fcd_emb = self.fcd_emb(emp_fcd)

        corr_l1 = self.output_fc_score(input_emb + fc_emb)
        ks = self.output_fcd_score(input_emb + fcd_emb)

        return torch.cat((corr_l1, ks), dim=1)  # [xxx, 3]
