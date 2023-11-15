import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from ..hyperpose.MSHyperPose import PoseRegressorHyper

"""
AtLoc: Attention Guided Camera Localization - AAAI 2020 (Oral)
This is the PyTorch implementation of AtLoc
Reference: https://github.com/BingCS/AtLoc
"""


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.g = nn.Linear(in_channels, in_channels // 8)
        self.theta = nn.Linear(in_channels, in_channels // 8)
        self.phi = nn.Linear(in_channels, in_channels // 8)

        self.W = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        g_x = self.g(x).view(batch_size, out_channels // 8, 1)

        theta_x = self.theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, out_channels // 8, 1)
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.view(batch_size, out_channels // 8)
        W_y = self.W(y)
        z = W_y + x
        return z


class AtLoc(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True, feat_dim=2048):
        super(AtLoc, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.att = AttentionBlock(feat_dim)
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 4)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)

        x = self.att(x.view(x.size(0), -1))

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), dim=1)


class HyperAtLoc(nn.Module):
    def __init__(self, config, droprate=0.5, pretrained=True, feat_dim=2048):
        super(HyperAtLoc, self).__init__()
        self.droprate = droprate
        self._backbone_dim = feat_dim

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.att = AttentionBlock(feat_dim)
        self.fc_xyz = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 4)

        # initialize
        if pretrained:
            init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
            init_modules = self.modules()

        # =========================================
        # Hyper networks
        # =========================================
        self.hyper_dim_t = config.get('hyper_dim_t')
        self.hyper_t_hidden_scale = config.get('hyper_t_hidden_scale')
        self.hyper_in_t_proj = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_t)
        self.hyper_in_t = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_t)
        self.hyper_in_t_fc_2 = nn.Linear(in_features=self.hyper_dim_t, out_features=self.hyper_dim_t)
        self.hypernet_t_fc_h2 = nn.Linear(self.hyper_dim_t, 3 * (self.hyper_dim_t + 1))

        self.hyper_dim_rot = config.get('hyper_dim_rot')
        self.hyper_in_rot_proj = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_rot)
        self.hyper_in_rot = nn.Linear(in_features=self._backbone_dim, out_features=self.hyper_dim_rot)
        self.hyper_in_rot_fc_2 = nn.Linear(in_features=self.hyper_dim_rot, out_features=self.hyper_dim_rot)
        self.hypernet_rot_fc_h2 = nn.Linear(self.hyper_dim_rot, 4 * (self.hyper_dim_rot + 1))

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hyper-networks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(self.hyper_dim_t, self.hyper_dim_t, 3, hidden_scale=1.0)
        self.regressor_hyper_rot = PoseRegressorHyper(self.hyper_dim_rot, self.hyper_dim_rot, 4, hidden_scale=1.0)

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def forward(self, x):
        ##################################################
        # Backbone Forward Pass
        ##################################################
        x = self.feature_extractor(x)
        x = F.relu(x)

        x = self.att(x.view(x.size(0), -1))


        ##################################################
        # Hyper-networks Forward Pass
        ##################################################
        t_input = self.hyper_in_t_proj(x)
        hyper_in_h2 = self._swish(self.hyper_in_t_fc_2(t_input))
        hyper_w_t_fc_h2 = self.hypernet_t_fc_h2(hyper_in_h2)

        rot_input = self.hyper_in_rot_proj(x)
        hyper_in_h2 = self._swish(self.hyper_in_rot_fc_2(rot_input))
        hyper_w_rot_fc_h2 = self.hypernet_rot_fc_h2(hyper_in_h2)

        self.w_t = {'w_o': hyper_w_t_fc_h2}
        self.w_rot = {'w_o': hyper_w_rot_fc_h2}

        ##################################################
        # Regression Forward Pass
        ##################################################
        # (1) Hyper-network's regressors
        p_x_hyper = self.regressor_hyper_t(self.hyper_in_t(x), self.w_t)
        p_q_hyper = self.regressor_hyper_rot(self.hyper_in_rot(x), self.w_rot)

        # (2) Trained regressors
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        ##################################################
        # Output
        ##################################################
        xyz = torch.add(xyz, p_x_hyper)
        wpqr = torch.add(wpqr, p_q_hyper)

        est_pose = torch.cat((xyz, wpqr), dim=1)
        return est_pose
