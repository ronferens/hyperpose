import torch
import torch.nn as nn
import torch.nn.functional as F
from .MSHyperPose import PoseRegressorHyper
import torchvision


class HyperPose(nn.Module):
    """
    An enhanced PoseNet implementation with Hyper-Network
    """
    def __init__(self, config, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(HyperPose, self).__init__()

        if config.get('backbone_type') == 'googlenet':
            self._backbone = torchvision.models.googlenet(pretrained=True)
            self._backbone_forward = self._backbone_forward_googlenet
            self._backbone_dim = 1000
        elif config.get('backbone_type') == 'efficientnet':
            # EfficientNet Backbone
            self._backbone = torch.load(backbone_path)
            self._backbone_forward = self._backbone_forward_efficientnet
            self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
            self._backbone_dim = 1280
        latent_dim = 1024

        # Regressor layers
        self.fc1 = nn.Linear(self._backbone_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 3)
        self.fc3 = nn.Linear(latent_dim, 4)
        self.dropout = nn.Dropout(p=0.1)

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

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hyper-networks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(self.hyper_dim_t, self.hyper_dim_t, 3, hidden_scale=1.0)
        self.regressor_hyper_rot = PoseRegressorHyper(self.hyper_dim_rot, self.hyper_dim_rot, 4, hidden_scale=1.0)

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def _backbone_forward_googlenet(self, samples):
        x = self._backbone(samples)
        return x

    def _backbone_forward_efficientnet(self, samples):
        x = self._backbone.extract_features(samples)
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, samples):
        """
        Forward pass
        :param samples: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        ##################################################
        # Backbone Forward Pass
        ##################################################
        x = self._backbone_forward(samples)

        x_embs = x.cpu().detach().numpy().flatten()

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
        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)

        ##################################################
        # Output
        ##################################################
        x_t = torch.add(p_x, p_x_hyper)
        x_rot = torch.add(p_q, p_q_hyper)

        est_pose = torch.cat((x_t, x_rot), dim=1)
        return est_pose, x_embs
