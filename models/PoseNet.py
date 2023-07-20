from .model_base import BasePoseLightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNet(BasePoseLightningModule):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self, config):
        super(PoseNet, self).__init__(config)

        # Efficient net
        self.backbone_dim = 1280
        self.latent_dim = 1024

        # Regressor layers
        self.fc1 = nn.Linear(self.backbone_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 3)
        self.fc3 = nn.Linear(self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self._backbone.extract_features(x)
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)
        return torch.cat((p_x, p_q), dim=1)

