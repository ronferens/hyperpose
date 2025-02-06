import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class PoseNet(nn.Module):
    """
    A class to represent a classic pose regressor (PoseNet)
    PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization,
    Kendall et al., 2015
    """
    def __init__(self):
        """
        Constructor
        """
        super(PoseNet, self).__init__()

        # GoogleNet Backbone
        self.backbone = torchvision.models.googlenet(pretrained=True)
        self.backbone_dim = 1000
        self.latent_dim = 1024

        self._init_regression_head()

    def _init_regression_head(self):
        # Regressor layers
        self.fc1 = nn.Linear(self.backbone_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.latent_dim, 3)
        self.fc3 = nn.Linear(self.latent_dim, 4)

        self.dropout = nn.Dropout(p=0.1)

        # Initialize FC layers
        for m in list(self.modules()):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward_backbone(self, samples):
        x = self.backbone(samples)
        return x

    def forward(self, samples):
        """
        Forward pass
        :param samples: (torch.Tensor) dictionary with key-value 'img' -- input image (N X C X H X W)
        :return: (torch.Tensor) dictionary with key-value 'pose' -- 7-dimensional absolute pose for (N X 7)
        """
        x = self.forward_backbone(samples)

        x_embs = x.cpu().detach().numpy().flatten()

        x = self.dropout(F.relu(self.fc1(x)))
        p_x = self.fc2(x)
        p_q = self.fc3(x)

        est_pose = torch.cat((p_x, p_q), dim=1)
        return est_pose, x_embs


class EffPoseNet(PoseNet):
    """
    A class to represent a classic pose regressor (PoseNet) with an efficient-net backbone
    """
    def __init__(self, backbone_path):
        """
        Constructor
        :param backbone_path: backbone path to a resnet backbone
        """
        super(EffPoseNet, self).__init__()

        # EfficientNet Backbone
        self.backbone = torch.load(backbone_path)
        self.backbone_dim = 1280

        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

        self._init_regression_head()

    def forward_backbone(self, samples):
        x = self.backbone.extract_features(samples)
        x = self.avg_pooling_2d(x)
        x = x.flatten(start_dim=1)
        return x
