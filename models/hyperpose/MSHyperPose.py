import torch
from torch import nn
import torch.nn.functional as F
from .MSTransPoseNet import MSTransPoseNet, PoseRegressor
from .PoseRegressorHyper import PoseRegressorHyper


class MSHyperPose(MSTransPoseNet):
    """
        The Multi-Scene HyperPose model

        The model consist of a backbone feature extractor module followed by a dual-branch Transformer Encoders.
        With the addition of a dedicated hypernet at each branch the input query image's position and orientation is
        regressed.
        """

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__(config, pretrained_path)

        decoder_dim = self.transformer_t.d_model

        # =========================================
        # Hyper networks
        # =========================================
        self.hyper_dim_t = config.get('hyper_dim_t')
        self.hyper_t_hidden_scale = config.get('hyper_t_hidden_scale')
        self.hyper_in_t_proj = nn.Linear(in_features=1280, out_features=decoder_dim)
        self.hyper_in_t_fc_0 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_t_fc_1 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_t_fc_2 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hypernet_t_fc_h0 = nn.Linear(decoder_dim, self.hyper_dim_t * (decoder_dim + 1))
        self.hypernet_t_fc_h1 = nn.Linear(decoder_dim,
                                          int(self.hyper_dim_t * self.hyper_t_hidden_scale) * (self.hyper_dim_t + 1))
        self.hypernet_t_fc_h2 = nn.Linear(decoder_dim, 3 * (int(self.hyper_dim_t * self.hyper_t_hidden_scale) + 1))

        self.hyper_dim_rot = config.get('hyper_dim_rot')
        self.hyper_in_rot_proj = nn.Linear(in_features=1280, out_features=decoder_dim)
        self.hyper_in_rot_fc_0 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_rot_fc_1 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hyper_in_rot_fc_2 = nn.Linear(in_features=decoder_dim, out_features=decoder_dim)
        self.hypernet_rot_fc_h0 = nn.Linear(decoder_dim, self.hyper_dim_rot * (decoder_dim + 1))
        self.hypernet_rot_fc_h1 = nn.Linear(decoder_dim, self.hyper_dim_rot * (self.hyper_dim_rot + 1))
        self.hypernet_rot_fc_h2 = nn.Linear(decoder_dim, 4 * (self.hyper_dim_rot + 1))

        # =========================================
        # Regressor Heads
        # =========================================
        # (1) Hyper-networks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(decoder_dim, self.hyper_dim_t, 3,
                                                    hidden_scale=self.hyper_t_hidden_scale)
        self.regressor_hyper_rot = PoseRegressorHyper(decoder_dim, self.hyper_dim_rot, 4, hidden_scale=1.0)

        # (2) Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

        self.w_t, self.w_rot = None, None

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    def forward_heads(self):
        """
        Forward pass of the MLP heads
        The forward pass excepts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """

        ##################################################
        # Hyper-networks Forward Pass
        ##################################################
        t_input = torch.add(self._global_desc_t, self.hyper_in_t_proj(self._embeds))
        hyper_in_h0 = self._swish(self.hyper_in_t_fc_0(t_input))
        hyper_w_t_fc_h0 = self.hypernet_t_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_t_fc_1(t_input))
        hyper_w_t_fc_h1 = self.hypernet_t_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_t_fc_2(t_input))
        hyper_w_t_fc_h2 = self.hypernet_t_fc_h2(hyper_in_h2)

        rot_input = torch.add(self._global_desc_rot, self.hyper_in_rot_proj(self._embeds))
        hyper_in_h0 = self._swish(self.hyper_in_rot_fc_0(rot_input))
        hyper_w_rot_fc_h0 = self.hypernet_rot_fc_h0(hyper_in_h0)
        hyper_in_h1 = self._swish(self.hyper_in_rot_fc_1(rot_input))
        hyper_w_rot_fc_h1 = self.hypernet_rot_fc_h1(hyper_in_h1)
        hyper_in_h2 = self._swish(self.hyper_in_rot_fc_2(rot_input))
        hyper_w_rot_fc_h2 = self.hypernet_rot_fc_h2(hyper_in_h2)

        self.w_t = {'w_h1': hyper_w_t_fc_h0, 'w_h2': hyper_w_t_fc_h1, 'w_o': hyper_w_t_fc_h2}
        self.w_rot = {'w_h1': hyper_w_rot_fc_h0, 'w_h2': hyper_w_rot_fc_h1, 'w_o': hyper_w_rot_fc_h2}

        ##################################################
        # Regression Forward Pass
        ##################################################
        # (1) Hyper-network's regressors
        x_hyper_t = self.regressor_hyper_t(self._global_desc_t, self.w_t)
        x_hyper_rot = self.regressor_hyper_rot(self._global_desc_rot, self.w_rot)

        # (2) Trained regressors
        x_t = self.regressor_head_t(self._global_desc_t)
        x_rot = self.regressor_head_rot(self._global_desc_rot)

        ##################################################
        # Output
        ##################################################
        x_t = torch.add(x_t, x_hyper_t)
        x_rot = torch.add(x_rot, x_hyper_rot)

        est_pose = torch.cat((x_t, x_rot), dim=1)
        return est_pose, self._scene_log_distr
