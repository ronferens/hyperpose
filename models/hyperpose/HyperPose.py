"""
HyperPose model
"""
import torch
import torch.nn.functional as F
from torch import nn

import util.utils
from .transformer_encoder import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone


class HyperPose(nn.Module):

    def __init__(self, config, pretrained_path):
        """
        config: (dict) configuration of the model
        pretrained_path: (str) path to the pretrained backbone
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True

        # =========================================
        # CNN Backbone
        # =========================================
        self.backbone = build_backbone(config)

        # =========================================
        # Transformers
        # =========================================
        # Position (t) and orientation (rot) encoders
        self.transformer_t = Transformer(config)
        self.transformer_rot = Transformer(config)

        decoder_dim = self.transformer_t.d_model

        # The learned pose token for position (t) and orientation (rot)
        self.pose_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.pose_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        # The projection of the activation map before going into the Transformer's encoder
        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        # Whether to use prior from the position for the orientation
        self.use_quat_mul = config.get("use_quat_mul")

        # =========================================
        # Hypernetwork
        # =========================================
        self.hyper_dim_t = config.get('hyper_dim_t')
        self.hypernet_t = Transformer(config)
        self.hypernet_t_fc_h1 = nn.Linear(decoder_dim, self.hyper_dim_t * (decoder_dim + 1))
        self.hypernet_t_fc_h2 = nn.Linear(decoder_dim, (self.hyper_dim_t // 2) * (self.hyper_dim_t + 1))
        self.hypernet_t_fc_o = nn.Linear(decoder_dim, 3 * ((self.hyper_dim_t // 2) + 1))

        self.hyper_dim_rot = config.get('hyper_dim_rot')
        self.hypernet_rot = Transformer(config)
        self.hypernet_rot_fc_h1 = nn.Linear(decoder_dim, self.hyper_dim_rot * (decoder_dim + 1))
        self.hypernet_rot_fc_h2 = nn.Linear(decoder_dim, self.hyper_dim_rot * (self.hyper_dim_rot + 1))
        self.hypernet_rot_fc_o = nn.Linear(decoder_dim, 4 * (self.hyper_dim_rot + 1))

        # The learned pose token for position (t) and orientation (rot)
        self.hypernet_token_embed_t = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.hypernet_token_embed_rot = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        # The projection of the activation map before going into the Transformer's encoder
        self.hypernet_input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.hypernet_input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        self.w_t = None
        self.w_rot = None

        # =========================================
        # Regressors
        # =========================================
        # (1) Hypernetworks' regressors for position (t) and orientation (rot)
        self.regressor_hyper_t = PoseRegressorHyper(decoder_dim, self.hyper_dim_t, 3, hidden_scale=0.5)
        self.regressor_hyper_rot = PoseRegressorHyper(decoder_dim, self.hyper_dim_rot, 4, hidden_scale=1.0)

        # (2) Regressors for position (t) and orientation (rot)
        self.regressor_head_t = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

    @staticmethod
    def _swish(x):
        return x * F.sigmoid(x)

    @staticmethod
    def quat_mul(q, r):
        """
        Multiply quaternion(s) q with quaternion(s) r.
        Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
        Returns q*r as a tensor of shape (*, 4).
        reference: https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
        """
        assert q.shape[-1] == 4
        assert r.shape[-1] == 4

        original_shape = q.shape

        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape)

    def forward_transformers(self, data):
        """
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
        """
        samples = data.get('img')

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        features, pos, representation = self.backbone(samples)

        src_t, mask_t = features[0].decompose()
        src_rot, mask_rot = features[1].decompose()

        # Run through the transformer to translate to "camera-pose" language
        assert mask_t is not None
        assert mask_rot is not None
        local_descs_t = self.transformer_t(self.input_proj_t(src_t), mask_t, pos[0], self.pose_token_embed_t)
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, pos[1],
                                               self.pose_token_embed_rot)

        # Take the global desc from the pose token
        global_desc_t = local_descs_t[:, 0, :]
        global_desc_rot = local_descs_rot[:, 0, :]

        return {'global_desc_t': global_desc_t,
                'global_desc_rot': global_desc_rot,
                'src_t': src_t,
                'mask_t': mask_t,
                'src_rot': src_rot,
                'mask_rot': mask_rot,
                'pos_0': pos[0],
                'pos_1': pos[1]
                }

    def forward_heads(self, transformers_res):
        """
        The forward pass excepts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        returns: dictionary with key-value 'pose'--expected pose (NX7)
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')
        src_t = transformers_res.get('src_t')
        mask_t = transformers_res.get('mask_t')
        src_rot = transformers_res.get('src_rot')
        mask_rot = transformers_res.get('mask_rot')
        pos_0 = transformers_res.get('pos_0')
        pos_1 = transformers_res.get('pos_1')

        ##################################################
        # Hypernet
        ##################################################
        local_t_res = self.hypernet_t(self.hypernet_input_proj_t(src_t), mask_t, pos_0,
                                      self.hypernet_token_embed_t)
        global_hyper_t = local_t_res[:, 0, :]

        local_rot_res = self.hypernet_rot(self.hypernet_input_proj_rot(src_rot), mask_rot, pos_1,
                                          self.hypernet_token_embed_rot)
        global_hyper_rot = local_rot_res[:, 0, :]

        w_t_h1 = self._swish(self.hypernet_t_fc_h1(global_hyper_t))
        w_t_h2 = self._swish(self.hypernet_t_fc_h2(global_hyper_t))
        w_t_o = self._swish(self.hypernet_t_fc_o(global_hyper_t))

        w_rot_h1 = self._swish(self.hypernet_rot_fc_h1(global_hyper_rot))
        w_rot_h2 = self._swish(self.hypernet_rot_fc_h2(global_hyper_rot))
        w_rot_o = self._swish(self.hypernet_rot_fc_o(global_hyper_rot))

        self.w_t = {'w_h1': w_t_h1, 'w_h2': w_t_h2, 'w_o': w_t_o}
        self.w_rot = {'w_h1': w_rot_h1, 'w_h2': w_rot_h2, 'w_o': w_rot_o}

        ##################################################
        # Regression
        ##################################################
        # (1) Hypernetwork's regressors
        x_hyper_t = self.regressor_hyper_t(global_desc_t, self.w_t)
        x_hyper_rot = self.regressor_hyper_rot(global_desc_rot, self.w_rot)

        # (2) Trained regressors
        x_t = self.regressor_head_t(global_desc_t)
        x_rot = self.regressor_head_rot(global_desc_rot)

        ##################################################
        # Output
        ##################################################
        x_t = torch.add(x_t, x_hyper_t)
        if self.use_quat_mul:
            x_rot = self.quat_mul(x_rot, x_hyper_rot)
        else:
            x_rot = torch.add(x_rot, x_hyper_rot)
        expected_pose = torch.cat((x_t, x_rot), dim=1)
        return {'pose': expected_pose, 'w_t': self.w_t['w_o'], 'w_rot': self.w_rot['w_o']}

    def forward(self, data):
        """ The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED

            returns dictionary with key-value 'pose'--expected pose (NX7)
        """
        transformers_encoders_res = self.forward_transformers(data)

        # Regress the pose from the image descriptors
        heads_res = self.forward_heads(transformers_encoders_res)
        return heads_res


class PoseRegressorHyper(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, hidden_dim, output_dim, hidden_scale=1.0):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        hidden_scale: (float) Ratio between the input and the hidden layers' dimensions
        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_scale = hidden_scale

    @staticmethod
    def batched_linear_layer(x, wb):
        # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
        one = torch.ones(*x.shape[:-1], 1, device=x.device)
        linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
        return linear_res.squeeze(1)

    def _swish(self, x):
        return x * F.sigmoid(x)

    def forward(self, x, weights):
        """
        Forward pass
        """
        x = self._swish(self.batched_linear_layer(x, weights.get('w_h1').view(weights.get('w_h1').shape[0],
                                                                              (self.decoder_dim + 1),
                                                                              self.hidden_dim)))
        for index in range(len(weights.keys()) - 2):
            x = self._swish(self.batched_linear_layer(x,
                                                      weights.get(f'w_h{index + 2}').view(
                                                          weights.get(f'w_h{index + 2}').shape[0],
                                                          (self.hidden_dim + 1),
                                                          (int(self.hidden_dim * self.hidden_scale)))))
        x = self.batched_linear_layer(x, weights.get('w_o').view(weights.get('w_o').shape[0],
                                                                 (int(self.hidden_dim * self.hidden_scale) + 1),
                                                                 self.output_dim))
        return x


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        x = F.gelu(self.fc_h(x))
        return self.fc_o(x)
