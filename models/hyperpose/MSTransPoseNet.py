"""
The Multi-Scene TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone


class MSTransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = False
        num_scenes = config.get("num_scenes")
        self.backbone = build_backbone(config)

        config_t = {**config,
                    "num_encoder_layers": config["num_t_encoder_layers"],
                    "num_decoder_layers": config["num_t_decoder_layers"]}
        config_rot = {**config,
                      "num_encoder_layers": config["num_rot_encoder_layers"],
                      "num_decoder_layers": config["num_rot_decoder_layers"]}
        self.transformer_t = Transformer(config_t)
        self.transformer_rot = Transformer(config_rot)

        decoder_dim = self.transformer_t.d_model

        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)

        self.query_embed_t = nn.Embedding(num_scenes, decoder_dim)
        self.query_embed_rot = nn.Embedding(num_scenes, decoder_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.scene_embed = nn.Linear(decoder_dim * 2, 1)
        self.regressor_head_t = nn.Sequential(*[PoseRegressor(decoder_dim, 3) for _ in range(num_scenes)])
        self.regressor_head_rot = nn.Sequential(*[PoseRegressor(decoder_dim, 4) for _ in range(num_scenes)])

        self._features, self._embeds, self._pos = None, None, None
        self._src_t, self._mask_t = None, None
        self._src_rot, self._mask_rot = None, None
        self._max_indices = None
        self._global_desc_t, self._global_desc_rot = None, None
        self._scene_log_distr = None

    def forward_transformers(self, samples, scene_indices):
        """
        Forward of the Transformers
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
            scene_log_distr: the log softmax over the scenes
            max_indices: the index of the max value in the scene distribution
        """
        batch_size = samples.shape[0]

        # Handle data structures
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # Extract the features and the position embedding from the visual backbone
        self._features, self._embeds, self._pos = self.backbone(samples)

        self._src_t, self._mask_t = self._features[0].decompose()
        self._src_rot, self._mask_rot = self._features[1].decompose()

        # Run through the transformer to translate to "camera-pose" language
        assert self._mask_t is not None
        assert self._mask_rot is not None
        local_descs_t = self.transformer_t(self.input_proj_t(self._src_t),
                                           self._mask_t,
                                           self.query_embed_t.weight,
                                           self._pos[0])[0][0]
        local_descs_rot = self.transformer_rot(self.input_proj_rot(self._src_rot),
                                               self._mask_rot,
                                               self.query_embed_rot.weight,
                                               self._pos[1])[0][0]

        # Get the scene index with FC + log-softmax
        self._scene_log_distr = torch.log(nn.Softmax(dim=1)(self.scene_embed(torch.cat((local_descs_t,
                                                                                  local_descs_rot), dim=2)))).squeeze(2)

        _, self._max_indices = self._scene_log_distr.max(dim=1)
        if scene_indices is not None:
            self._max_indices = scene_indices

        # Take the global latents by zeroing other scene's predictions and summing up
        w = local_descs_t * 0
        w[range(batch_size), self._max_indices, :] = 1
        self._global_desc_t = torch.sum(w * local_descs_t, dim=1)
        self._global_desc_rot = torch.sum(w * local_descs_rot, dim=1)

    def forward_heads(self):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        batch_size = self._global_desc_t.shape[0]
        expected_pose = torch.zeros((batch_size, 7)).to(self._global_desc_t.device).to(self._global_desc_t.dtype)
        for i in range(batch_size):
            x_t = self.regressor_head_t[self._max_indices[i]](self._global_desc_t[i].unsqueeze(0))
            x_rot = self.regressor_head_rot[self._max_indices[i]](self._global_desc_rot[i].unsqueeze(0))
            expected_pose[i, :] = torch.cat((x_t, x_rot), dim=1)
        return expected_pose, self._scene_log_distr

    def forward(self, samples, scene_indices):
        """ The forward pass expects a dictionary with the following keys-values
         'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
         'scene_indices': ground truth scene indices for each image (can be None)

        returns a dictionary with the following keys-values;
        'pose': expected pose (NX7)
        'log_scene_distr': (log) probability distribution over scenes
        """
        self.forward_transformers(samples, scene_indices)

        # Regress the pose from the image descriptors
        expected_pose, self._scene_log_distr = self.forward_heads()

        return expected_pose, self._scene_log_distr


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
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
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)
