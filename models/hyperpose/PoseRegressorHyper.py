import torch
from torch import nn
import torch.nn.functional as F


class PoseRegressorHyper(nn.Module):
    """
    The hyper-networks-based regression head
    This module receives both the input vector to process and the weights for its regression layers
    """

    def __init__(self, decoder_dim, hidden_dim, output_dim, hidden_scale=1.0):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the output dimension
        hidden_scale: (float) Determines the ratio between the input and the hidden layers' dimensions
        """
        super().__init__()
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_scale = hidden_scale

    @staticmethod
    def batched_linear_layer(x, wb):
        """
        Explicit implementation of a batched linear regression
        x: (Tensor) the input tensor to process
        wb: (Tensor) The weights and bias of the regression layer
        """
        # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
        one = torch.ones(*x.shape[:-1], 1, device=x.device)
        linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
        return linear_res.squeeze(1)

    @staticmethod
    def _swish(x):
        """
        Implementation of the Swish activation layer
        Reference: "Searching for Activation Functions" [https://arxiv.org/abs/1710.05941v2]
        """
        return x * F.sigmoid(x)

    def forward(self, x, weights):
        """
        Forward pass
        x: (Tensor) the input tensor to process
        weights: (Dict) A dictionary holding the weights and biases of the input, hidden and output regression layers
        """
        # Regressing over the input layer
        if 'w_h1' in weights:
            x = self._swish(self.batched_linear_layer(x, weights.get('w_h1').view(weights.get('w_h1').shape[0],
                                                                                  (self.decoder_dim + 1),
                                                                                  self.hidden_dim)))
        # Regressing over all hidden layers
        for index in range(len(weights.keys()) - 2):
            if f'w_h{index + 2}' in weights:
                x = self._swish(self.batched_linear_layer(x,
                                                          weights.get(f'w_h{index + 2}').view(
                                                              weights.get(f'w_h{index + 2}').shape[0],
                                                              (self.hidden_dim + 1),
                                                              (int(self.hidden_dim * self.hidden_scale)))))
        # Regressing over the output layer
        if 'w_o' in weights:
            x = self.batched_linear_layer(x, weights.get('w_o').view(weights.get('w_o').shape[0],
                                                                     (int(self.hidden_dim * self.hidden_scale) + 1),
                                                                     self.output_dim))
        return x
