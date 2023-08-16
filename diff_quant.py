import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentialQuantizedLayer(nn.Module):
    def __init__(self, layer, bitwidth):
        super(DifferentialQuantizedLayer, self).__init__()
        self.layer = layer
        self.bitwidth = bitwidth
        self.num_levels = 2 ** bitwidth

    def quantize(self, weights):
        # 1. Convert weights to [0, 1]
        weights = (weights - weights.min()) / (weights.max() - weights.min())

        # 2. Scale weights to the desired range
        weights = torch.round(weights * (self.num_levels - 1))

        # 3. Convert back to the original range with differential encoding
        pos_weights = torch.clamp(weights, 0, (self.num_levels - 1) / 2)
        neg_weights = weights - pos_weights

        return pos_weights, neg_weights

    def forward(self, x):
        pos_weights, neg_weights = self.quantize(self.layer.weight)

        # Using the original forward operation for ease
        # Alternatively, you could split the input and perform two separate operations
        pos_output = F.conv2d(x, pos_weights) if isinstance(self.layer, nn.Conv2d) else F.linear(x, pos_weights)
        neg_output = F.conv2d(x, -neg_weights) if isinstance(self.layer, nn.Conv2d) else F.linear(x, -neg_weights)

        return pos_output + neg_output
