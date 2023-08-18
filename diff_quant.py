import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch.autograd as autograd

class STEQuantize(autograd.Function):
    
    @staticmethod
    def forward(ctx, input, bitwidth):
        # Store bitwidth for backward, though we won't actually use it there.
        ctx.save_for_backward(bitwidth)
        
        # Normalize weights to [0,1] from [-1,1]
        normalized_weights = (input + 1) / 2
        
        # Quantize weights
        num_levels = 2 ** bitwidth.item()
        quantized_weights = torch.round(normalized_weights * (num_levels - 1)) / (num_levels - 1)
        
        # Convert quantized weights back to [-1,1] range
        quantized_weights = 2 * quantized_weights - 1
        
        return quantized_weights
    
    @staticmethod
    def backward(ctx, grad_output):
        # In backward pass, we just return the gradient as-is (i.e., STE).
        # We also need to return a None gradient for the second input (bitwidth), as it's not trainable.
        return grad_output, None

class DifferentialQuantizedLayer(nn.Module):
    def __init__(self, layer, bitwidth):
        super(DifferentialQuantizedLayer, self).__init__()
        self.layer = layer
        self.bitwidth = bitwidth
        self.num_levels = 2 ** bitwidth

    def quantize(self, weights):
        # Separate positive and negative weights
        positive_weights = torch.clamp(weights, 0, 1)
        negative_weights = torch.clamp(weights, -1, 0)
        
        # Normalize and quantize positive weights
        if positive_weights.max() != 0:  # Avoid division by zero
            positive_weights = (positive_weights - positive_weights.min()) / (positive_weights.max() - positive_weights.min())
        positive_weights_quantized = torch.round(positive_weights * (self.num_levels - 1) / 2) / ((self.num_levels - 1) / 2)
        
        # Normalize and quantize negative weights, then convert to [-1, 0] range
        if negative_weights.min() != 0:  # Avoid division by zero
            negative_weights = 1 + (negative_weights - negative_weights.max()) / (negative_weights.min() - negative_weights.max())
        negative_weights_quantized = torch.round(negative_weights * (self.num_levels - 1) / 2) / ((self.num_levels - 1) / 2) - 1

        return positive_weights_quantized, negative_weights_quantized

    def visualize_weights(self, original_weights, quantized_weights, title_prefix=""):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(original_weights.detach().cpu().numpy().flatten(), bins=100, alpha=0.7, color='blue', label='Original Weights')
        plt.title(f"{title_prefix} Original Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        plt.hist(quantized_weights.detach().cpu().numpy().flatten(), bins=100, alpha=0.7, color='red', label='Quantized Weights')
        plt.title(f"{title_prefix} Quantized Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()

    def forward(self, x):
        pos_weights, neg_weights = self.quantize(self.layer.weight)

        self.visualize_weights(self.layer.weight, pos_weights, title_prefix="Positive")
        self.visualize_weights(-self.layer.weight, neg_weights, title_prefix="Negative")
        import pdb;pdb.set_trace()

        # Using the original forward operation for ease
        pos_output = F.conv2d(x, pos_weights) if isinstance(self.layer, nn.Conv2d) else F.linear(x, pos_weights)
        neg_output = F.conv2d(x, -neg_weights) if isinstance(self.layer, nn.Conv2d) else F.linear(x, -neg_weights)

        return pos_output + neg_output

class DifferentialQuantizedLayerV2(nn.Module):
    """
    Easier implementation of the DifferentialQuantizedLayer. instead of storing two weights, we store one and then split
    positive and negative after training the model.
    """
    def __init__(self, layer, bitwidth):
        super(DifferentialQuantizedLayerV2, self).__init__()
        self.layer = layer
        self.bitwidth = bitwidth * 2  # Doubling the bitwidth to allow full [-1, 1] range
        self.num_levels = 2 ** self.bitwidth

    def quantize(self, weights):
        # Normalize weights to [0,1] from [-1,1]
        normalized_weights = (weights + 1) / 2
        
        # Quantize weights
        quantized_weights = torch.round(normalized_weights * (self.num_levels - 1)) / (self.num_levels - 1)
        
        # Convert quantized weights back to [-1,1] range
        quantized_weights = 2 * quantized_weights - 1
        
        return quantized_weights

    def visualize_weights(self, original_weights, quantized_weights, title_prefix=""):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(original_weights.detach().cpu().numpy().flatten(), bins=100, alpha=0.7, color='blue', label='Original Weights')
        plt.title(f"{title_prefix} Original Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        plt.subplot(1, 2, 2)
        plt.hist(quantized_weights.detach().cpu().numpy().flatten(), bins=100, alpha=0.7, color='red', label='Quantized Weights')
        plt.title(f"{title_prefix} Quantized Weights")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.show()

    def forward(self, x):
        quantized_weights = STEQuantize.apply(self.layer.weight, torch.tensor(self.bitwidth, dtype=torch.int))
        #self.visualize_weights(self.layer.weight, quantized_weights, title_prefix="Weights")
        
        # Using the original forward operation for ease
        output = F.conv2d(x, quantized_weights) if isinstance(self.layer, nn.Conv2d) else F.linear(x, quantized_weights)
        
        return output
