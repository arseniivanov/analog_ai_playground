# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 3: MNIST training.

MNIST training example based on the paper:
https://www.frontiersin.org/articles/10.3389/fnins.2016.00333/full

Uses learning rates of η = 0.01, 0.005, and 0.0025
for epochs 0–10, 11–20, and 21–30, respectively.
"""
# pylint: disable=invalid-name

import os
from time import time
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from diff_quant import DifferentialQuantizedLayer, DifferentialQuantizedLayerV2

# Check device

USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Training parameters.
EPOCHS = 20
BATCH_SIZE = 64

def adjust_model_weights_to_bins(model, pos_bins, neg_bins):
    """
    Adjust the weights of the model to be the quantized weights using the given bins.
    """
    # Combine positive and negative bins and compute all pairwise differences
    combined_bins = pos_bins + neg_bins + [p-n for p in pos_bins for n in neg_bins]
    
    # Iterate over each layer in the model
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # Quantize each weight in the layer using the combined_bins
            with torch.no_grad():
                flattened_weights = layer.weight.data.flatten()
                quantized_weights = torch.Tensor([quantize_weight(w.item(), combined_bins) for w in flattened_weights])
                layer.weight.data = quantized_weights.reshape(layer.weight.data.shape)
                
                # If there is a bias, quantize it as well
                if layer.bias is not None:
                    flattened_bias = layer.bias.data.flatten()
                    quantized_bias = torch.Tensor([quantize_weight(b.item(), combined_bins) for b in flattened_bias])
                    layer.bias.data = quantized_bias.reshape(layer.bias.data.shape)
    
    # Return the modified model
    return model

# Extract weights from the model
def extract_weights_from_model(model):
    weights = []
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            weights.append(layer.weight.detach().cpu().numpy().flatten())
            if layer.bias is not None:
                weights.append(layer.bias.detach().cpu().numpy().flatten())
    return np.concatenate(weights)

def initialize_bins(weights, N):
    """Initialize bins for quantization."""
    # Extract positive and negative weights
    pos_weights = [w for w in weights if w >= 0]
    neg_weights = [-w for w in weights if w < 0]
    
    # Initial delta calculations
    delta_pos = max(pos_weights) / N if pos_weights else 0
    delta_neg = max(neg_weights) / N if neg_weights else 0
    
    # Create bins based on deltas
    pos_bins = [i * delta_pos for i in range(1, N+1)]
    neg_bins = [-i * delta_neg for i in range(1, N+1)]
    
    return pos_bins, neg_bins

def differential_quantization(model, bins, preset=False):
    model_weights = extract_weights_from_model(model)
    if not preset:
        pos_bins, neg_bins = initialize_bins(model_weights, bins)
        delta_pos, delta_neg = optimize_bins_strict_multiplicative(model_weights, pos_bins, neg_bins)
    else:
        delta_pos = preset[0]
        delta_neg = preset[1]
    print(delta_pos, delta_neg)
    adjusted_model = adjust_model_weights_to_bins(model, delta_pos, delta_neg).to("cuda")
    print(adjusted_model)
    return adjusted_model

def quantize_weight(w, bins):
    """Quantizes a single weight using the provided bins."""
    closest_value = min(bins, key=lambda x: abs(w - x))
    return closest_value

def quantization_error_for_bins(weights, pos_bins, neg_bins):
    """Computes the total quantization error for a set of weights using the provided bins."""
    # Combine positive and negative bins and compute all pairwise differences
    combined_bins = pos_bins + neg_bins + [p-n for p in pos_bins for n in neg_bins]
    error = sum([abs(w - quantize_weight(w, combined_bins)) for w in weights])
    return error

def adjust_bin_multiplicative(bin_val, delta, direction, factor=0.1):
    """Adjust a bin value by a fraction of delta in the given direction."""
    return bin_val + delta * factor * direction

import random
def optimize_bins_strict_multiplicative(weights, pos_bins, neg_bins, iterations=100, factor=1, sample_fraction=0.1, convergence_threshold=0.01):
    """Optimize bins using hill climbing while maintaining strict multiplicative constraint."""
    previous_error = float('inf')
    
    # Sample a subset of weights for faster error estimation
    sample_size = int(len(weights) * sample_fraction)
    sampled_weights = random.sample(list(weights), sample_size)
    
    # Ensure minimum delta values
    min_delta_pos = max(weights) * 0.01  # 1% of the maximum positive weight
    min_delta_neg = abs(min(weights)) * 0.01  # 1% of the maximum negative weight
    
    for c in range(iterations):
        print("Iteration: ", c)
        print("Error: ", previous_error)
        d_0_pos = max(pos_bins[0], min_delta_pos)
        d_0_neg = max(abs(neg_bins[0]), min_delta_neg)
        
        # Adjust each positive bin
        for i in range(len(pos_bins)):
            original_bin = pos_bins[i]
            error_original = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
            
            # Increase the bin value slightly
            pos_bins[i] = adjust_bin_multiplicative(original_bin, d_0_pos, 1, factor)
            error_increase = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
            
            # Decrease the bin value slightly if increasing didn't reduce error
            if error_increase >= error_original:
                pos_bins[i] = adjust_bin_multiplicative(original_bin, d_0_pos, -1, factor)
                error_decrease = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
                
                # Revert to original value if neither direction reduced error
                if error_decrease >= error_original:
                    pos_bins[i] = original_bin
        
        # Adjust each negative bin
        for i in range(len(neg_bins)):
            original_bin = neg_bins[i]
            error_original = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
            
            # Increase the bin value slightly
            neg_bins[i] = adjust_bin_multiplicative(original_bin, d_0_neg, 1, factor)
            error_increase = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
            
            # Decrease the bin value slightly if increasing didn't reduce error
            if error_increase >= error_original:
                neg_bins[i] = adjust_bin_multiplicative(original_bin, d_0_neg, -1, factor)
                error_decrease = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
                
                # Revert to original value if neither direction reduced error
                if error_decrease >= error_original:
                    neg_bins[i] = original_bin
        
        # Check for convergence
        current_error = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
        if abs(previous_error - current_error) < convergence_threshold:
            break
        previous_error = current_error
    
    # Ensure strict multiplicative constraints
    pos_bins = [i * d_0_pos for i in range(1, len(pos_bins) + 1)]
    neg_bins = [-i * d_0_neg for i in range(1, len(neg_bins) + 1)]
    
    return pos_bins, neg_bins

def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data

def create_analog_network():
# Modify the original model to use the custom layers
    model = nn.Sequential(
        # 1st Convolutional Layer
        #DifferentialQuantizedLayerV2(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1), 4),
        nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=3, padding=1),

        # 2nd Convolutional Layer
        #DifferentialQuantizedLayerV2(nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1), 4),
        nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

        # Flattening before passing to dense layers
        nn.Flatten(),

        # 1st Dense Layer
        #DifferentialQuantizedLayerV2(nn.Linear(192,10), 2),
        nn.Linear(192,10),
        nn.LogSoftmax(dim=1)
    )
    if USE_CUDA:
        model.cuda()
    return model

def create_sgd_optimizer(model, lr):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = SGD(model.parameters(), lr=lr)
    return optimizer

def train(model, train_set):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    lr = 0.1
    optimizer = create_sgd_optimizer(model, lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)
    model.train()

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {} - Training loss: {:.16f}".format(epoch_number, total_loss / len(train_set)))
        if total_loss / len(train_set) < 0.1:
            break
        
        # Decay learning rate if needed.
        scheduler.step()

    print("\nTraining Time (s) = {}".format(time() - time_init))


def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Save initial state of the model for resetting before each drift operation.

    model.eval()

    predicted_ok = 0
    total_images = 0

    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    print("\nFinal Number Of Images Tested = {}".format(total_images))
    print("Final Model Accuracy = {}".format(predicted_ok / total_images))


TEST = 1

def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.
    train_dataset, validation_dataset = load_images()

    model = create_analog_network()

    if not TEST:
        # Train the model.
        train(model, train_dataset)

        # Evaluate the trained model.
        test_evaluation(model, validation_dataset)

        torch.save(model.state_dict(), 'model_checkpoint.pth')
    
    model.load_state_dict(torch.load('model_checkpoint.pth', map_location="cuda"))

    preset = [[0.00875668168067932, 0.01751336336135864, 0.026270045042037962, 0.03502672672271728, 0.043783408403396604, 0.052540090084075924, 0.061296771764755245, 0.07005345344543457], [-0.1439693421125412, -0.2879386842250824, -0.4319080263376236, -0.5758773684501648, -0.719846710562706, -0.8638160526752472, -1.0077853947877884, -1.1517547369003296]]

    quantized_model = differential_quantization(model, 8, preset=preset)

    # Evaluate the differentially quantized model.
    test_evaluation(quantized_model, validation_dataset)
    
    torch.save(model.state_dict(), 'quantized_mnist.pth')



if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()