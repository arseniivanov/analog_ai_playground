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
import matplotlib.pyplot as plt
import random
import copy

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
    combined_bins = pos_bins + neg_bins + [p+n for p in pos_bins for n in neg_bins]
    
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
    print(pos_bins, neg_bins)
    
    return pos_bins, neg_bins

def differential_quantization(model, bins, preset=False):
    model_weights = extract_weights_from_model(model)
    if not preset:
        pos_bins, neg_bins = initialize_bins(model_weights, bins)
        delta_pos, delta_neg = optimize_bins_simulated_annealing_linear(model_weights, pos_bins, neg_bins)
    else:
        delta_pos = preset[0]
        delta_neg = preset[1]
    print(delta_pos, delta_neg)
    adjusted_model = adjust_model_weights_to_bins(model, delta_pos, delta_neg).to("cuda")
    print(adjusted_model)
    return adjusted_model, delta_pos, delta_neg

def quantize_weight(w, bins):
    """Quantizes a single weight using the provided bins."""
    closest_value = min(bins, key=lambda x: abs(w - x))
    return closest_value

def quantization_error_for_bins(weights, pos_bins, neg_bins):
    """Computes the total quantization error for a set of weights using the provided bins."""
    # Combine positive and negative bins and compute all pairwise differences
    combined_bins = pos_bins + neg_bins + [p+n for p in pos_bins for n in neg_bins]
    error = sum([abs(w - quantize_weight(w, combined_bins)) for w in weights])
    return error

def adjust_bin_multiplicative(bin_val, delta, direction, factor=0.1):
    """Adjust a bin value by a fraction of delta in the given direction."""
    return bin_val + delta * factor * direction

def adjust_bins_multiplicative(bins, delta, direction, bin_count, factor=0.1):
    """Adjust a bin value by a fraction of delta in the given direction."""
    std = bins[0] + delta * factor * direction
    new_bins = [std * k for k in range(1,bin_count)]
    return new_bins

def simulated_annealing_schedule(T, alpha=0.95):
    """Cooling schedule for simulated annealing."""
    return T * alpha

def perturb_bin_base(bin_base, delta, direction):
    """Perturb a bin base value in the given direction while maintaining the multiplicative constraint."""
    return bin_base + delta * direction

def create_non_linear_bins(smallest_bin, bin_count, max_multiple):
    """Create non-linear bins using various multiples of the smallest bin."""
    multiples = [1] + list(np.random.choice(range(2, max_multiple+1), size=bin_count-1, replace=False))
    return [smallest_bin * multiple for multiple in multiples]

def optimize_bins_simulated_annealing_non_linear(weights, pos_bins, neg_bins, iterations=80, T=1.0, alpha=0.95, max_multiple=12):
    """Optimize bins using simulated annealing allowing for non-linear binning."""
    current_pos_smallest_bin = pos_bins[0]
    current_neg_smallest_bin = abs(neg_bins[0])
    bin_count = len(pos_bins)
    current_error = quantization_error_for_bins(weights, pos_bins, neg_bins)

    best_pos_bins = pos_bins.copy()
    best_neg_bins = neg_bins.copy()
    best_error = current_error

    for i in range(iterations):
        print("Iteration: ", i)
        print("Error: ", current_error)
        # Apply cooling schedule
        T = simulated_annealing_schedule(T, alpha)

        # Perturb positive and negative smallest bins
        proposed_pos_smallest_bin = current_pos_smallest_bin + T * np.random.randn() * (max(weights) / 100)
        proposed_neg_smallest_bin = current_neg_smallest_bin + T * np.random.randn() * (abs(min(weights)) / 100)

        # Create non-linear bins using various multiples
        proposed_pos_bins = create_non_linear_bins(proposed_pos_smallest_bin, bin_count, max_multiple)
        proposed_neg_bins = [-x for x in create_non_linear_bins(proposed_neg_smallest_bin, bin_count, max_multiple)]

        # Compute error for proposed bins
        proposed_error = quantization_error_for_bins(weights, proposed_pos_bins, proposed_neg_bins)

        # Metropolis criterion
        if proposed_error < current_error or np.random.rand() < np.exp(-(proposed_error - current_error) / T):
            current_pos_smallest_bin = proposed_pos_smallest_bin
            current_neg_smallest_bin = proposed_neg_smallest_bin
            current_error = proposed_error

            if proposed_error < best_error:
                best_pos_bins = proposed_pos_bins.copy()
                best_neg_bins = proposed_neg_bins.copy()
                best_error = proposed_error

    return sorted(best_pos_bins), sorted(best_neg_bins, reverse=True)


def optimize_bins_simulated_annealing_linear(weights, pos_bins, neg_bins, iterations=15, T=1.0, alpha=0.95):
    """Optimize bins using simulated annealing."""
    current_pos_base = pos_bins[0]
    current_neg_base = abs(neg_bins[0])
    bin_count = len(pos_bins)
    current_error = quantization_error_for_bins(weights, pos_bins, neg_bins)

    best_pos_base = current_pos_base
    best_neg_base = current_neg_base
    best_error = current_error

    for i in range(iterations):
        print("Iteration: ", i)
        print("Error: ", current_error)
        # Apply cooling schedule
        T = simulated_annealing_schedule(T, alpha)

        # Perturb positive and negative bases
        direction_pos = np.random.choice([-1, 1])
        direction_neg = np.random.choice([-1, 1])
        proposed_pos_base = perturb_bin_base(current_pos_base, T * np.random.rand() * (max(weights) / 100), direction_pos)
        proposed_neg_base = perturb_bin_base(current_neg_base, T * np.random.rand() * (abs(min(weights)) / 100), direction_neg)

        # Recreate bins based on new bases
        proposed_pos_bins = [proposed_pos_base * k for k in range(1, bin_count+1)]
        proposed_neg_bins = [-proposed_neg_base * k for k in range(1, bin_count+1)]

        # Compute error for proposed bins
        proposed_error = quantization_error_for_bins(weights, proposed_pos_bins, proposed_neg_bins)

        # Metropolis criterion
        if proposed_error < current_error or np.random.rand() < np.exp(-(proposed_error - current_error) / T):
            current_pos_base = proposed_pos_base
            current_neg_base = proposed_neg_base
            current_error = proposed_error

            if proposed_error < best_error:
                best_pos_base = proposed_pos_base
                best_neg_base = proposed_neg_base
                best_error = proposed_error

    best_pos_bins = [best_pos_base * k for k in range(1, bin_count+1)]
    best_neg_bins = [-best_neg_base * k for k in range(1, bin_count+1)]

    return best_pos_bins, best_neg_bins

def optimize_bins_strict_multiplicative(weights, pos_bins, neg_bins, iterations=100, factor=1, sample_fraction=1.0, convergence_threshold=0.01):
    """Optimize bins using hill climbing while maintaining strict multiplicative constraint."""
    previous_error = float('inf')
    bin_count = 8
    
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
        
        # Adjust all positive bins with the same factor
        original_pos_bins = pos_bins.copy()
        pos_bins = adjust_bins_multiplicative(original_pos_bins, d_0_pos, 1, bin_count, factor)
        error_increase_pos = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)

        # Revert to original if increasing didn't reduce error
        if error_increase_pos >= previous_error:
            if adjust_bin_multiplicative(copy.deepcopy(original_pos_bins[0]), d_0_pos, -1, factor) < 0:
                error_decrease_pos = 1000
            else:
                pos_bins = adjust_bins_multiplicative(original_pos_bins, d_0_pos, -1, bin_count, factor)
                error_decrease_pos = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)

            # Revert to original if neither direction reduced error
            if error_decrease_pos >= previous_error:
                pos_bins = original_pos_bins

        # Adjust all negative bins with the same factor
        original_neg_bins = neg_bins.copy()
        if adjust_bin_multiplicative(copy.deepcopy(original_neg_bins[0]), d_0_neg, 1, factor) >= 0:
            error_increase_neg = 1000
        else:
            neg_bins = adjust_bins_multiplicative(original_neg_bins, d_0_neg, 1, bin_count, factor)
            error_increase_neg = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)

        # Revert to original if increasing didn't reduce error
        if error_increase_neg >= previous_error:
            neg_bins = adjust_bins_multiplicative(original_neg_bins, d_0_neg, -1, bin_count, factor)
            error_decrease_neg = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
            
            # Revert to original if neither direction reduced error
            if error_decrease_neg >= previous_error:
                neg_bins = original_neg_bins
        
        # Check for convergence
        current_error = quantization_error_for_bins(sampled_weights, pos_bins, neg_bins)
        if abs(previous_error - current_error) < convergence_threshold:
            break
        previous_error = current_error
    
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

def train(model, train_set, epsilon=1e-1):
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

            # Add a penalty term for weights below epsilon
            penalty = 0.0
            for param in model.parameters():
                penalty += torch.sum((epsilon - torch.abs(param)) * (torch.abs(param) < epsilon))
            
            loss = loss + penalty
            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            total_loss += loss.item()

        print("Epoch {} - Training loss: {:.16f}".format(epoch_number, total_loss / len(train_set)))
        if total_loss / len(train_set) < 0.08:
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

def construct_weight_matrix_and_histogram(model, pos_bins, neg_bins):
    # Extract the quantized weights from the model
    quantized_weights = extract_weights_from_model(model)

    # Create combined bins and initialize weight counts
    combined_bins = [p + n for p in pos_bins for n in neg_bins]
    all_bins = pos_bins + neg_bins + combined_bins
    weight_counts = {bin_value: 0 for bin_value in all_bins}

    # Iterate through the quantized weights and find the corresponding bin
    for weight in quantized_weights:
        closest_bin = min(all_bins, key=lambda x: abs(weight - x))
        weight_counts[closest_bin] += 1

    # Filter out bins with 0 entries
    weight_counts = {k: v for k, v in weight_counts.items() if v > 0}

    # Separate bins and counts by type
    pos_counts = [(k, v) for k, v in weight_counts.items() if k in pos_bins]
    neg_counts = [(k, v) for k, v in weight_counts.items() if k in neg_bins]
    combined_counts = [(k, v) for k, v in weight_counts.items() if k in combined_bins]

    # Plot the scatter plot
    for counts, color, label in zip([pos_counts, neg_counts, combined_counts], ['blue', 'red', 'green'], ['Positive', 'Negative', 'Combined']):
        if counts:  # Check if counts is not empty
            bins, values = zip(*counts)
            plt.scatter(bins, values, color=color, label=label, s=30)
    
    plt.xlabel('Bin Value')
    plt.ylabel('Count')
    plt.title('Scatter Plot of Weight Counts')
    plt.legend()
    plt.grid(True)
    plt.show()

    return weight_counts


def construct_multipliers(model, pos_bins, neg_bins, pos_multipliers_dict, neg_multipliers_dict):
    combined_bins = pos_bins + neg_bins + [p + n for p in pos_bins for n in neg_bins]

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            with torch.no_grad():
                # Initialize matrices to store multipliers
                pos_multipliers = torch.zeros_like(layer.weight, dtype=torch.int8)
                neg_multipliers = torch.zeros_like(layer.weight, dtype=torch.int8)

                # Flatten the weight tensor for easier iteration
                flattened_weights = layer.weight.data.flatten()

                for idx, w in enumerate(flattened_weights):
                    closest_value = quantize_weight(w.item(), combined_bins)
                    
                    # Determine which bin the closest_value belongs to and store its index (multiplier)
                    if closest_value in pos_bins:
                        pos_multipliers.view(-1)[idx] = pos_bins.index(closest_value) + 1  # +1 to account for zero-based index
                    elif closest_value in neg_bins:
                        neg_multipliers.view(-1)[idx] = neg_bins.index(closest_value) + 1  # +1 to account for zero-based index

                # Store the multiplier matrices in the dictionaries
                pos_multipliers_dict[name] = pos_multipliers
                neg_multipliers_dict[name] = neg_multipliers
    return pos_multipliers_dict, neg_multipliers_dict

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

    test_evaluation(model, validation_dataset)

    preset = [[0.2300284672271867, 0.4600569344543734, 0.6900854016815601, 0.9201138689087468, 1.1501423361359335, 1.3801708033631201, 1.610199270590307, 1.8402277378174936], [-0.17908795726111681, -0.35817591452223363, -0.5372638717833504, -0.7163518290444673, -0.8954397863055841, -1.0745277435667009, -1.2536157008278177, -1.4327036580889345]]
    #preset = None

    quantized_model, pos_bins, neg_bins = differential_quantization(model, 8, preset=preset)

    # Evaluate the differentially quantized model.
    test_evaluation(quantized_model, validation_dataset)
    print(pos_bins, neg_bins)
    
    simplified_weights = construct_weight_matrix_and_histogram(quantized_model, pos_bins, neg_bins)

    torch.save(model.state_dict(), 'quantized_mnist.pth')

    pos_multipliers_dict = {}
    neg_multipliers_dict = {}
    # Call the function to construct the multipliers
    construct_multipliers(quantized_model, pos_bins, neg_bins, pos_multipliers_dict, neg_multipliers_dict)
    # Save the multiplier dictionaries as .pth files
    concatenated_pos = torch.cat([torch.flatten(tensor) for tensor in pos_multipliers_dict.values()])
    concatenated_neg = torch.cat([torch.flatten(tensor) for tensor in neg_multipliers_dict.values()])

    # Find the maximum value in the concatenated tensor
    largest_multiplier_pos = torch.max(concatenated_pos).item()
    largest_multiplier_neg = torch.max(concatenated_neg).item()

    torch.save(pos_multipliers_dict, f"{pos_bins[0]}_{largest_multiplier_pos}_multipliers.pth")
    torch.save(neg_multipliers_dict, f"{neg_bins[0]}_{largest_multiplier_neg}_multipliers.pth")





if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()