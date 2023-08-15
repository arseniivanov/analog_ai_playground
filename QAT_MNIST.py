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

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from torch.nn import Sequential, Linear

# Check device

USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10
N_BITS = 8

# Training parameters.
EPOCHS = 20
BATCH_SIZE = 64

def find_global_min_max(model):
    """Find global minimum and maximum weight values across all layers."""
    all_weights = []
    for param in model.parameters():
        all_weights.append(param.data.flatten())
    all_weights = torch.cat(all_weights)
    global_min = all_weights.min()
    global_max = all_weights.max()
    return global_min, global_max

def global_quantize_weights(model, n_bits):
    """Quantize the weights of the entire model to n_bits levels using the same scale."""
    w_min, w_max = find_global_min_max(model)
    for param in model.parameters():
        # Map weights to [0, 1] range
        param.data = (param.data - w_min) / (w_max - w_min)

        # Quantize
        scale = 2 ** n_bits - 1
        param.data = torch.round(param.data * scale) / scale

        # Map weights back to original range
        param.data = param.data * (w_max - w_min) + w_min

    return model



def load_images():
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    return train_data, validation_data

def create_analog_network(input_size, hidden_sizes, output_size):
    model = Sequential(
        Linear(
            input_size,
            hidden_sizes[0],
            bias=False,
        ),
        nn.ReLU(),
        Linear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=False,
        ),
        nn.ReLU(),
        Linear(
            hidden_sizes[1],
            output_size,
            bias=False,
        ),
        nn.LogSoftmax(dim=1),
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
    lr = 0.5
    optimizer = create_sgd_optimizer(model, lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3)

    time_init = time()
    for epoch_number in range(EPOCHS):
        total_loss = 0
        for images, labels in train_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            # Flatten MNIST images into a 784 vector.
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            # Add training Tensor to the model (input).
            output = model(images)
            loss = classifier(output, labels)

            # Run training (backward propagation).
            loss.backward()

            # Optimize weights.
            optimizer.step()

            # Quantize weights
            with torch.no_grad():
                global_quantize_weights(model, N_BITS)

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
    # Quantize weights
    with torch.no_grad():
        global_quantize_weights(model, N_BITS)

    model.eval()

    predicted_ok = 0
    total_images = 0

    for images, labels in val_set:
        # Predict image.
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        images = images.view(images.shape[0], -1)
        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()

    print("\nFinal Number Of Images Tested = {}".format(total_images))
    print("Final Model Accuracy = {}".format(predicted_ok / total_images))




def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.
    train_dataset, validation_dataset = load_images()

    # Prepare the model.
    model = create_analog_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

    # Train the model.
    train(model, train_dataset)

    # Evaluate the trained model.
    test_evaluation(model, validation_dataset)
    import pdb;pdb.set_trace()
    
    torch.save(model.state_dict(), 'model_checkpoint.pth')



if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()