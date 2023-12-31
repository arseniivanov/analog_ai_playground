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
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tiling import BitSlicedLinear

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, InferenceRPUConfig, WeightModifierType, WeightClipType, WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.rpu_base import cuda

from torch import Tensor
from typing import Optional, Type
from aihwkit.simulator.parameters.base import RPUConfigBase

class QuantizationAwareLinear(AnalogLinear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            rpu_config: Optional[RPUConfigBase] = None,
            tile_module_class: Optional[Type] = None,
            num_levels: int = 2**9 #2^10 working 10+epochs, lr = 0.215
    ):
        super().__init__(in_features, out_features, bias, rpu_config, tile_module_class)
        self.num_levels = num_levels

    def forward(self, x_input: Tensor) -> Tensor:
        weight, bias = self.get_weights()
        self.set_weights(self.quantize(weight), self.quantize(bias) if bias is not None else None)
        return super().forward(x_input)

    def quantize(self, x: Tensor) -> Tensor:
        x = (x + 1) / 2
        x = torch.round(x * (self.num_levels - 1)) / (self.num_levels - 1)
        # De-normalize to [-1, 1].
        x = x * 2 - 1
        return x



# Check device
USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Path where the datasets will be stored.
PATH_DATASET = os.path.join("data", "DATASET")

# Network definition.
INPUT_SIZE = 784
HIDDEN_SIZES = [256, 128]
OUTPUT_SIZE = 10

# Training parameters.
EPOCHS = 8
BATCH_SIZE = 64

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
    """Create the neural network using analog and digital layers.

    Args:
        input_size (int): size of the Tensor at the input.
        hidden_sizes (list): list of sizes of the hidden layers (2 layers).
        output_size (int): size of the Tensor at the output.

    Returns:
        nn.Module: created analog model
    """
    rpu_config = InferenceRPUConfig()
    rpu_config.forward.out_res = -1.0  # Turn off (output) ADC discretization.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02  # Short-term w-noise.

    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.pdrop = 0.03  # Drop connect.
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL  # Fwd/bwd weight noise.
    rpu_config.modifier.std_dev = 0.1
    rpu_config.modifier.rel_to_actual_wmax = True

    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()

    num_slices = 8

    model = AnalogSequential(
        BitSlicedLinear(
            input_size,
            hidden_sizes[0],
            num_slices,
            rpu_config=rpu_config,
            bias=False,
        ),
        nn.Sigmoid(),
        BitSlicedLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            num_slices,
            rpu_config=rpu_config,
            bias=False,
        ),
        nn.Sigmoid(),
        BitSlicedLinear(
            hidden_sizes[1],
            output_size,
            num_slices,
            rpu_config=rpu_config,
            bias=False,
        ),
        nn.LogSoftmax(dim=1),
    )

    if USE_CUDA:
        model.cuda()

    print(model)
    return model


def create_sgd_optimizer(model, lr):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer

def simulate_quantization(tensor):
    return torch.round(tensor)

def train(model, train_set):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    lr = 0.5
    optimizer = create_sgd_optimizer(model, lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

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

            # Simulate the quantization effects on the gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = simulate_quantization(param.grad.data)

            # Optimize weights.
            optimizer.step()

            for module in model.modules():
                if isinstance(module, BitSlicedLinear):
                    consolidated_weights = module.get_consolidated_weights()
                    module.transfer_weights(consolidated_weights)

            total_loss += loss.item()

        print("Epoch {} - Training loss: {:.16f}".format(epoch_number, total_loss / len(train_set)))
        
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
    initial_state = model.state_dict()
    model.eval()

    for t_inference in [0.0, 1.0, 20.0, 1000.0, 1e5]:
        # Reset the model to its initial state.
        model.load_state_dict(initial_state)
        # Apply drift to the weights.
        model.drift_analog_weights(t_inference)

        # Setup counter of images predicted to 0.
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

        print("Number Of Images Tested at t={} = {}".format(t_inference, total_images))
        print("Model Accuracy at t={} = {}".format(t_inference, predicted_ok / total_images))

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


if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()