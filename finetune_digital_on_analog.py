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
from aihwkit.nn.conversion import convert_to_analog

from torch import Tensor
from typing import Optional, Type

PATH_DATASET = os.path.join("data", "DATASET")

# Training parameters.
EPOCHS = 4
BATCH_SIZE = 64
DEVICE = torch.device("cpu")

def load_digital_model():
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

def digital_to_analog(model):
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
    model.load_state_dict(torch.load('model_checkpoint.pth', map_location="cpu"))

    return model, rpu_config

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

def train(model, train_set, epsilon=1e-1):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    lr = 0.01
    optimizer = create_sgd_optimizer(model, lr)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.5)

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
            penalty = 0.0
            for param in model.parameters():
                penalty += torch.sum((epsilon - torch.abs(param)) * (torch.abs(param) < epsilon))
            
            loss = loss + penalty
            # Run training (backward propagation).
            loss.backward()

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

def test_evaluation_future(model, val_set):
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

    model = load_digital_model()
    model, rpu_config = digital_to_analog(model)
    test_evaluation(model, validation_dataset)
    analog_model = convert_to_analog(model, rpu_config)
    test_evaluation(analog_model, validation_dataset)
    train(analog_model, train_dataset)
    test_evaluation_future(analog_model, validation_dataset)

if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()