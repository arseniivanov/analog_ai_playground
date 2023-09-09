import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, InferenceRPUConfig, WeightModifierType, WeightClipType, WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog

from lib_digital import load_digital_model, load_images, create_identity_input
from lib_analog import digital_to_analog, create_sgd_optimizer_analog
from config import DEVICE

# Training parameters.
EPOCHS = 15
BATCH_SIZE = 64

POS_WEIGHTS_PTH = "0.2300284672271867_7_multipliers.pth"
NEG_WEIGHTS_PTH = "-0.17908795726111681_8_multipliers.pth"

def train(model, train_set, epsilon=1e-1):
    """Train the network.

    Args:
        model (nn.Module): model to be trained.
        train_set (DataLoader): dataset of elements to use as input for training.
    """
    classifier = nn.NLLLoss()
    lr = 0.01
    optimizer = create_sgd_optimizer_analog(model, lr)
    scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

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
        
        # Decay learning rate if needed.
        scheduler.step()
        
    print("\nTraining Time (s) = {}".format(time() - time_init))
    state_dict = model.state_dict()
    torch.save(state_dict, 'analog_aware_model.pth')


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

def test_evaluation_self_repairing(model, val_set, pos_multipliers_file, neg_multipliers_file):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Save initial state of the model for resetting before each drift operation.
    pos_multipliers_dict = torch.load(pos_multipliers_file, map_location="cpu")
    neg_multipliers_dict = torch.load(neg_multipliers_file, map_location="cpu")

    # Extract initial factors from filenames
    
    pos_factor = float(pos_multipliers_file.split("_")[0])
    neg_factor = float(neg_multipliers_file.split("_")[0])

    # Dictionary to store initial values
    initial_values = {}

    model.eval()
    with torch.no_grad():
        for name, layer in model.named_children():
            id_input = create_identity_input(layer)
            if id_input is not None:
                x = layer(id_input)
                initial_values[name] = x.clone()

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
    #test_evaluation_future(analog_model, validation_dataset, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH)

if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()