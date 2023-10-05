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

from lib_digital import load_digital_model, load_images, calculate_accuracy, plot_accuracies
from lib_analog import prepare_digital_to_analog, create_sgd_optimizer_analog
from config import DEVICE, PATH_DATASET, MODEL_PATH

# Training parameters.
EPOCHS = 9
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
    model.eval()
    times = []
    accuracies = []
    multiple = 300
    timesteps = 20
    
    initial_accuracy, _ = calculate_accuracy(model, val_set)
    print(f"Model Accuracy at initialization = {initial_accuracy}")
    times.append(0)
    accuracies.append(initial_accuracy)

    for t_inference in range(1, timesteps+1):
        model.drift_analog_weights(multiple)
        accuracy, total_images = calculate_accuracy(model, val_set)
        print(f"Number Of Images Tested at t={t_inference*multiple} = {total_images}")
        print(f"Model Accuracy at t={t_inference*multiple} = {accuracy}")
        
        times.append(t_inference * multiple)
        accuracies.append(accuracy)

    print(f"\nFinal Number Of Images Tested = {total_images}")
    print(f"Final Model Accuracy = {accuracy}")
    
    plot_accuracies(times, accuracies)

def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.
    train_dataset, validation_dataset = load_images(PATH_DATASET, BATCH_SIZE)
    path = os.path.join(MODEL_PATH, "quantized_mnist.pth")
    model = load_digital_model()
    model, rpu_config = prepare_digital_to_analog(model, path)
    test_evaluation(model, validation_dataset)
    analog_model = convert_to_analog(model, rpu_config)
    test_evaluation(analog_model, validation_dataset)
    train(analog_model, train_dataset)
    test_evaluation_future(analog_model, validation_dataset)
    #test_evaluation_future(analog_model, validation_dataset, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH)

if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()