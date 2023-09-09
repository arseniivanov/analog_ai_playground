import os
from time import time

# Imports from PyTorch.
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from torch.nn import Sequential

from digital_tiling import DigitalBitSlicedLinear
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

def create_bit_sliced_network(input_size, hidden_sizes, output_size, num_slices):
    model = Sequential(
        DigitalBitSlicedLinear(
            input_size,
            hidden_sizes[0],
            num_slices,
            bias=False
        ),
        nn.ReLU(),
        DigitalBitSlicedLinear(
            hidden_sizes[0],
            hidden_sizes[1],
            num_slices,
            bias=False
        ),
        nn.ReLU(),
        DigitalBitSlicedLinear(
            hidden_sizes[1],
            output_size,
            num_slices,
            bias=False
        ),
        nn.LogSoftmax(dim=1)
    )

    if USE_CUDA:
        model.cuda()
    return model

def transfer_to_bit_sliced(original_model, bit_sliced_model, num_slices):
    """Transfer weights from a standard model to a bit-sliced model."""
    # Iterate over pairs of layers from original and bit-sliced models
    for orig_layer, bit_sliced_layer in zip(original_model, bit_sliced_model):
        if isinstance(orig_layer, nn.Linear) and isinstance(bit_sliced_layer, DigitalBitSlicedLinear):
            # Transfer weights from the original layer to the bit-sliced layer
            bit_sliced_layer.transfer_weights(orig_layer.weight.data)

    return bit_sliced_model


def test_evaluation(model, val_set):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
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

def create_original_network(input_size, hidden_sizes, output_size):
    model = Sequential(
        nn.Linear(
            input_size,
            hidden_sizes[0],
            bias=False,
        ),
        nn.ReLU(),
        nn.Linear(
            hidden_sizes[0],
            hidden_sizes[1],
            bias=False,
        ),
        nn.ReLU(),
        nn.Linear(
            hidden_sizes[1],
            output_size,
            bias=False,
        ),
        nn.LogSoftmax(dim=1),
    )
    if USE_CUDA:
        model.cuda()
    return model

def main():
    """Test a PyTorch digital bit-sliced model with the MNIST dataset."""
    # Load datasets.
    _, validation_dataset = load_images()

    # Prepare the model.
    num_slices = 8  # Example value, adjust as needed
    bit_sliced_model = create_bit_sliced_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE, num_slices)
    
    # Create the original model and load its weights
    original_model = create_original_network(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)
    original_model.load_state_dict(torch.load('byte_sliced_mnist.pth'))

    # Transfer weights from the original model to the bit-sliced model
    bit_sliced_model = transfer_to_bit_sliced(original_model, bit_sliced_model, num_slices)

    # Evaluate the trained model.
    test_evaluation(bit_sliced_model, validation_dataset)


if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()