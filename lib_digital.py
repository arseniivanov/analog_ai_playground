import torch
from torchvision import datasets, transforms
from torch import nn
from torch.optim import SGD
from matplotlib import pyplot as plt
from config import DEVICE

def create_sgd_optimizer_digital(model, lr):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = SGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer

def create_identity_input(layer):
    if isinstance(layer, torch.nn.Conv2d):
        # For Conv2d, creating an identity matrix is non-trivial.
        # You might want to pass a specific kind of tensor here.
        return torch.ones((1, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]))
    elif isinstance(layer, torch.nn.Linear):
        return torch.eye(layer.in_features)
    else:
        return None
    
def load_images(path, batch_size):
    """Load images for train from the torchvision datasets."""
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the images.
    train_set = datasets.MNIST(path, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(path, download=True, train=False, transform=transform)
    train_data = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

    return train_data, validation_data

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

def calculate_accuracy(model, val_set):
    predicted_ok = 0
    total_images = 0
    for images, labels in val_set:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        pred = model(images)
        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
    return predicted_ok / total_images, total_images

def plot_accuracies(times, accuracies):
    plt.figure(figsize=(10, 6))
    # Loop through the data and plot accordingly
    last_t = None
    last_acc = None

    for i in range(len(times)):
        t = times[i]
        acc = accuracies[i]
        
        if i > 0:
            linestyle = '--' if last_t == t else '-'
            color = 'r' if last_t == t else 'b'
            plt.plot([last_t, t], [last_acc, acc], linestyle=linestyle, color=color, marker='o')
        
        last_t = t
        last_acc = acc

    plt.xlabel('Time (t)')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over time with analog-aware training')
    plt.ylim(0.930, 0.975)  # Set the y-axis limits
    plt.grid(True)
    plt.show()