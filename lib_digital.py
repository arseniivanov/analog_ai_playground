import torch
from torchvision import datasets, transforms
from torch import nn
from torch.optim import SGD

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