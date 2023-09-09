import torch

import os

from lib_digital import create_identity_input, load_images, load_digital_model
from lib_analog import prepare_digital_to_analog, convert_to_analog
from config import DEVICE, MODEL_PATH, PATH_DATASET, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH

def test_evaluation_self_repairing(model, val_set, pos_multipliers_file, neg_multipliers_file):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """
    # Save initial state of the model for resetting before each drift operation.
    pos_multipliers_dict = torch.load(pos_multipliers_file, map_location=DEVICE)
    neg_multipliers_dict = torch.load(neg_multipliers_file, map_location=DEVICE)

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
    batch_size = 64
    validation_dataset = load_images(PATH_DATASET, batch_size)
    path = os.path.join(MODEL_PATH, "quantized_mnist.pth")

    model = load_digital_model()
    model, rpu_config = prepare_digital_to_analog(model, path)
    analog_model = convert_to_analog(model, rpu_config)
    test_evaluation_self_repairing(analog_model, validation_dataset, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH)

if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()