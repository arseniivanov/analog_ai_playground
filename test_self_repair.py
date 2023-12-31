import torch
import matplotlib.pyplot as plt

import os

from lib_digital import load_images, load_digital_model, plot_accuracies, calculate_accuracy
from lib_analog import prepare_digital_to_analog, convert_to_analog_step, DriftMonitor
from config import DEVICE, MODEL_PATH, PATH_DATASET, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH, POS_FACTOR, NEG_FACTOR

def reset_weights_to_nearest_multiplier(layer, layer_name, pos_multipliers_dict, neg_multipliers_dict, pos_factor, neg_factor):
    with torch.no_grad():
        pos_multipliers = pos_multipliers_dict[layer_name]
        neg_multipliers = neg_multipliers_dict[layer_name]
        
        for name, param in layer.named_parameters():
            if "weight" in name:
                flat_weights = param.data.view(-1)
                flat_pos_multipliers = pos_multipliers.view(-1)
                flat_neg_multipliers = neg_multipliers.view(-1)
                
                for i in range(len(flat_weights)):
                    flat_weights[i] = flat_pos_multipliers[i] * pos_factor + flat_neg_multipliers[i] * neg_factor
                
                param.data = flat_weights.view(param.data.shape)

def test_evaluation_self_repairing(model, val_set, pos_multipliers_file, neg_multipliers_file):
    """Test trained network

    Args:
        model (nn.Model): Trained model to be evaluated
        val_set (DataLoader): Validation set to perform the evaluation
    """

    times = []
    accuracies = []

    # Save initial state of the model for resetting before each drift operation.
    pos_multipliers_dict = torch.load(pos_multipliers_file, map_location=DEVICE)
    neg_multipliers_dict = torch.load(neg_multipliers_file, map_location=DEVICE)

    drift_monitor = DriftMonitor(model)
    drift_monitor.record_initial_outputs()

    #We do not want to reset the weights
    #initial_state = model.state_dict()
    model.eval()

    accuracy, total_images = calculate_accuracy(model, val_set)
    print("Model Accuracy at initialization = {}".format(accuracy))
    times.append(0)
    accuracies.append(accuracy)

    multiple = 300
    timesteps = 20
    drift_threshold = 0.05

    for t_inference in range(1,timesteps+1):
        # Reset the model to its initial state.
        #model.load_state_dict(initial_state)
        # Apply drift to the weights.
        model.drift_analog_weights(multiple)
        model.eval()

        # Setup counter of images predicted to 0.
        accuracy, total_images = calculate_accuracy(model, val_set)

        print("Number Of Images Tested at t={} = {}".format(t_inference*multiple, total_images))
        print("Model Accuracy at t={} = {}".format(t_inference*multiple, accuracy))
        times.append(t_inference*multiple)
        accuracies.append(accuracy)
        
        # Check for drift
        drifted_layers = drift_monitor.check_drift(threshold=1e-6)
        # Print drifted layers and their drift metrics
        reset = False
        for layer_name, metrics in drifted_layers.items():
            print(f"Layer {layer_name} - Total Drift: {metrics['total_drift']}, Normalized Drift: {metrics['normalized_drift']}")
                
            if metrics['normalized_drift'] > drift_threshold:
                print(f"Resetting weights for Layer {layer_name} due to excessive drift.")
                layer = dict(model.named_children())[layer_name]
                reset_weights_to_nearest_multiplier(layer, layer_name, pos_multipliers_dict, neg_multipliers_dict, POS_FACTOR, NEG_FACTOR)
                reset = True
            
        if reset:
            accuracy, total_images = calculate_accuracy(model, val_set)
            print("Model Accuracy after reset = {}".format(accuracy))
            times.append(t_inference*multiple)
            accuracies.append(accuracy)

    plot_accuracies(times, accuracies)

def main():
    """Train a PyTorch analog model with the MNIST dataset."""
    # Load datasets.
    batch_size = 64
    _, validation_dataset = load_images(PATH_DATASET, batch_size)
    path = os.path.join(MODEL_PATH, "quantized_mnist.pth")

    model = load_digital_model()
    model, rpu_config = prepare_digital_to_analog(model, path)
    analog_model = convert_to_analog_step(model, rpu_config)
    test_evaluation_self_repairing(analog_model, validation_dataset, POS_WEIGHTS_PTH, NEG_WEIGHTS_PTH)

if __name__ == "__main__":
    # Execute only if run as the entry point into the program.
    main()