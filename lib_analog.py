import torch
from torch import nn
    
from aihwkit.nn import AnalogLinear, AnalogSequential, AnalogConv2d
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, InferenceRPUConfig, WeightModifierType, WeightClipType, WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog
from config import DEVICE

class DriftMonitor:
    def __init__(self, model):
        self.model = model
        self.initial_outputs = {}

    def run_identity_through_layer(self, layer):
        if isinstance(layer, AnalogLinear):
            identity_input = torch.eye(layer.in_features).unsqueeze(0).to(DEVICE)
        elif isinstance(layer, AnalogConv2d):
            in_channels = layer.in_channels
            kernel_size = layer.kernel_size[0]
            identity_input = torch.zeros((1, in_channels, kernel_size, kernel_size)).to(DEVICE)
            for i in range(min(in_channels, kernel_size)):
                identity_input[0, i, i, i] = 1.0
        return layer(identity_input)

    def record_initial_outputs(self):
        for name, layer in self.model.named_children():
            if isinstance(layer, (AnalogLinear, AnalogConv2d)):
                self.initial_outputs[name] = self.run_identity_through_layer(layer)

    def check_drift(self, threshold=1e-6):
        drifted_layers = {}
        for name, layer in self.model.named_children():
            if isinstance(layer, (AnalogLinear, AnalogConv2d)) and name in self.initial_outputs:
                current_output = self.run_identity_through_layer(layer)
                diff = torch.abs(current_output - self.initial_outputs[name]).sum().item()
                
                num_elements = current_output.numel()  # Number of elements in the tensor
                
                normalized_diff = diff / num_elements  # Normalized drift

                if diff > threshold:
                    drifted_layers[name] = {
                        'total_drift': diff,
                        'normalized_drift': normalized_diff
                    }

        return drifted_layers


def convert_to_analog_step(model, rpu_config):
    return convert_to_analog(model, rpu_config)

def create_sgd_optimizer_analog(model, lr):
    """Create the analog-aware optimizer.

    Args:
        model (nn.Module): model to be trained.
    Returns:
        nn.Module: optimizer
    """
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer

def prepare_digital_to_analog(model, path_to_model):
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
    model.load_state_dict(torch.load(path_to_model, map_location=DEVICE))

    return model, rpu_config