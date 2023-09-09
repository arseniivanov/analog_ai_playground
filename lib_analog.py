import torch
    
from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, InferenceRPUConfig, WeightModifierType, WeightClipType, WeightNoiseType
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.rpu_base import cuda
from aihwkit.nn.conversion import convert_to_analog
from config import DEVICE

def convert_to_analog(model, rpu_config):
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