import torch

def load_quantized_weights(file_path):
    """Load quantized weights from a file."""
    return torch.load(file_path)

def rescale_weights_to_byte(model_state):
    """Rescale the weights to fit within [0, 255] range and convert to byte format."""
    all_weights = []
    for key, weight in model_state.items():
        all_weights.append(weight.flatten())
    all_weights = torch.cat(all_weights)
    w_min = all_weights.min()
    w_max = all_weights.max()
    
    # Rescale to [0, 255] range
    for key in model_state:
        model_state[key] = ((model_state[key] - w_min) / (w_max - w_min) * 255).byte()
    
    return model_state

def main():
    # Load the quantized weights
    model_state = load_quantized_weights('quantized_mnist.pth')
    
    # Transform the weights to bit-sliced format
    byte_weights = rescale_weights_to_byte(model_state)
    import pdb;pdb.set_trace()
    # Optionally, you can save these byte weights if needed
    torch.save(byte_weights, 'byte_sliced_mnist.pth')
    
    print("Transformation completed!")

if __name__ == "__main__":
    main()
