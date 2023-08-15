import torch
from torch import nn

class DigitalBitSlicedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_slices, bias=True):
        super(DigitalBitSlicedLinear, self).__init__()
        self.num_slices = num_slices
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Create nn.Linear layers for each slice
        self.slices = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias) for _ in range(num_slices)])
        self.sig_factors = torch.tensor([2 ** i for i in range(num_slices)], dtype=torch.float32).to(self.slices[0].weight.device)

    def forward(self, x):
        # Collect outputs from all slices
        import pdb;pdb.set_trace()
        outputs = [slice_layer(x) for slice_layer in self.slices]
        
        # Combine outputs using significance
        combined_output = sum(output * sig_factor for output, sig_factor in zip(outputs, self.sig_factors))
        
        return combined_output

    def transfer_weights(self, original_weights):
        int_weights = original_weights.to(torch.int64)

        for i, slice_layer in enumerate(self.slices):
            binary_bit = (int_weights >> i) & 1 
            slice_layer.weight.data = binary_bit.float()
            if self.bias:
                slice_layer.bias.data = torch.zeros_like(slice_layer.bias.data)
                
    def get_consolidated_weights(self):
        # Initialize a zero tensor for the consolidated weights
        consolidated_weights = torch.zeros(self.in_features, self.out_features, device=self.slices[0].weight.device)

        # Iterate over each slice
        for i, slice_layer in enumerate(self.slices):
            # Extract the weight from the slice and add it to the consolidated weights tensor
            consolidated_weights += (2 ** i) * slice_layer.weight.data

        return consolidated_weights
