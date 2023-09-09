# Imports from PyTorch.
import torch
from torch import nn, Tensor

# Imports from aihwkit.
from aihwkit.nn import AnalogLinear 

class BitSlicedLinear(nn.Module):
    def __init__(self, in_features, out_features, num_slices, rpu_config=None, bias=True):
        super().__init__()
        self.num_slices = num_slices    
        self.tiles = nn.ModuleList([AnalogLinear(in_features, out_features, rpu_config=rpu_config, bias=bias) for _ in range(num_slices)])
        self.sig_factors = [2 ** i for i in range(num_slices)]
        self._init_weights()

    def _init_weights(self):
        self.transfer_weights(self.tiles[0].get_weights()[0])

    def transfer_weights(self, original_weights):
        # 1. Normalize weights to [0, 1]
        normalized_weights = (original_weights + 1) / 2.0
        
        # 2. Scale to the bit depth
        scaled_weights = torch.round(normalized_weights * (2 ** self.num_slices - 1))
        
        # 3. Convert to integer type
        int_weights = scaled_weights.to(torch.int64)
        
        # 4. Convert each weight to binary and set in the corresponding tile
        for i in range(self.num_slices):
            binary_bit = (int_weights >> i) & 1 
            self.tiles[i].set_weights(binary_bit)

    def get_consolidated_weights(self):
        # Initialize a zero tensor for the consolidated weights
        consolidated_weights = torch.zeros_like(self.tiles[0].get_weights()[0])

        # Iterate over each tile
        for i, tile in enumerate(self.tiles):
            # Extract the weight from the tile and add it to the consolidated weights tensor
            consolidated_weights += (2 ** i) * tile.get_weights()[0]

        return consolidated_weights

    def forward(self, x):
        # 1. Transform input to [0, 255]
        norm_factor = sum(self.sig_factors)
        x = x * norm_factor
        # 2. Bit slice the input to get num_slices bits representation
        x_slices = [((x.int() >> i) & 1).float() for i in range(self.num_slices)]
        # The shape of x_slices should be: (num_slices, batch_size, input_size)

        y = torch.zeros_like(self.tiles[0](x))

        # 3 & 4. Multiply with tiles and accumulate with significance
        for i, (tile, sig_factor) in enumerate(zip(self.tiles, self.sig_factors)):
            y += sig_factor * tile(x_slices[i])
        # 5. Normalize output to [0, 1]
        norm_factor = x.shape[1] * norm_factor #Max possible value of matrix
        norm_factor = y.max()
        y = y / norm_factor

        return y

    
class BitSlicedLinearADC(BitSlicedLinear):
    def __init__(self, in_features, out_features, num_slices, adc_bits=8):
        super().__init__(in_features, out_features, num_slices)
        self.adc_bits = adc_bits

    def adc_convert(self, value):
        # Simulate an ADC conversion. The exact implementation will depend on the desired ADC behavior.
        # For a simple quantization:
        max_val = 2**self.adc_bits - 1
        return torch.round(value * max_val) / max_val

    def forward(self, x_input: Tensor) -> Tensor:
        # Collect outputs from all tiles.
        outputs = [tile.forward(x_input) for tile in self.tiles]
        
        # ADC conversion
        outputs = [self.adc_convert(output) for output in outputs]

        # Combine outputs.
        combined_output = sum(output * (2**i) for i, output in enumerate(outputs))
        
        return combined_output
