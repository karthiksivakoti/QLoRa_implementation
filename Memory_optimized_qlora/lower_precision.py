from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm

def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Similar to block_quantize_4bit but for 3-bit quantization"""
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    
    x = x.view(-1, group_size)
    maxvals = x.abs().max(dim=1, keepdim=False).values
    x_norm = (x.t() / (maxvals + 1e-5)).t()
    x_quant = (x_norm * 3.5 + 3.5).round().clamp(0, 7).to(torch.uint8)
    
    # Pack 8 3-bit values into 3 bytes
    packed = torch.zeros(x.size(0), (group_size * 3) // 8, dtype=torch.uint8, device=x.device)
    
    for i in range(0, group_size, 8):
        j = (i * 3) // 8
        bits = x_quant[:, i:i+8]
        
        packed[:, j] = bits[:, 0] | (bits[:, 1] << 3) | ((bits[:, 2] & 0x3) << 6)
        packed[:, j+1] = ((bits[:, 2] & 0x4) >> 2) | (bits[:, 3] << 1) | (bits[:, 4] << 4) | ((bits[:, 5] & 0x1) << 7)
        packed[:, j+2] = ((bits[:, 5] & 0x6) >> 1) | (bits[:, 6] << 2) | (bits[:, 7] << 5)
    
    return packed.reshape(-1), maxvals.to(torch.float16)

def block_dequantize_3bit(x_packed: torch.Tensor, maxvals: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Similar to block_dequantize_4bit but for 3-bit quantization"""
    num_groups = len(maxvals)
    x_packed = x_packed.view(num_groups, -1)
    x_quant = torch.zeros(num_groups, group_size, dtype=torch.uint8, device=x_packed.device)
    
    for i in range(0, group_size, 8):
        j = (i * 3) // 8
        byte0 = x_packed[:, j]
        byte1 = x_packed[:, j+1]
        byte2 = x_packed[:, j+2]
        
        x_quant[:, i] = byte0 & 0x7
        x_quant[:, i+1] = (byte0 >> 3) & 0x7
        x_quant[:, i+2] = ((byte0 >> 6) & 0x3) | ((byte1 & 0x1) << 2)
        x_quant[:, i+3] = (byte1 >> 1) & 0x7
        x_quant[:, i+4] = (byte1 >> 4) & 0x7
        x_quant[:, i+5] = ((byte1 >> 7) & 0x1) | ((byte2 & 0x3) << 1)
        x_quant[:, i+6] = (byte2 >> 2) & 0x7
        x_quant[:, i+7] = (byte2 >> 5) & 0x7
    
    x_norm = x_quant.float() / 7.0 * 2.0 - 1.0
    x_norm = x_norm * maxvals.unsqueeze(1).to(torch.float32)
    return x_norm.reshape(-1)

class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size
        
        total_elements = in_features * out_features
        num_groups = (total_elements + group_size - 1) // group_size
        num_groups = num_groups + (8 - (num_groups % 8)) if num_groups % 8 != 0 else num_groups
        
        # For 3-bit quantization: each 8 values need 3 bytes
        packed_size = num_groups * ((group_size * 3) // 8)
        
        self.register_buffer("weight_q3", torch.zeros(packed_size, dtype=torch.uint8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(num_groups, dtype=torch.float16), persistent=False)
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def _load_state_dict_pre_hook(self, state_dict, prefix, *args):
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            weight = state_dict[weight_key]
            weight_flat = weight.reshape(-1)
            if weight_flat.size(0) % self._group_size != 0:
                pad_size = self._group_size - (weight_flat.size(0) % self._group_size)
                weight_flat = torch.nn.functional.pad(weight_flat, (0, pad_size))
            weight_q3, weight_norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(weight_q3)
            self.weight_norm.copy_(weight_norm)
            del state_dict[weight_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_dequant = block_dequantize_3bit(self.weight_q3, self.weight_norm, self._group_size)
        weight = weight_dequant[:self._shape[0] * self._shape[1]].reshape(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)

class LowerPrecisionBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x):
        return self.model(x)

def load(path: Path | None) -> LowerPrecisionBigNet:
    net = LowerPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net