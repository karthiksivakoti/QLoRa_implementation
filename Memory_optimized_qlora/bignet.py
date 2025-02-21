from pathlib import Path

import torch

BIGNET_DIM = 1024
# Comment me out if you want to play with a 2GB model
# Please note though that you will not be able to load the bignet.pth checkpoint
# and grading will not work
# BIGNET_DIM = 6144


class LayerNorm(torch.nn.Module):
    """
    torch.nn.LayerNorm is a bit weird with the shape of the input tensor.
    We instead use torch.nn.functional.group_norm with num_groups=1.
    """

    num_channels: int
    eps: float

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        if affine:
            self.weight = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))
            self.bias = torch.nn.Parameter(torch.empty(num_channels, device=device, dtype=dtype))

            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch.nn.LayerNorm is a bit weird with the shape of the input tensor
        # GroupNorm handles this better
        r = torch.nn.functional.group_norm(x, 1, self.weight, self.bias, self.eps)
        return r


class BigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
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


def load(path: Path | None) -> BigNet:
    net = BigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
