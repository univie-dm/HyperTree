import math
import torch
import torch.nn as nn

from typing import Dict, Union
from src.hyperbolic_math.src.manifolds import Manifold, Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.nn_layers import (
    HyperbolicLinearPoincarePP,
    HyperbolicLinearHyperboloidFHNN,
    HyperbolicLinearHyperboloidFHCNN,
    HyperbolicLinearHyperboloidPP,
)


def get_dtype_from_precision(precision: str) -> torch.dtype:
    # Dictionary mapping of dtype strings to torch dtypes
    DTYPE_MAP: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if precision not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported precision: {precision}. Supported precisions are: {list(DTYPE_MAP.keys())}."
        )
    return DTYPE_MAP[precision]


class HyperboloidEncoder2TangentInput(torch.nn.Module):
    """Convert Euclidean encoder outputs to tangent vectors."""

    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.manifold, Hyperboloid):
            return torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
        else:
            return x


def get_hyperbolic_encoding_layer_by_name(
    layer_name: str,
    manifold: Manifold,
    input_dim: int,
    output_dim: int,
    params_dtype: str = "float32",
) -> Union[nn.Linear, nn.Sequential, HyperbolicLinearPoincarePP]:
    if isinstance(manifold, Euclidean):
        layer = nn.Linear(input_dim, output_dim)
    # Hyperboloid layers
    elif layer_name == "HyperbolicLinearHyperboloidFHNN" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperboloidEncoder2TangentInput(manifold=manifold),
            HyperbolicLinearHyperboloidFHNN(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="tangent",
                learnable_scale=True,
            ),
        )
    elif layer_name == "HyperbolicLinearHyperboloidFHCNN" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperboloidEncoder2TangentInput(manifold=manifold),
            HyperbolicLinearHyperboloidFHCNN(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="tangent",
                learnable_scale=True,
                normalize=True,
            ),
        )
    elif layer_name == "HyperbolicLinearPP" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperboloidEncoder2TangentInput(manifold=manifold),
            HyperbolicLinearHyperboloidPP(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="tangent",
            ),
        )
    # PoincareBall layers
    elif layer_name == "HyperbolicLinearPP" and isinstance(manifold, PoincareBall):
        layer = HyperbolicLinearPoincarePP(
            manifold=manifold,
            input_dim=input_dim,
            output_dim=output_dim,
            params_dtype=params_dtype,
            input_space="tangent",
        )
        # Fan-in (Kaiming) scaling of the random weight init
        with torch.no_grad():
            layer.weight.div_(math.sqrt(input_dim))
    else:
        raise ValueError(
            f"Unknown hyperbolic layer name: {layer_name}. "
            "Available options are: 'HyperbolicLinearHyperboloidFHNN', "
            "'HyperbolicLinearHyperboloidFHCNN', 'HyperbolicLinearPP'."
        )
    return layer
