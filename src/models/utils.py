import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Union
from src.hyperbolic_math.src.manifolds import Manifold, Euclidean, Hyperboloid, PoincareBall
from src.hyperbolic_math.src.nn_layers import (
    HyperbolicLinearPoincarePP,
    HyperbolicLinearHyperboloidFHNN,
    HyperbolicLinearHyperboloidFHCNN,
    HyperbolicLinearHyperboloidPP,
    Expmap_0,
    Logmap_0
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


class HyperboloidTangent2DecoderInput(torch.nn.Module):
    """Convert tangent vectors at the Hyperboloid origin to inputs of the Euclidean decoder."""

    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.manifold, Hyperboloid):
            return x[..., 1:]
        else:
            return x


class DtypeConversion(torch.nn.Module):
    """Convert tensor dtype."""

    def __init__(self, target_dtype: torch.dtype):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=self.target_dtype)


def get_hyperbolic_encoding_layer_by_name(
    layer_name: str,
    manifold: Manifold,
    input_dim: int,
    output_dim: int,
    params_dtype: str = "float32",
    clamping_factor: float = 1.0,
    smoothing_factor: float = 50.0,
    learnable_scale: bool = False,
    FHCNN_normalize: bool = True,
) -> Union[nn.Sequential, HyperbolicLinearPoincarePP]:
    if isinstance(manifold, Euclidean):
        layer = nn.Linear(input_dim, output_dim)
    elif layer_name == "EuclideanLinear_Exp" and not isinstance(manifold, Euclidean):
        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            HyperboloidEncoder2TangentInput(manifold=manifold),
            Expmap_0(manifold=manifold),
        )
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
                learnable_scale=learnable_scale,
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
                learnable_scale=learnable_scale,
                normalize=FHCNN_normalize,
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
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor,
            ),
        )
    # PoincareBall layers
    elif layer_name == "HyperbolicLinearPP" and isinstance(manifold, PoincareBall):
        # Kaiming scaling is handled inside the layer's __init__.
        layer = HyperbolicLinearPoincarePP(
            manifold=manifold,
            input_dim=input_dim,
            output_dim=output_dim,
            params_dtype=params_dtype,
            input_space="tangent",
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor,
        )
    else:
        raise ValueError(
            f"Unknown hyperbolic layer name: {layer_name}. "
            "Available options are: 'EuclideanLinear_Exp', "
            "'HyperbolicLinearHyperboloidFHNN', 'HyperbolicLinearHyperboloidFHCNN', 'HyperbolicLinearPP'."
        )
    return layer


def get_hyperbolic_decoding_layer_by_name(
    layer_name: str,
    manifold: Manifold,
    input_dim: int,
    output_dim: int,
    params_dtype: str = "float32",
    clamping_factor: float = 1.0,
    smoothing_factor: float = 50.0,
    activation_function: nn.Module = nn.SiLU(),
    learnable_scale: bool = False,
    FHCNN_normalize: bool = True,
) -> nn.Sequential:
    if isinstance(manifold, Euclidean):
        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
    elif layer_name == "Log_EuclideanLinear" and not isinstance(manifold, Euclidean):
        layer = nn.Sequential(
            Logmap_0(manifold=manifold),
            HyperboloidTangent2DecoderInput(manifold=manifold),
            DtypeConversion(target_dtype=get_dtype_from_precision(params_dtype)),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
    # Hyperboloid layers
    elif layer_name == "HyperbolicLinearHyperboloidFHNN_Log" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperbolicLinearHyperboloidFHNN(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="manifold",
                learnable_scale=learnable_scale,
            ),
            Logmap_0(manifold=manifold),
            HyperboloidTangent2DecoderInput(manifold=manifold),
            DtypeConversion(target_dtype=get_dtype_from_precision(params_dtype)),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
    elif layer_name == "HyperbolicLinearHyperboloidFHCNN_Log" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperbolicLinearHyperboloidFHCNN(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="manifold",
                learnable_scale=learnable_scale,
                normalize=FHCNN_normalize,
            ),
            Logmap_0(manifold=manifold),
            HyperboloidTangent2DecoderInput(manifold=manifold),
            DtypeConversion(target_dtype=get_dtype_from_precision(params_dtype)),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
    elif layer_name == "HyperbolicLinearPP_Log" and isinstance(manifold, Hyperboloid):
        layer = nn.Sequential(
            HyperbolicLinearHyperboloidPP(
                manifold=manifold,
                input_dim=input_dim + 1,
                output_dim=output_dim + 1,
                params_dtype=params_dtype,
                input_space="manifold",
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor,
            ),
            Logmap_0(manifold=manifold),
            HyperboloidTangent2DecoderInput(manifold=manifold),
            DtypeConversion(target_dtype=get_dtype_from_precision(params_dtype)),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
    # PoincareBall layers
    elif layer_name == "HyperbolicLinearPP_Log" and isinstance(manifold, PoincareBall):
        layer = nn.Sequential(
            HyperbolicLinearPoincarePP(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space="manifold",
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor,
            ),
            Logmap_0(manifold=manifold),
            DtypeConversion(target_dtype=get_dtype_from_precision(params_dtype)),
            nn.BatchNorm1d(output_dim),
            activation_function,
        )
        layer[0].weight.data *= (1.0 / np.sqrt(input_dim))
    else:
        raise ValueError(
            f"Unknown hyperbolic layer name: {layer_name}. "
            "Available options are: 'Log_EuclideanLinear', "
            "'HyperbolicLinearHyperboloidFHNN_Log', 'HyperbolicLinearHyperboloidFHCNN_Log', "
            "'HyperbolicLinearPP_Log'."
        )
    return layer
