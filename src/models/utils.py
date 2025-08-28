import torch
import torch.nn as nn

from typing import Dict, Union
from src.hyperbolic_math.src.manifolds import Manifold, PoincareBall
from src.hyperbolic_math.src.nn_layers import *


def get_dtype_from_precision(precision: str) -> torch.dtype:
    # Dictionary mapping of dtype strings to torch dtypes
    DTYPE_MAP: Dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if precision not in DTYPE_MAP:
        raise ValueError(f"Unsupported precision: {precision}. Supported precisions are: {list(DTYPE_MAP.keys())}.")
    return DTYPE_MAP[precision]

def get_hyperbolic_encoding_layer_by_name(
    layer_name: str,
    manifold: Manifold,
    input_dim: int,
    output_dim: int,
    params_dtype: str = 'float32',
    clamping_factor: float = 1.0,
    smoothing_factor: float = 50.0
) -> Union[nn.Sequential, HyperbolicLinearPoincare, HyperbolicLinearPoincarePP]:
    if layer_name =='EuclideanLinear_Exp':
        layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
                Expmap_0(manifold=manifold)
            )
    elif layer_name == 'HyperbolicLinear' and isinstance(manifold, PoincareBall):
        layer = HyperbolicLinearPoincare(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space='tangent')
    elif layer_name == 'HyperbolicLinearPP' and isinstance(manifold, PoincareBall):
        layer = HyperbolicLinearPoincarePP(
            manifold=manifold,
            input_dim=input_dim,
            output_dim=output_dim,
            params_dtype=params_dtype,
            input_space='tangent',
            clamping_factor=clamping_factor,
            smoothing_factor=smoothing_factor)
    else:
        raise ValueError(f"Unknown hyperbolic layer name: {layer_name}. "
                         "Available options are: 'EuclideanLinear_Exp', 'HyperbolicLinear', 'HyperbolicLinearPP'.")
    return layer

def get_hyperbolic_decoding_layer_by_name(
    layer_name: str,
    manifold: Manifold,
    input_dim: int,
    output_dim: int,
    params_dtype: str = 'float32',
    clamping_factor: float = 1.0,
    smoothing_factor: float = 50.0
) -> nn.Sequential:
    if layer_name == 'Log_EuclideanLinear':
        layer = nn.Sequential(
            Logmap_0(manifold=manifold),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    elif layer_name == 'HyperbolicLinear_Log' and isinstance(manifold, PoincareBall):
        layer = nn.Sequential(
            HyperbolicLinearPoincare(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space='manifold'),
            Logmap_0(manifold=manifold),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    elif layer_name == 'HyperbolicLinearPP_Log' and isinstance(manifold, PoincareBall):
        layer = nn.Sequential(
            HyperbolicLinearPoincarePP(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space='manifold',
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor),
            Logmap_0(manifold=manifold),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    elif layer_name == 'HyperbolicRegression' and isinstance(manifold, PoincareBall):
        layer = nn.Sequential(
            HyperbolicRegressionPoincare(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space='manifold',
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    elif layer_name == 'HyperbolicRegressionPP' and isinstance(manifold, PoincareBall):
        layer = nn.Sequential(
            HyperbolicRegressionPoincarePP(
                manifold=manifold,
                input_dim=input_dim,
                output_dim=output_dim,
                params_dtype=params_dtype,
                input_space='manifold',
                clamping_factor=clamping_factor,
                smoothing_factor=smoothing_factor),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    else:
        raise ValueError(f"Unknown hyperbolic layer name: {layer_name}. "
                         "Available options are: 'Log_EuclideanLinear', 'HyperbolicLinear_Log', 'HyperbolicLinearPP_Log', "
                         "'HyperbolicRegression', 'HyperbolicRegressionPP'.")
    return layer
