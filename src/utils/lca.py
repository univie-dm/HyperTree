import torch

from src.hyperbolic_math.src.manifolds import Manifold, Hyperboloid, PoincareBall


def compute_lca_depth(
    points: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    manifold: Manifold,
    batch_size: int = 1_000_000,
    version: str = 'rectified',
    dist_version: str | None = None,
) -> torch.Tensor:
    """Numerically stabilized LCA depth computation for the point pairs (points[row_idx], points[col_idx]).

    'rectified' constrains the LCA to the geodesic segment between the two points, 'unrectified'
    returns the depth of the orthogonal projection of the origin onto the full geodesic.
    The depth is the distance to the origin in the given dist_version of the manifold
    (None: the version the manifold was created with, e.g. 'smoothened' for training).
    """
    if version not in ["rectified", "unrectified"]:
        raise ValueError(f"Invalid LCA version: {version}. Supported versions are: 'rectified', 'unrectified'.")
    c: torch.Tensor = manifold.c  # type: ignore
    lca_depths = []

    # Pre-compute point norms
    if version == 'rectified':
        point_norms = points.norm(p=2, dim=-1, keepdim=True)
        point_dists_0 = manifold.dist_0(points, version=dist_version)

    # Process the pairs in batches
    total_pairs = row_idx.shape[0]
    for start_idx in range(0, total_pairs, batch_size):
        end_idx = min(start_idx + batch_size, total_pairs)
        batch_row = row_idx[start_idx:end_idx]
        batch_col = col_idx[start_idx:end_idx]

        x = points[batch_row]
        y = points[batch_col]
        xy = (x * y).sum(dim=-1, keepdim=True)

        if isinstance(manifold, Hyperboloid):
            # Compute the orthogonal geodesic projection(s) onto the Minkowski subspace
            sqrt_c = c.sqrt()
            x0 = x.narrow(-1, 0, 1)
            y0 = y.narrow(-1, 0, 1)
            xy_mink = manifold._minkowski_inner(x, y)
            denom = (1 - (c * xy_mink).pow(2)).clamp(max=-1e-15)
            alpha = (sqrt_c * (x0 + c * xy_mink * y0)) / denom
            beta = (sqrt_c * (y0 + c * xy_mink * x0)) / denom
            z = alpha * x + beta * y

            # Rescale the vector to lie on the Hyperboloid
            z_mink = manifold._minkowski_inner(z, z)
            lca = z / (-c * z_mink).clamp(min=1e-15).sqrt()
        elif isinstance(manifold, PoincareBall):
            # Compute the center of geodesic circle Gamma through x and y
            x2 = x.pow(2).sum(dim=-1, keepdim=True)
            y2 = y.pow(2).sum(dim=-1, keepdim=True)

            denom = (2 * c * (x2 * y2 - xy.pow(2))).clamp(min=1e-15)
            alpha = ((1 + c * x2) * y2 - (1 + c * y2) * xy) / denom
            beta = ((1 + c * y2) * x2 - (1 + c * x2) * xy) / denom
            z = alpha * x + beta * y

            # Compute the lca and its distance to the origin w.r.t. the full geodesic
            z2 = z.pow(2).sum(dim=-1, keepdim=True)
            z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

            # Computation via conjugation to avoid numerical issues
            sqrt_term = (z2 - 1 / c).clamp(min=1e-15).sqrt()
            denom = c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
            lca = z / denom
        else:
            raise ValueError(f"Unsupported manifold: {manifold}")

        lca_depth = manifold.dist_0(lca, version=dist_version)

        # Apply LCA rectification
        if version == 'rectified':
            # Use pre-computed norms
            x_norm = point_norms[batch_row]
            y_norm = point_norms[batch_col]

            # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
            xlca = (x * lca).sum(dim=-1, keepdim=True)
            ylca = (y * lca).sum(dim=-1, keepdim=True)
            lca_norm = lca.norm(p=2, dim=-1, keepdim=True)

            cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)
            cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)
            cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(-1.0, 1.0)

            # Get the lca depth w.r.t. the geodesic segment between x and y
            condition = (cos_x0y > cos_x0c) | (cos_x0y > cos_y0c) | (cos_x0y > 1-1e-15)  # True iff lca lies outside
            lca_depth = torch.where(condition, torch.min(point_dists_0[batch_row], point_dists_0[batch_col]), lca_depth)
            lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth)  # Handle antipodal points

        lca_depths.append(lca_depth.reshape(-1))

    if not lca_depths:
        return torch.zeros(0, dtype=points.dtype, device=points.device)
    return torch.cat(lca_depths, dim=0)


def compute_pairwise_lca_depths(
    points: torch.Tensor,
    manifold: Manifold,
    batch_size: int = 1_000_000,
    version: str = 'rectified',
    dist_version: str | None = None,
) -> torch.Tensor:
    """LCA depths for all pairs of points in the given tensor, as a symmetric matrix with zero diagonal."""
    n = points.shape[0]
    row_idx, col_idx = torch.triu_indices(n, n, 1, device=points.device)
    lca_depths = compute_lca_depth(
        points, row_idx, col_idx, manifold, batch_size=batch_size, version=version, dist_version=dist_version
    )
    lcas = torch.zeros((n, n), dtype=points.dtype, device=points.device)
    lcas[row_idx, col_idx] = lca_depths
    lcas[col_idx, row_idx] = lca_depths
    return lcas
