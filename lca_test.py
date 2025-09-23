import torch
from src.utils.math_utils import sinh, asinh, tanh, cosh
from src.manifolds import Hyperboloid, PoincareBall
from src.utils.vis_utils import create_figure
from matplotlib import pyplot as plt
import os
from icecream import ic
import time


def compute_lca_depth_hyperboloid(manifold, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    """Generalized, corrected, and numerically stabilized implementation of
        Chami et al.'s (2020) LCA depth computation using inverse geometry."""
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        # Compute the orthogonal geodesic projection(s) [z B A^T (A B A^T)^-1 A] of the Hyperboloid's origin z
        # onto all 2d geodesic submanifold(s) spanned by all pair-wise combination of vectors x and y.
        # That is, A is the batched tensor of (2, dim) matrices whose rows span on of the 2d submanifold we
        # project onto, and B is the Minkowski inner product matrix
        # Construct the matrix A
        A = torch.stack([x, y], dim=1) # (B, 2, dim)
        # 1) Compute the coefficients [z_coeffs = z B A^T (A B A^T)^-1] of the projection
        z = manifold._create_origin_from_reference(x) # (B, d)
        zBAt = manifold._minkowski_inner(z.unsqueeze(-1), A.transpose(-2, -1), axis=1).squeeze(1) # (B, 2)
        ABAt = manifold._minkowski_inner(A.unsqueeze(-1), A.transpose(-2, -1).unsqueeze(1), axis=2).squeeze(2) # (B, 2, 2)
        z_coeffs = torch.linalg.solve(ABAt, zBAt) # (B, 2)
        # 2) Compute the orthogonal geodesic projection onto the submanifold(s)
        mink_proj = (z_coeffs.unsqueeze(1) @ A).squeeze(1) # (B, d)
        mink_proj_normalized = (-manifold.c * manifold._minkowski_inner(mink_proj, mink_proj)).sqrt() # (B, d)
        lca = mink_proj / mink_proj_normalized # (B, d)
        lca_depth = manifold.dist_0(lca)

        # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
        xlca = (x * lca).sum(dim=-1, keepdim=True)
        ylca = (y * lca).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        lca_norm = lca.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-10) # see if this clamping is necessary

        x_norm = x.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        y_norm = y.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)

        # Get the lca depth w.r.t. the geodesic segment between x and y
        condition = torch.logical_or(torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1) # True iff the lca lies outside
        lca_depth = torch.where(condition, torch.min(manifold.dist_0(x), manifold.dist_0(y)), lca_depth)
        lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth) # Handle antipodal points

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]
    return lcas


def compute_lca_depth_hyperboloid_new(manifold, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    """Generalized, corrected, and numerically stabilized implementation of
        Chami et al.'s (2020) LCA depth computation using inverse geometry."""
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :] # (B, d)
        y = points[indices[1,:batch_size], :] # (B, d)

        xy_mink = manifold._minkowski_inner(x, y) # (B, 1)
        sqrt_c = manifold.c.sqrt()
        x0 = x.narrow(-1, 0, 1)
        y0 = y.narrow(-1, 0, 1)
        denom = 1 - (manifold.c * xy_mink).pow(2).clamp(min=1e-15) # (B, 1)

        alpha = (sqrt_c * (x0 + manifold.c * xy_mink * y0)) / denom # (B, 1)
        beta = (sqrt_c * (y0 + manifold.c * xy_mink * x0)) / denom # (B, 1)

        z = alpha * x + beta * y # (B, d)
        z_mink = manifold._minkowski_inner(z, z) # (B, 1)
        lca = z / (-manifold.c * z_mink).sqrt() # (B, d)
        lca_depth = manifold.dist_0(lca)

        # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
        xlca = (x * lca).sum(dim=-1, keepdim=True)
        ylca = (y * lca).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        lca_norm = lca.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-10) # see if this clamping is necessary

        x_norm = x.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        y_norm = y.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)

        # Get the lca depth w.r.t. the geodesic segment between x and y
        print(cos_x0y)
        condition = torch.logical_or(torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1) # True iff the lca lies outside
        lca_depth = torch.where(condition, torch.min(manifold.dist_0(x), manifold.dist_0(y)), lca_depth)
        lca_depth = torch.where(cos_x0y < -1+1e-04, torch.zeros_like(lca_depth), lca_depth) # Handle antipodal points

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]
    return lcas


def compute_lca_depth_poincare(manifold, points: torch.Tensor, batch_size: int=1_000) -> torch.Tensor:
    """Generalized, corrected, and numerically stabilized implementation of
        Chami et al.'s (2020) LCA depth computation using inverse geometry."""
    device = points.device
    lcas = torch.zeros((points.shape[0], points.shape[0]), dtype=points.dtype).to(device)
    indices = torch.triu_indices(points.shape[0], points.shape[0], 1).to(device)
    while indices.shape[1] > 0:
        x = points[indices[0,:batch_size], :]
        y = points[indices[1,:batch_size], :]

        # Compute the center of geodesic circle Gamma through x and y
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        denom = (2 * manifold.c * (x2 * y2 - xy ** 2)).clamp(min=1e-15)
        alpha = ((1 + manifold.c * x2) * y2 - (1 + manifold.c * y2) * xy) / denom
        beta = ((1 + manifold.c * y2) * x2 - (1 + manifold.c * x2) * xy) / denom
        z = alpha * x + beta * y

        # Compute the lca and its distance to the origin w.r.t. the full geodesic
        z2 = z.pow(2).sum(dim=-1, keepdim=True)
        z_norm = z.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-15)

        # Direct computation of the lca
        # second_term = (z2 - 1/manifold.c).clamp(min=1e-15).sqrt()
        # lca = (1 - second_term / z_norm) * z

        # Computation via conjugation to avoid numerical issues
        sqrt_term = (z2 - 1 / manifold.c).clamp(min=1e-15).sqrt()
        denom = manifold.c * z_norm * (z_norm + sqrt_term).clamp(min=1e-15)
        lca = z / denom
        lca_depth = manifold.dist_0(lca)

        # Compute the cosine of the angles at the origin in the triangles {x, origin, c}, {y, origin, c}  {x, origin, y}
        xlca = (x * lca).sum(dim=-1, keepdim=True)
        ylca = (y * lca).sum(dim=-1, keepdim=True)
        lca_norm = lca.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-10) # see if this clamping is necessary

        x_norm = x.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        y_norm = y.norm(p=2, dim=-1, keepdim=True)#.clamp(min=1e-15) # see if this clamping is necessary
        cos_x0c = (xlca / (x_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_y0c = (ylca / (y_norm * lca_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)
        cos_x0y = (xy / (x_norm * y_norm).clamp(min=1e-15)).clamp(min=-1.0, max=1.0)

        # Get the lca depth w.r.t. the geodesic segment between x and y
        condition = torch.logical_or(torch.logical_or(cos_x0y > cos_x0c, cos_x0y > cos_y0c), cos_x0y == 1) # True iff the lca lies outside
        lca_depth = torch.where(condition, torch.min(manifold.dist_0(x), manifold.dist_0(y)), lca_depth)
        lca_depth = torch.where(cos_x0y == -1, torch.zeros_like(lca_depth), lca_depth) # Handle antipodal points

        lca_depth = lca_depth.reshape(-1)
        lcas[indices[0,:batch_size], indices[1,:batch_size]] = lca_depth
        lcas[indices[1,:batch_size], indices[0,:batch_size]] = lca_depth
        indices = indices[:, batch_size:]
    return lcas


def plot_routine(points, res_ours, res_chami, manifold, printing=True):
    if printing:
        print("---------------------")
        print(f"Res_ours: {res_ours} \nRes_chami: {res_chami}")

    x_dist_str = f"{manifold.dist_0(points[0]).item():.2f}"
    y_dist_str = f"{manifold.dist_0(points[1]).item():.2f}"
    fig = create_figure(points, manifold,
                        #labels=[f'x: {x_dist_str}', f'y: {y_dist_str}'],
                        edges=([0], [1])
                        )

    ax = fig.gca()
    res_ours_converted = tanh(manifold.c.sqrt() * res_ours / 2) / manifold.c.sqrt()
    res_chami_converted = tanh(manifold.c.sqrt() * res_chami / 2) / manifold.c.sqrt()

    # Handle both scalar and matrix inputs
    if hasattr(res_ours, 'shape') and len(res_ours.shape) > 0:
        # Matrix input - iterate over all pairs
        for i in range(res_ours.shape[0]):
            for j in range(i+1, res_ours.shape[1]):
                circle_ours = plt.Circle((0, 0), res_ours_converted[i,j].cpu().numpy(), fill=False, color='green', zorder=5, linewidth=0.5)
                circle_chami = plt.Circle((0, 0), res_chami_converted[i,j].cpu().numpy(), fill=False, color='red', zorder=6, linestyle='--', linewidth=0.5)
                ax.add_artist(circle_ours)
                ax.add_artist(circle_chami)
    else:
        # Scalar input - just draw one circle for each
        circle_ours = plt.Circle((0, 0), res_ours_converted.cpu().numpy() if hasattr(res_ours_converted, 'cpu') else res_ours_converted, fill=False, color='green', zorder=5, linewidth=0.5)
        circle_chami = plt.Circle((0, 0), res_chami_converted.cpu().numpy() if hasattr(res_chami_converted, 'cpu') else res_chami_converted, fill=False, color='red', zorder=6, linestyle='--', linewidth=0.5)
        ax.add_artist(circle_ours)
        ax.add_artist(circle_chami)

    # Save the figure
    file_path = os.getcwd()
    path = os.path.join(file_path, 'images')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, "hyperbolic_embeddings.png"))
    plt.close(fig)



##########
dtype = torch.float32
version_1 = "compute_lca_depth_hyperboloid"
version_2 = "compute_lca_depth_hyperboloid_new"
curvature = 1.0
##########

poincare = PoincareBall(c=torch.tensor([curvature], dtype=dtype))
hyperboloid = Hyperboloid(c=torch.tensor([curvature], dtype=dtype))

opposites = torch.tensor([[0.5, 0.0], [-0.6, 0.0]], dtype=dtype) # res = 0
same_side_x = torch.tensor([[0.5, 0.0], [0.7, 0.0]], dtype=dtype) # res = x
same_side_y = torch.tensor([[0.7, 0.0], [0.5, 0.0]], dtype=dtype) # res = y

opposites_flipped = torch.tensor([[-0.6, 0.0], [0.5, 0.0]], dtype=dtype) # res = 0
same_side_y_flipped = torch.tensor([[0.7, 0.0], [0.5, 0.0]], dtype=dtype) # res = y
same_side_x_flipped = torch.tensor([[0.5, 0.0], [0.7, 0.0]], dtype=dtype) # res = x

p1 = torch.tensor([[ 0.0706, -0.9943],
        [ 0.8838,  0.0425]], dtype=dtype)

p2 = torch.tensor([[ -0.1, -0.5],
        [ -0.4,  -0.8]], dtype=dtype)

p3 = torch.tensor([[ 0.3, 0.],
        [ 0.2,  -0.007]], dtype=dtype)

##########
# random_dirs = torch.normal(0, 1, size=(20, 2), dtype=dtype)
# random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
# random_radii = torch.rand((20, 1), dtype=dtype).pow(1 / 2)
# points_poincare = poincare.c**-0.5 * (random_dirs * random_radii)

points_poincare = opposites

points_hyperbol = poincare.to_hyperboloid(points_poincare)
##########


res_hyper = compute_lca_depth_hyperboloid(hyperboloid, points_hyperbol)
res_hyper_new = compute_lca_depth_hyperboloid_new(hyperboloid, points_hyperbol)
plot_routine(points_poincare, res_hyper, res_hyper_new, poincare, printing=True)
#exit()

torch.manual_seed(48)
count = 0
for i in range(200):
    num_pts = 500
    dim = 2
    random_dirs = torch.normal(0, 1, size=(num_pts, dim), dtype=dtype)
    random_dirs /= random_dirs.norm(p=2, dim=-1, keepdim=True)
    random_radii = torch.rand((num_pts, 1), dtype=dtype).pow(1 / 2)
    points_poincare = poincare.c**-0.5 * (random_dirs * random_radii)
    points_hyperbol = poincare.to_hyperboloid(points_poincare)

    res_hyper = compute_lca_depth_hyperboloid(hyperboloid, points_hyperbol)
    res_hyper_new = compute_lca_depth_hyperboloid_new(hyperboloid, points_hyperbol)

    if not torch.allclose(res_hyper, res_hyper_new, atol=1e-4):
        print(res_hyper.shape)
        print("DIFFERENT")
        print(f"diff: {torch.abs(res_hyper - res_hyper_new).max()}")
        idx = torch.argmax(torch.abs(res_hyper - res_hyper_new), keepdim=True)
        idx_row, idx_col = divmod(idx.item(), res_hyper.shape[1])
        print(idx.shape)
        plot_routine(points_poincare[[idx_row, idx_col]], res_hyper[idx_row, idx_col], res_hyper_new[idx_row, idx_col], poincare)
        print(i)
        count += 1

        #break
        #time.sleep(10)
print("count:", count)
