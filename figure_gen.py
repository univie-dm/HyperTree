import torch
import os
from src.manifolds import PoincareBall, Hyperboloid
import matplotlib.pyplot as plt

points = torch.tensor([
    [0.0, 0.0], # origin
    [0.5, 0.0], # right branch root
    [0.83, -0.25], # right branch bot
    [0.83, 0.25], # right branch top
    [-0.25, 3**0.5/4], # middle branch root
    [(-83+25*3**0.5)/200, (25+83*3**0.5)/200], # middle branch top
    [-(25*3**0.5+83)/200, (-25+83*3**0.5)/200], # middle branch bot
    [-0.25, -(3**0.5)/4], # left branch root
    [-(25*3**0.5+83)/200, -(-25+83*3**0.5)/200] # left branch child ## free placement
    ], dtype=torch.float64)

def compute_lca_depth_poincare(manifold, points: torch.Tensor, batch_size: int=1_000, version: str="ours") -> torch.Tensor:
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
        return lca
        #lca_depth = manifold.dist_0(lca)

        if version == "ours":
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


##### Plotting #####
settings = {
    "title": "Hyperbolic Embeddings",
    "file_name": "lca_paper_plot",
    "file_format": "svg",
}

poincare = PoincareBall(c=torch.tensor([1], dtype=torch.float64), dtype=torch.float64)

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect('equal')
poincare_closure = 1 / poincare.c.sqrt().cpu().detach()
ax.set_xlim(-1.1 * poincare_closure, 1.1 * poincare_closure)
ax.set_ylim(-1.1 * poincare_closure, 1.1 * poincare_closure)

# Draw the closure of the PoincareBall
circle = plt.Circle((0, 0), poincare_closure, fill=False, color='black', zorder=5)
ax.add_artist(circle)

points = points.detach()
ax.set_title(f"{settings['title']}")

# Get the old LCA
old_lca = compute_lca_depth_poincare(poincare, points[-2:], version="chami")
print(old_lca)

# Plot the points in 2D
x = points.cpu()
ax.scatter(x[:, 0], x[:, 1], c='green', alpha=0.6, zorder=2)
ax.scatter(old_lca[0, 0], old_lca[0, 1], c='red', alpha=0.6, zorder=2)
handles = []

# Plot geodesics between specified points
edges = ([0,1,1,0,4,4,0,7], [1,2,3,4,5,6,7,8])
spacing = 100
t = torch.linspace(0, 1.0, spacing, device=poincare.c.device).reshape(-1, 1)
x = points[edges[0]]
y = points[edges[1]]
dir = poincare.addition(-x, y)
for _x, _dir in zip(x, dir):
    # Compute points on the geodesic segment connecting x and y
    second_term = poincare.scalar_mul(t, _dir.repeat(spacing, 1), backproject=True)
    geodesic = poincare.addition(_x.repeat(spacing, 1), second_term, backproject=True)
    geodesic = geodesic.cpu()
    ax.plot(geodesic[:, 0], geodesic[:, 1], c='dimgrey', zorder=4, linewidth=1.5)
# Compute points on full geodesic connecting the last two points
s = torch.linspace(-5.0, 5.0, spacing, device=poincare.c.device).reshape(-1, 1)
second_term = poincare.scalar_mul(s, dir[-1:].repeat(spacing, 1), backproject=True)
geodesic = poincare.addition(x[-1:].repeat(spacing, 1), second_term, backproject=True)
geodesic = geodesic.cpu()
ax.plot(geodesic[:, 0], geodesic[:, 1], c='dimgrey', linestyle='--', zorder=3, linewidth=1.5)
handles.append(plt.Line2D([0], [0], color='dimgrey', label='Geodesic'))

if handles:
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

file_path = os.getcwd()
path = os.path.join(file_path)
os.makedirs(path, exist_ok=True)
fig.savefig(os.path.join(path, f"{settings['file_name']}.{settings['file_format']}"))
plt.close(fig)
