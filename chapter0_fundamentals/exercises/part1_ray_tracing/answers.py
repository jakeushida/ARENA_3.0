# %% set up
import einops
import tests
import torch as t
from jaxtyping import Bool, Float
from torch import Tensor
from utils import render_lines_with_plotly


# %% Exercise - implement `make_rays_1d`
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    rays = t.zeros(num_pixels, 2, 3)
    rays[:, 1, 0] = 1
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    return rays
    raise NotImplementedError()

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
# %% Exercise - implement `intersect_ray_1d`
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray[0, :2], ray[1, :2]
    L1, L2 = segment[0, :2], segment[1, :2]
    
    mat = t.stack([D, L1 - L2], dim=1)
    vec = L1 - O
    
    try:
        u, v = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False
    
    return (u >= 0) and (0 <= v <= 1)
    raise NotImplementedError()


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
# %% Exercise - implement `intersect_rays_1d`
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    rays = einops.repeat(rays, 'rays points coords -> rays segs points coords', segs=segments.size(0))
    segments = einops.repeat(segments, 'segs points coords -> rays segs points coords', rays=rays.size(0))

    O, D = rays[..., 0, :2], rays[..., 1, :2]
    L1, L2 = segments[..., 0, :2], segments[..., 1, :2]

    mat = t.stack([D, L1 - L2], dim=-1)
    vec = L1 - O

    is_singular = mat.det().abs() < 1e-8
    mat[is_singular] = t.eye(2)

    u, v = t.linalg.solve(mat, vec).unbind(-1)

    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)
    raise NotImplementedError()


tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %% Exercise - implement `make_rays_2d`
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    """
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    """
    rays = t.zeros(num_pixels_y * num_pixels_z, 2, 3)
    
    y_vals = t.linspace(-y_limit, y_limit, num_pixels_y)
    z_vals = t.linspace(-z_limit, z_limit, num_pixels_z)

    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(y_vals, 'ypix -> (ypix zpix)', zpix=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(z_vals, 'zpix -> (ypix zpix)', ypix=num_pixels_y)

    return rays
    raise NotImplementedError()


rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
render_lines_with_plotly(rays_2d)
# %%
