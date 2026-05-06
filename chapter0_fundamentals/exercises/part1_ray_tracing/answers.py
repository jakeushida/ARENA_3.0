# %% set up
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import tests
import torch as t
from jaxtyping import Bool, Float
from plotly_utils import imshow
from torch import Tensor
from tqdm import tqdm
from utils import render_lines_with_plotly

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))


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
# %% Exercise - implement `triangle_ray_intersects`
Point = Float[Tensor, "points=3"]


def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    """
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    """
    mat = t.stack([-D, B - A, C - A], dim=1)
    vec = O - A

    try:
        s, u, v = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False

    return ((s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1)).item()
    raise NotImplementedError()


tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %% Exercise - implement `raytrace_triangle`
def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"],
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if the triangle intersects that ray.
    """
    O, D = rays[:, 0, :], rays[:, 1, :]
    A, B, C = einops.repeat(triangle, 'pts dims -> rays pts dims', rays=rays.size(0)).unbind(1)

    mat = t.stack([-D, B - A, C - A], dim=1)
    vec = O - A

    is_singular = mat.det().abs() < 1e-8
    mat[is_singular] = t.eye(3)

    s, u, v = t.linalg.solve(mat, vec).unbind(-1)

    return (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1) & ~is_singular
    raise NotImplementedError()


A = t.tensor([1, 0.0, -0.5])
B = t.tensor([1, -0.5, 0.0])
C = t.tensor([1, 0.5, 0.5])
num_pixels_y = num_pixels_z = 15
y_limit = z_limit = 0.5

# Plot triangle & rays
test_triangle = t.stack([A, B, C], dim=0)
rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
render_lines_with_plotly(rays2d, triangle_lines)

# Calculate and display intersections
intersects = raytrace_triangle(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %% Exercise - implement `raytrace_mesh`
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    """
    O, D = einops.repeat(rays, 'rays pts dims -> pts rays triangles dims', triangles=triangles.size(0))
    A, B, C = einops.repeat(triangles, 'triangles pts dims -> pts rays triangles dims', rays=rays.size(0))

    mat = t.stack([-D, B - A, C - A], dim=2)
    vec = O - A

    is_singular = mat.det().abs() < 1e-8
    mat[is_singular] = t.eye(3)

    s, u, v = t.linalg.solve(mat, vec).unbind(-1)

    intersects = (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1) & ~is_singular

    s *= D[..., 0] # get the x value of the intersection (if it exists)

    s[~intersects] = float('inf')
    
    return einops.reduce(s, 'rays triangles -> rays', 'min')
    raise NotImplementedError()


num_pixels_y = 120
num_pixels_z = 120
y_limit = z_limit = 1

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-2, 0.0, 0.0])
triangles = t.load(section_dir / "pikachu.pt", weights_only=True)
dists = raytrace_mesh(rays, triangles)
intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
dists_square = dists.view(num_pixels_y, num_pixels_z)
img = t.stack([intersects, dists_square], dim=0)

fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
fig.update_layout(coloraxis_showscale=False)
for i, text in enumerate(["Intersects", "Distance"]):
    fig.layout.annotations[i]["text"] = text
fig.show()
# %% Exercise - rotation matrix
def rotation_matrix(theta: Float[Tensor, ""]) -> Float[Tensor, "rows cols"]:
    """
    Creates a rotation matrix representing a counterclockwise rotation of `theta` around the y-axis.
    """
    return t.tensor([[t.cos(theta), 0, t.sin(theta)],
                     [0, 1, 0], 
                     [-t.sin(theta), 0, t.cos(theta)]])
    raise NotImplementedError()


tests.test_rotation_matrix(rotation_matrix)
# %% animation
def raytrace_mesh_video(
    rays: Float[Tensor, "nrays points dim"],
    triangles: Float[Tensor, "ntriangles points dims"],
    rotation_matrix: Callable[[float], Float[Tensor, "rows cols"]],
    raytrace_function: Callable,
    num_frames: int,
) -> Bool[Tensor, "nframes nrays"]:
    """
    Creates a stack of raytracing results, rotating the triangles by `rotation_matrix` each frame.
    """
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_function(rays, triangles))
        t.cuda.empty_cache()  # clears GPU memory (this line will be more important later on!)
    return t.stack(result, dim=0)


def display_video(distances: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is distance to the closest triangle for the i-th frame & the [y, z]-th ray in our 2D
    grid of rays.
    """
    px.imshow(
        distances,
        animation_frame=0,
        origin="lower",
        zmin=0.0,
        zmax=distances[distances.isfinite()].quantile(0.99).item(),
        color_continuous_scale="viridis_r",  # "Brwnyl"
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video").show()


num_pixels_y = 250
num_pixels_z = 250
y_limit = z_limit = 0.8
num_frames = 50

rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
rays[:, 0] = t.tensor([-3.0, 0.0, 0.0])
# %% run animation
dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)

display_video(dists)
# %% check if cuda is available
t.cuda.is_available()
# %% Exercise - use GPUs
def raytrace_mesh_gpu(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"],
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the distance to the closest intersecting triangle, or infinity.

    All computations should be performed on the GPU.
    """
    device = 'cuda'
    rays = rays.to(device)
    triangles = triangles.to(device)
    
    O, D = einops.repeat(rays, 'rays pts dims -> pts rays triangles dims', triangles=triangles.size(0))
    A, B, C = einops.repeat(triangles, 'triangles pts dims -> pts rays triangles dims', rays=rays.size(0))

    mat = t.stack([-D, B - A, C - A], dim=2)
    vec = O - A

    is_singular = mat.det().abs() < 1e-8
    mat[is_singular] = t.eye(3).to(device)

    s, u, v = t.linalg.solve(mat, vec).unbind(-1)

    intersects = (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1) & ~is_singular

    s *= D[..., 0] # get the x value of the intersection (if it exists)

    s[~intersects] = float('inf')
    
    return einops.reduce(s, 'rays triangles -> rays', 'min').cpu()
    raise NotImplementedError()


dists = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_mesh_gpu, num_frames)
dists = einops.rearrange(dists, "frames (y z) -> frames y z", y=num_pixels_y)
display_video(dists)
# %% Exercise (bonus) - add lighting
def raytrace_mesh_lambert(
    rays: Float[Tensor, "nrays points=2 dims=3"],
    triangles: Float[Tensor, "ntriangles points=3 dims=3"],
    light: Float[Tensor, "dims=3"],
    ambient_intensity: float,
    device: str = "cuda" # exercise done on cpu so variable not used,
) -> Float[Tensor, " nrays"]:
    """
    For each ray, return the intensity of light hitting the triangle it intersects with (or zero if
    no intersection).

    Args:
        rays:   A tensor of rays, with shape `[nrays, 2, 3]`.
        triangles:  A tensor of triangles, with shape `[ntriangles, 3, 3]`.
        light:  A tensor representing the light vector, with shape `[3]`. We compute the intensity
                as the dot product of the triangle normals & the light vector, then set it to be
                zero if the sign is negative.
        ambient_intensity:  A float representing the ambient intensity. This is the minimum
                            brightness for a triangle, to differentiate it from the black background
                            (rays that don't hit any triangle).
        device: The device to perform the computation on.

    Returns:
        A tensor of intensities for each of the rays, flattened over the [y, z] dimensions. The
        values are zero when there is no intersection, and `ambient_intensity + intensity` when
        there is an interesection (where `intensity` is the dot product of the triangle's normal
        vector and the light vector, truncated at zero).
    """
    NR, NT = rays.size(0), triangles.size(0)
    
    O, D = einops.repeat(rays, 'rays pts dims -> pts rays triangles dims', triangles=NT)
    
    A, B, C = einops.repeat(triangles, 'triangles pts dims -> pts rays triangles dims', rays=NR)
    assert O.shape == A.shape == (NR, NT, 3)

    mat: Float[Tensor, 'NR NT 3 3'] = t.stack([-D, B - A, C - A], dim=2)
    vec: Float[Tensor, 'NR NT 3'] = O - A

    is_singular: Float[Tensor, 'NR NT'] = mat.det().abs() < 1e-8
    mat[is_singular] = t.eye(3)

    sol: Float[Tensor, 'NR NT 3'] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)
    assert s.shape == (NR, NT)

    intersects: Float[Tensor, 'NR NT'] = (s >= 0) & (u >= 0) & (v >= 0) & ((u + v) <= 1) & ~is_singular

    s *= D[..., 0] # get the x value of the intersection (if it exists)

    s[~intersects] = float('inf')

    closest_triangles_dist, closest_triangles_idx = s.min(1)

    cross: Float[Tensor, 'NT 3'] = t.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    
    normal = cross / cross.norm(keepdim=True)
    
    dot: Float[Tensor, 'NT'] = einops.einsum(normal, light, 'NT dims, dims -> NT')
    dot_non_negative = t.where(dot > 0, dot, 0)
    
    intensity = dot_non_negative[closest_triangles_idx] + ambient_intensity
    intensity = t.where(closest_triangles_dist.isfinite(), intensity, 0)

    return intensity
    raise NotImplementedError()


def display_video_with_lighting(intensity: Float[Tensor, "frames y z"]):
    """
    Displays video of raytracing results, using Plotly. `distances` is a tensor where the [i, y, z]
    element is the lighting intensity based on the angle of light & the surface of the triangle
    which this ray hits first.
    """
    px.imshow(
        intensity,
        animation_frame=0,
        origin="lower",
        color_continuous_scale="magma",
    ).update_layout(coloraxis_showscale=False, width=550, height=600, title="Raytrace mesh video (lighting)").show()


ambient_intensity = 0.5
light = t.tensor([0.0, -1.0, 1.0])
raytrace_function = partial(
    raytrace_mesh_lambert,
    ambient_intensity=ambient_intensity,
    light=light,
)

intensity = raytrace_mesh_video(rays, triangles, rotation_matrix, raytrace_function, num_frames)
intensity = einops.rearrange(intensity, "frames (y z) -> frames y z", y=num_pixels_y)
display_video_with_lighting(intensity)
# %%
