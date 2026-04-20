import random
from pathlib import Path

import h5py
import numpy as np
import torch


def get_param(params, name, default=None):
    return params.get(name, default) if hasattr(params, "get") else getattr(params, name, default)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path, base_dir=None):
    path = Path(path)
    if path.is_absolute():
        return path
    if base_dir is None:
        base_dir = Path.cwd()
    return (Path(base_dir) / path).resolve()


def load_mesh(mesh_location):
    """Load an HDF5 mesh stored at ``mesh/coordinates``."""
    with h5py.File(mesh_location, "r") as h5f:
        return np.asarray(h5f["mesh/coordinates"], dtype=np.float32)


def prepare_model_input(x, static_features, params):
    """Append static inlet and mesh-coordinate channels to each time step."""
    n_static = int(get_param(params, "n_static_channels", 0) or 0)
    if n_static <= 0:
        return x

    static = static_features[..., :n_static].to(device=x.device, dtype=x.dtype)
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape (batch, time, nodes, channels), got {tuple(x.shape)}")
    static = static.unsqueeze(1).expand(x.shape[0], x.shape[1], -1, -1)
    return torch.cat([x, static], dim=-1)


def get_grid_displacement(params, batch):
    if get_param(params, "grid_type") != "non uniform":
        return None, None
    in_grid = batch["d_grid_x"][0]
    out_grid = batch["d_grid_y"][0]
    return out_grid, in_grid


def fixed_boundary_indices(mesh, include_inlet_outlet=False, tol=1e-6):
    """Return fixed Turek-Hron channel/cylinder boundary node indices."""
    mesh = torch.as_tensor(mesh)
    indices = []
    length = 2.5
    height = 0.41
    cx, cy, radius = 0.2, 0.2, 0.05

    for idx, point in enumerate(mesh):
        x = float(point[0])
        y = float(point[1])
        on_cylinder = abs(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 - radius) < 1e-3
        on_top_bottom = abs(y) < tol or abs(y - height) < tol
        on_ends = abs(x) < tol or abs(x - length) < tol
        if on_cylinder or on_top_bottom or (include_inlet_outlet and on_ends):
            indices.append(idx)
    return torch.tensor(indices, dtype=torch.long)
