import enum

import numpy as np
import torch

from models.gnn import GNN
from models.vit import VitGno
from utils import get_param, load_mesh, resolve_path


class StageEnum(enum.Enum):
    PREDICTIVE = "PREDICTIVE"


def regular_grid_from_mesh(mesh, grid_size, device):
    minx, maxx = np.min(mesh[:, 0]), np.max(mesh[:, 0])
    miny, maxy = np.min(mesh[:, 1]), np.max(mesh[:, 1])
    size_x, size_y = grid_size
    idx_x = torch.linspace(float(minx), float(maxx), int(size_x), device=device)
    idx_y = torch.linspace(float(miny), float(maxy), int(size_y), device=device)
    x, y = torch.meshgrid(idx_x, idx_y, indexing="ij")
    return torch.stack([x.flatten(), y.flatten()], dim=-1).float()


def build_fluid_model(params, device=None):
    """Build the fluid surrogate used in the paper.

    Supported ``nettype`` values:
    - ``vit``: GNO-ViT-GNO, the proposed fluid model.
    - ``gnn``: pure GNO/GNN baseline for the ViT ablation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh = load_mesh(resolve_path(get_param(params, "input_mesh_location")))
    input_mesh = torch.from_numpy(mesh).float().to(device)
    output_mesh = regular_grid_from_mesh(mesh, get_param(params, "grid_size"), device)

    n_input_steps = int(get_param(params, "n_input_steps", 1))
    in_dim = n_input_steps * (int(get_param(params, "in_dim")) + int(get_param(params, "n_static_channels", 0)))
    out_dim = int(get_param(params, "out_dim", 3))
    nettype = get_param(params, "nettype", "vit")

    common = dict(
        in_dim=in_dim,
        out_dim=out_dim,
        input_grid=input_mesh,
        output_grid=output_mesh,
        n_neigbor=int(get_param(params, "n_neigbor", 10)),
        hidden_dim=int(get_param(params, "hidden_dim", 256)),
        lifting_dim=int(get_param(params, "lifting_dim", get_param(params, "hidden_dim", 256))),
        n_layers=int(get_param(params, "n_layers", 5)),
        initial_mesh=input_mesh,
        gno_mlp_layers=list(get_param(params, "gno_mlp_layers", [64, 64])),
        lifting=True,
        projection=True,
    )

    if nettype == "vit":
        model = VitGno(
            **common,
            radius=float(get_param(params, "radius", 0.08)),
            fixed_neighbour=bool(get_param(params, "fixed_neighbour", False)),
            grid_size=tuple(get_param(params, "grid_size")),
            patch_size=tuple(get_param(params, "patch_size")),
            heads=int(get_param(params, "heads", 4)),
            n_layers_transformer=int(get_param(params, "n_layers_transformer", 7)),
        )
    elif nettype == "gnn":
        model = GNN(**common)
    else:
        raise ValueError(f"Unsupported nettype {nettype!r}; expected 'vit' or 'gnn'.")

    return model.to(device), input_mesh

