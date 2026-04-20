import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from YParams import YParams
from data_utils.data_loaders import build_fluid_dataloaders, build_structure_dataloaders, infer_boundary_mask_path
from mesh.ale import ALEMeshUpdater
from models.get_models import build_fluid_model
from models.structure_lstm import DispVeloPredictor
from train.coupled import load_checkpoint, train_coupled_model
from utils import get_param, resolve_path, set_seed

# Backward compatibility for structure models saved directly from notebooks.
Disp_Velo_Predictor = DispVeloPredictor


def main():
    parser = argparse.ArgumentParser(description="Joint long-term training for the coupled FSI surrogate.")
    parser.add_argument("--config-file", default="configs/fsi_gno_vit.yaml")
    parser.add_argument("--config", default="coupled_nonperiodic")
    parser.add_argument("--fluid-weight", default=None, help="Optional initial fluid checkpoint.")
    parser.add_argument("--structure-weight", default=None, help="Optional initial structure checkpoint.")
    parser.add_argument("--mesh", default=None, help="Reference .msh file for the ALE updater.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--min-step-start", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    params = YParams(args.config_file, args.config, print_params=True)
    params.config = args.config
    if args.epochs is not None:
        params.coupled_epochs = args.epochs
    if args.horizon is not None:
        params.coupled_train_horizon = args.horizon
    if args.min_step_start is not None:
        params.coupled_min_step_start = args.min_step_start
    set_seed(int(get_param(params, "random_seed", 42)))

    _, _, fluid_normalizer, input_mesh = build_fluid_dataloaders(params)
    _, _, structure_normalizer, structure_meta = build_structure_dataloaders(params)

    fluid_model, _ = build_fluid_model(params, device=device)
    n_points = int(structure_meta["n_points"])
    n_vars = int(structure_meta["n_vars"])
    structure_model = DispVeloPredictor(
        input_size=n_points * n_vars,
        hidden_size=int(get_param(params, "structure_hidden_size", 512)),
        num_layers=int(get_param(params, "structure_num_layers", 3)),
        output_size=n_points * 4,
        dropout=float(get_param(params, "structure_dropout", 0.5)),
    )

    if args.fluid_weight:
        fluid_model = load_checkpoint(fluid_model, args.fluid_weight, device)
    if args.structure_weight:
        structure_model = load_checkpoint(structure_model, args.structure_weight, device)

    mesh_path = args.mesh or get_param(params, "ale_mesh", "square_with_circle_hole.msh")
    mesh_path = resolve_path(mesh_path, base_dir=ROOT)
    boundary_mask = infer_boundary_mask_path(params)
    ale = ALEMeshUpdater.from_mesh(mesh_path, boundary_mask, alpha=float(get_param(params, "ale_alpha", 0.1)))

    train_coupled_model(
        fluid_model=fluid_model,
        structure_model=structure_model,
        params=params,
        fluid_normalizer=fluid_normalizer,
        structure_normalizer=structure_normalizer,
        input_mesh=input_mesh,
        mask_all=structure_meta["mask_all"],
        fixed_points=structure_meta["fixed_points"],
        ale_updater=ale,
        device=device,
    )


if __name__ == "__main__":
    main()
