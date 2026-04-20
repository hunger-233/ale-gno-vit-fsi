import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalize_paths(params, get_param, resolve_path):
    for key in ("data_location", "input_mesh_location", "boundary_mask_path", "weight_path"):
        value = get_param(params, key, None)
        if value:
            params[key] = str(resolve_path(value, ROOT))


def main():
    parser = argparse.ArgumentParser(description="Train the GNO-ViT fluid surrogate.")
    parser.add_argument("--config-file", default=str(ROOT / "configs" / "fsi_gno_vit.yaml"))
    parser.add_argument("--config", default="fluid_periodic")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    from YParams import YParams
    from data_utils.data_loaders import build_fluid_dataloaders, infer_boundary_mask_path, load_boundary_indices
    from models.get_models import build_fluid_model
    from train.trainer import train_fluid_model
    from utils import fixed_boundary_indices, get_param, resolve_path, set_seed

    params = YParams(args.config_file, args.config, print_params=True)
    params.config = args.config
    if args.epochs is not None:
        params.epochs = args.epochs
    if args.ntrain is not None:
        params.ntrain = args.ntrain
    if args.ntest is not None:
        params.ntest = args.ntest
    normalize_paths(params, get_param, resolve_path)
    set_seed(int(get_param(params, "random_seed", 24)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader, normalizer, mesh = build_fluid_dataloaders(params)
    mask_path = infer_boundary_mask_path(params)
    mask_all = load_boundary_indices(mask_path)["all"]
    fixed_points = fixed_boundary_indices(mesh)
    model, _ = build_fluid_model(params, device=device)
    history = train_fluid_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        params=params,
        normalizer=normalizer,
        mask_all=mask_all,
        fixed_points=fixed_points,
        device=device,
    )

    weight_dir = Path(get_param(params, "weight_path", ROOT / "weights"))
    torch.save(
        {"mean": normalizer.mean.cpu(), "std": normalizer.std.cpu(), "history": history},
        weight_dir / f"{args.config}_fluid_normalizer.pt",
    )


if __name__ == "__main__":
    main()
