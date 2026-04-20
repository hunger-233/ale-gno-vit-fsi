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
    parser = argparse.ArgumentParser(description="Train the LSTM structural-interface surrogate.")
    parser.add_argument("--config-file", default=str(ROOT / "configs" / "fsi_gno_vit.yaml"))
    parser.add_argument("--config", default="structure_periodic")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ntrain", type=int, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    from YParams import YParams
    from data_utils.data_loaders import build_structure_dataloaders
    from train.structure_trainer import train_structure_model
    from utils import get_param, resolve_path, set_seed

    params = YParams(args.config_file, args.config, print_params=True)
    params.config = args.config
    if args.epochs is not None:
        params.structure_epochs = args.epochs
    if args.ntrain is not None:
        params.structure_ntrain = args.ntrain
    if args.ntest is not None:
        params.structure_ntest = args.ntest
    normalize_paths(params, get_param, resolve_path)
    set_seed(int(get_param(params, "random_seed", 24)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader, normalizer, metadata = build_structure_dataloaders(params)
    model, history = train_structure_model(train_loader, test_loader, params, normalizer, metadata, device=device)
    weight_dir = Path(get_param(params, "weight_path", ROOT / "weights"))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean": normalizer.mean.cpu(),
            "std": normalizer.std.cpu(),
            "metadata": metadata,
            "history": history,
        },
        weight_dir / f"{args.config}_structure_final.pt",
    )


if __name__ == "__main__":
    main()
