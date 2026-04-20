from pathlib import Path
from timeit import default_timer

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

from train.new_adam import Adam
from utils import get_grid_displacement, get_param, prepare_model_input


class RelativeErrorLoss(nn.Module):
    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        self.p = p
        self.reduction = reduction

    def forward(self, predicted, target):
        batch_size = predicted.shape[0]
        diff = predicted.reshape(batch_size, -1) - target.reshape(batch_size, -1)
        diff_norm = torch.norm(diff, self.p, dim=1)
        target_norm = torch.norm(target.reshape(batch_size, -1), self.p, dim=1)
        rel = diff_norm / (target_norm + 1e-8)
        if self.reduction == "sum":
            return rel.sum()
        if self.reduction == "mean":
            return rel.mean()
        raise ValueError("reduction must be 'mean' or 'sum'")


def _boundary_correct(out, target, mask_all, fixed_points, normalizer):
    if mask_all is not None and len(mask_all) > 0:
        out[:, mask_all, :2] = target[:, mask_all, :2]
    if fixed_points is not None and len(fixed_points) > 0:
        zero = torch.zeros_like(normalizer.mean, device=out.device)
        fixed_uv = normalizer(zero)[:2].view(1, 1, 2)
        out[:, fixed_points, :2] = fixed_uv.expand(out.shape[0], fixed_points.numel(), -1)
    return out


def _run_eval(model, loader, params, loss_fn, device, normalizer, mask_all, fixed_points):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"][..., :3].to(device)
            static = batch["static_features"].to(device)
            inp = prepare_model_input(x, static, params)
            out_grid, in_grid = get_grid_displacement(params, batch)
            if out_grid is not None:
                out_grid = out_grid.to(device)
                in_grid = in_grid.to(device)
            out = model(inp, out_grid_displacement=out_grid, in_grid_displacement=in_grid)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out[..., :3]
            if bool(get_param(params, "boundarymask", True)):
                out = _boundary_correct(out, y, mask_all, fixed_points, normalizer)
            total += loss_fn(out, y).item() * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def train_fluid_model(model, train_loader, test_loader, params, normalizer, mask_all, fixed_points, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    normalizer.to(device)
    mask_all = torch.as_tensor(mask_all, dtype=torch.long, device=device) if mask_all is not None else None
    fixed_points = fixed_points.to(device) if fixed_points is not None else None

    lr = float(get_param(params, "lr", 1e-4))
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=float(get_param(params, "weight_decay", 0.0)), amsgrad=False)
    if get_param(params, "scheduler_type", "rlp") == "step":
        scheduler = StepLR(
            optimizer,
            step_size=int(get_param(params, "scheduler_step", 50)),
            gamma=float(get_param(params, "scheduler_gamma", 0.5)),
        )
    else:
        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=int(get_param(params, "scheduler_step", 50)),
            factor=float(get_param(params, "scheduler_gamma", 0.5)),
            min_lr=float(get_param(params, "min_lr", 1e-6)),
        )

    epochs = int(get_param(params, "epochs", 300))
    loss_fn = RelativeErrorLoss()
    save_dir = Path(get_param(params, "weight_path", "weights"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(get_param(params, "weight_saving_interval", 50))
    log_every = int(get_param(params, "log_interval", 1))
    config_name = get_param(params, "config", "fsi_gno_vit")

    history = []
    for epoch in range(epochs):
        model.train()
        start = default_timer()
        train_loss = 0.0
        count = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
            optimizer.zero_grad()
            x = batch["x"].to(device)
            y = batch["y"][..., :3].to(device)
            static = batch["static_features"].to(device)
            inp = prepare_model_input(x, static, params)
            out_grid, in_grid = get_grid_displacement(params, batch)
            if out_grid is not None:
                out_grid = out_grid.to(device)
                in_grid = in_grid.to(device)

            out = model(inp, out_grid_displacement=out_grid, in_grid_displacement=in_grid)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = out[..., :3]
            if bool(get_param(params, "boundarymask", True)):
                out = _boundary_correct(out, y, mask_all, fixed_points, normalizer)

            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
            count += x.shape[0]

        avg_train = train_loss / max(count, 1)
        test_loss = _run_eval(model, test_loader, params, loss_fn, device, normalizer, mask_all, fixed_points)
        if get_param(params, "scheduler_type", "rlp") == "step":
            scheduler.step()
        else:
            scheduler.step(test_loss)

        elapsed = default_timer() - start
        history.append({"epoch": epoch, "train_loss": avg_train, "test_loss": test_loss})
        if epoch % log_every == 0:
            print(
                f"epoch={epoch:04d} train={avg_train:.6e} test={test_loss:.6e} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} time={elapsed:.2f}s"
            )

        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), save_dir / f"{config_name}_fluid_{epoch}.pt")

    return history

