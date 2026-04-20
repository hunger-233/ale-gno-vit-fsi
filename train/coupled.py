from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_utils.data_loaders import (
    IrregularMeshTensorDataset,
    build_fluid_case_dataset,
    build_structure_case_sequences,
)
from models.structure_lstm import apply_structure_constraints
from train.new_adam import Adam
from train.trainer import RelativeErrorLoss
from utils import get_grid_displacement, get_param, prepare_model_input


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(model, checkpoint_path, device):
    """Load either a state_dict checkpoint or an older full-module checkpoint."""
    checkpoint = torch_load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=False)
    return model.to(device)


def single_item_batch(dataset, index):
    item = dataset[index]
    batch = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        else:
            batch[key] = value
    return batch


def normalized_zero(normalizer, device):
    zero = torch.zeros_like(normalizer.mean, device=device)
    return normalizer(zero)


def _make_continuation_dataset(raw_window, result, normalizer, params, input_mesh, mu, x1, x2):
    next_window = torch.cat((raw_window[:, 1:], result.unsqueeze(1)), dim=1)
    return IrregularMeshTensorDataset(
        next_window,
        next_window,
        normalizer,
        normalizer,
        x1=x1,
        x2=x2,
        mu=mu,
        equation=list(get_param(params, "equation_dict", {"NS": 3, "ES": 4}).keys()),
        mesh=input_mesh.cpu(),
    )


def coupled_rollout(
    fluid_model,
    structure_model,
    params,
    fluid_normalizer,
    structure_normalizer,
    input_mesh,
    mask_all,
    fixed_points,
    ale_updater,
    x2,
    step_start,
    steps,
    device=None,
    mu=None,
    x1=None,
):
    """Run the long-horizon coupled FSI calculation from the notebook.

    The loop follows:
    fluid(t, mesh(t+dt)) -> structure boundary(t+2dt) -> ALE mesh(t+2dt)
    -> next fluid input.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu = float(mu if mu is not None else get_param(params, "mu_list", [1.0])[0])
    x1 = float(x1 if x1 is not None else get_param(params, "supervised_inlets_x1", [-4.0])[0])
    mask_all = torch.as_tensor(mask_all, dtype=torch.long)
    fixed_points = torch.as_tensor(fixed_points, dtype=torch.long)
    input_mesh = torch.as_tensor(input_mesh, dtype=torch.float32)
    fluid_normalizer.to(device)
    structure_normalizer.to(device)
    fluid_model.to(device).eval()
    structure_model.to(device).eval()

    case_dataset = build_fluid_case_dataset(params, x2=x2, normalizer=fluid_normalizer, mu=mu, x1=x1)
    _, _, structure_x_norm, structure_y_norm = build_structure_case_sequences(
        params, x2=x2, normalizer=structure_normalizer, mask_all=mask_all.numpy(), mu=mu, x1=x1
    )

    t_in = int(get_param(params, "structure_t_in", 10))
    n_points = int(structure_x_norm.shape[-2])
    n_vars = int(structure_x_norm.shape[-1])
    static_channels = 3
    target_channels = n_vars - 1 - static_channels
    n_nodes = int(input_mesh.shape[0])
    step_end = step_start + steps
    if step_start - t_in + 2 < 0:
        raise ValueError(f"step_start={step_start} is too early for structure_t_in={t_in}")
    if step_end >= len(case_dataset):
        raise ValueError(f"Rollout end step {step_end} exceeds fluid case length {len(case_dataset)}")
    if step_end >= structure_x_norm.shape[0]:
        raise ValueError(f"Rollout end step {step_end} exceeds structure case length {structure_x_norm.shape[0]}")

    data_truth = np.empty((steps, n_nodes, 5), dtype=np.float32)
    data_pred = np.empty((steps, n_nodes, 5), dtype=np.float32)
    predicted_boundaries = []

    current_dataset = case_dataset
    current_index = step_start
    current_structure_input = None
    previous_boundary = None

    with torch.no_grad():
        for local_idx, step in enumerate(range(step_start, step_end)):
            batch = single_item_batch(current_dataset, current_index)
            x = batch["x"].to(device)
            y = single_item_batch(case_dataset, step)["y"].to(device)
            static = batch["static_features"].to(device)
            inp = prepare_model_input(x, static, params)
            out_grid, in_grid = get_grid_displacement(params, batch)
            out_grid = out_grid.to(device)
            in_grid = in_grid.to(device)

            fluid_out = fluid_model(
                inp,
                in_grid_displacement=in_grid,
                out_grid_displacement=out_grid,
            )
            if isinstance(fluid_out, (tuple, list)):
                fluid_out = fluid_out[0]
            fluid_out = fluid_out.clone()

            if local_idx == 0:
                fluid_out[:, mask_all.to(device), :2] = y[:, mask_all.to(device), :2]
            else:
                boundary_uv = previous_boundary[:, :2].clone().to(device)
                boundary_uv[:, 0] = (boundary_uv[:, 0] - fluid_normalizer.mean[0]) / fluid_normalizer.std[0]
                boundary_uv[:, 1] = (boundary_uv[:, 1] - fluid_normalizer.mean[1]) / fluid_normalizer.std[1]
                fluid_out[:, mask_all.to(device), :2] = boundary_uv.unsqueeze(0)

            fixed_uv = normalized_zero(fluid_normalizer, device)[:2].view(1, 1, 2)
            fluid_out[:, fixed_points.to(device), :2] = fixed_uv.expand(
                fluid_out.shape[0], fixed_points.numel(), -1
            )
            fluid_out = fluid_out[..., :3]

            pred_fluid_raw = fluid_normalizer.denormalize_coarse(fluid_out)
            truth_fluid_raw = fluid_normalizer.denormalize_coarse(y[..., :3])
            mesh_t_dt = input_mesh + out_grid.cpu()
            data_truth[local_idx, :, :3] = truth_fluid_raw[0].detach().cpu().numpy()
            data_truth[local_idx, :, 3:] = mesh_t_dt.numpy()
            data_pred[local_idx, :, :3] = pred_fluid_raw[0].detach().cpu().numpy()
            data_pred[local_idx, :, 3:] = mesh_t_dt.numpy()

            pressure = pred_fluid_raw[0, mask_all.to(device), 2:3]
            pressure = (pressure - structure_normalizer.mean[2]) / structure_normalizer.std[2]

            if local_idx == 0:
                structure_idx = step - t_in + 2
                current_structure_input = structure_x_norm[structure_idx].reshape(1, t_in, n_points * n_vars).to(device)
            else:
                pred_view = structure_pred.reshape(1, n_points, target_channels)
                next_structure_step = torch.cat(
                    (
                        pred_view[..., :2],
                        pressure.unsqueeze(0),
                        pred_view[..., 2:],
                        structure_x_norm[step, -2:-1, :, 5:].to(device),
                    ),
                    dim=2,
                ).reshape(1, 1, n_points * n_vars)
                current_structure_input = torch.cat(
                    [current_structure_input[:, 1:, :], next_structure_step],
                    dim=1,
                )

            structure_pred = structure_model(current_structure_input)
            structure_pred = apply_structure_constraints(
                structure_pred,
                structure_normalizer,
                n_points,
                target_channels=target_channels,
            )
            previous_boundary = structure_normalizer.denormalize_structure(
                structure_pred.reshape(n_points, target_channels)
            ).detach().cpu()

            boundary_positions = input_mesh[mask_all] + previous_boundary[..., 2:]
            predicted_boundaries.append(boundary_positions.numpy())
            new_points = ale_updater.update(input_mesh.numpy(), boundary_positions.numpy())

            result = torch.zeros(1, n_nodes, 7, dtype=pred_fluid_raw.dtype, device="cpu")
            result[:, :, :3] = pred_fluid_raw.detach().cpu()
            result[:, :, 3:5] = batch["d_grid_y"].cpu()
            result[:, :, 5:] = torch.from_numpy(new_points).float().unsqueeze(0) - input_mesh.unsqueeze(0)
            raw_window = current_dataset.x[current_index].unsqueeze(0).cpu()
            current_dataset = _make_continuation_dataset(
                raw_window, result, fluid_normalizer, params, input_mesh, mu, x1, x2
            )
            current_index = 0

    return {
        "truth": data_truth,
        "prediction": data_pred,
        "predicted_boundaries": np.asarray(predicted_boundaries),
        "step_start": step_start,
        "steps": steps,
        "x2": x2,
    }


def train_coupled_model(
    fluid_model,
    structure_model,
    params,
    fluid_normalizer,
    structure_normalizer,
    input_mesh,
    mask_all,
    fixed_points,
    ale_updater,
    device=None,
):
    """Joint long-term fine-tuning loop extracted from the notebook.

    This keeps the notebook objective: fluid relative error plus a weighted
    structure-boundary relative error over short autoregressive windows.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fluid_model.to(device)
    structure_model.to(device)
    fluid_normalizer.to(device)
    structure_normalizer.to(device)
    input_mesh = torch.as_tensor(input_mesh, dtype=torch.float32)
    mask_all = torch.as_tensor(mask_all, dtype=torch.long)
    fixed_points = torch.as_tensor(fixed_points, dtype=torch.long)

    train_x2 = get_param(params, "supervised_inlets_x2_train", [-4.0, 0.0, 2.0, 4.0])
    mu = float(get_param(params, "mu_list", [1.0])[0])
    x1 = float(get_param(params, "supervised_inlets_x1", [-4.0])[0])
    cases = []
    for x2 in train_x2:
        fluid_case = build_fluid_case_dataset(params, x2=x2, normalizer=fluid_normalizer, mu=mu, x1=x1)
        _, _, sx, sy = build_structure_case_sequences(
            params, x2=x2, normalizer=structure_normalizer, mask_all=mask_all.numpy(), mu=mu, x1=x1
        )
        cases.append((float(x2), fluid_case, sx, sy))

    optimizer_fluid = Adam(
        fluid_model.parameters(),
        lr=float(get_param(params, "lr", 1e-4)),
        weight_decay=float(get_param(params, "weight_decay", 0.0)),
        amsgrad=False,
    )
    optimizer_structure = torch.optim.Adam(
        structure_model.parameters(),
        lr=float(get_param(params, "structure_lr", get_param(params, "lr", 1e-4))),
    )
    scheduler_fluid = ReduceLROnPlateau(
        optimizer_fluid,
        patience=int(get_param(params, "scheduler_step", 5)),
        factor=float(get_param(params, "scheduler_gamma", 0.5)),
        min_lr=float(get_param(params, "min_lr", 1e-7)),
    )
    scheduler_structure = ReduceLROnPlateau(
        optimizer_structure,
        patience=int(get_param(params, "structure_scheduler_step", get_param(params, "scheduler_step", 5))),
        factor=float(get_param(params, "scheduler_gamma", 0.5)),
        min_lr=float(get_param(params, "min_lr", 1e-7)),
    )

    epochs = int(get_param(params, "coupled_epochs", get_param(params, "epochs", 1000)))
    horizon = int(get_param(params, "coupled_train_horizon", 5))
    alpha = float(get_param(params, "coupled_structure_loss_weight", 5.0))
    t_in = int(get_param(params, "structure_t_in", 10))
    save_dir = Path(get_param(params, "weight_path", "weights"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(get_param(params, "weight_saving_interval", 50))
    config_name = get_param(params, "config", "coupled_fsi")
    loss_fn = RelativeErrorLoss()
    history = []

    for epoch in range(epochs):
        fluid_model.train()
        structure_model.train()
        epoch_fluid = 0.0
        epoch_structure = 0.0
        case_count = 0

        for x2, fluid_case, structure_x_norm, structure_y_norm in cases:
            max_start = min(len(fluid_case), structure_x_norm.shape[0]) - horizon - 1
            min_start = max(t_in, int(get_param(params, "coupled_min_step_start", 100)))
            if max_start <= min_start:
                continue
            step_start = int(np.random.randint(min_start, max_start))
            current_dataset = fluid_case
            current_index = step_start
            current_structure_input = None
            previous_boundary = None
            total_fluid = 0.0
            total_structure = 0.0

            optimizer_fluid.zero_grad()
            optimizer_structure.zero_grad()

            for local_idx, step in enumerate(range(step_start, step_start + horizon)):
                batch = single_item_batch(current_dataset, current_index)
                x = batch["x"].to(device)
                y = single_item_batch(fluid_case, step)["y"].to(device)
                inp = prepare_model_input(x, batch["static_features"].to(device), params)
                out_grid, in_grid = get_grid_displacement(params, batch)
                out_grid = out_grid.to(device)
                in_grid = in_grid.to(device)

                fluid_out = fluid_model(
                    inp,
                    in_grid_displacement=in_grid,
                    out_grid_displacement=out_grid,
                )
                if isinstance(fluid_out, (tuple, list)):
                    fluid_out = fluid_out[0]
                fluid_out = fluid_out.clone()
                if local_idx == 0:
                    fluid_out[:, mask_all.to(device), :2] = y[:, mask_all.to(device), :2]
                else:
                    boundary_uv = previous_boundary[:, :2].clone().to(device)
                    boundary_uv[:, 0] = (boundary_uv[:, 0] - fluid_normalizer.mean[0]) / fluid_normalizer.std[0]
                    boundary_uv[:, 1] = (boundary_uv[:, 1] - fluid_normalizer.mean[1]) / fluid_normalizer.std[1]
                    fluid_out[:, mask_all.to(device), :2] = boundary_uv.unsqueeze(0)
                fixed_uv = normalized_zero(fluid_normalizer, device)[:2].view(1, 1, 2)
                fluid_out[:, fixed_points.to(device), :2] = fixed_uv.expand(
                    fluid_out.shape[0], fixed_points.numel(), -1
                )
                fluid_out = fluid_out[..., :3]
                total_fluid = total_fluid + loss_fn(fluid_out, y[..., :3])

                pred_fluid_raw = fluid_normalizer.denormalize_coarse(fluid_out)
                pressure = pred_fluid_raw[0, mask_all.to(device), 2:3]
                pressure = (pressure - structure_normalizer.mean[2]) / structure_normalizer.std[2]

                if local_idx == 0:
                    structure_idx = step - t_in + 2
                    current_structure_input = structure_x_norm[structure_idx].reshape(
                        1, t_in, structure_x_norm.shape[-2] * structure_x_norm.shape[-1]
                    ).to(device)
                else:
                    n_points = structure_x_norm.shape[-2]
                    n_vars = structure_x_norm.shape[-1]
                    pred_view = structure_pred.reshape(1, n_points, 4)
                    next_structure_step = torch.cat(
                        (
                            pred_view[..., :2],
                            pressure.unsqueeze(0),
                            pred_view[..., 2:],
                            structure_x_norm[step, -2:-1, :, 5:].to(device),
                        ),
                        dim=2,
                    ).reshape(1, 1, n_points * n_vars)
                    current_structure_input = torch.cat(
                        [current_structure_input[:, 1:, :], next_structure_step],
                        dim=1,
                    )

                n_points = structure_x_norm.shape[-2]
                structure_pred = structure_model(current_structure_input)
                structure_pred = apply_structure_constraints(
                    structure_pred,
                    structure_normalizer,
                    n_points,
                    target_channels=4,
                )
                target_idx = step - t_in + 2
                total_structure = total_structure + loss_fn(
                    structure_pred,
                    structure_y_norm[target_idx:target_idx + 1].reshape(1, -1).to(device),
                )

                previous_boundary = structure_normalizer.denormalize_structure(
                    structure_pred.reshape(n_points, 4)
                )
                boundary_positions = input_mesh[mask_all] + previous_boundary.detach().cpu()[..., 2:]
                new_points = ale_updater.update(input_mesh.numpy(), boundary_positions.numpy())

                result = torch.zeros(1, input_mesh.shape[0], 7, dtype=pred_fluid_raw.dtype, device="cpu")
                result[:, :, :3] = pred_fluid_raw.detach().cpu()
                result[:, :, 3:5] = batch["d_grid_y"].cpu()
                result[:, :, 5:] = torch.from_numpy(new_points).float().unsqueeze(0) - input_mesh.unsqueeze(0)
                raw_window = current_dataset.x[current_index].unsqueeze(0).cpu()
                current_dataset = _make_continuation_dataset(
                    raw_window, result, fluid_normalizer, params, input_mesh, mu, x1, x2
                )
                current_index = 0

            total_fluid = total_fluid / horizon
            total_structure = total_structure / horizon
            loss = total_fluid + alpha * total_structure
            loss.backward()
            optimizer_fluid.step()
            optimizer_structure.step()
            epoch_fluid += float(total_fluid.detach().cpu())
            epoch_structure += float(total_structure.detach().cpu())
            case_count += 1

        if case_count == 0:
            raise RuntimeError("No valid coupled training cases were built.")
        epoch_fluid /= case_count
        epoch_structure /= case_count
        scheduler_fluid.step(epoch_fluid)
        scheduler_structure.step(epoch_structure)
        history.append({"epoch": epoch, "fluid_loss": epoch_fluid, "structure_loss": epoch_structure})
        print(
            f"epoch={epoch:04d} coupled_fluid={epoch_fluid:.6e} "
            f"coupled_structure={epoch_structure:.6e} "
            f"lr_fluid={optimizer_fluid.param_groups[0]['lr']:.3e} "
            f"lr_structure={optimizer_structure.param_groups[0]['lr']:.3e}"
        )
        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(fluid_model.state_dict(), save_dir / f"{config_name}_fluid_{epoch}.pt")
            torch.save(structure_model.state_dict(), save_dir / f"{config_name}_structure_{epoch}.pt")

    return history
