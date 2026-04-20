from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.structure_lstm import DispVeloPredictor, apply_structure_constraints
from train.trainer import RelativeErrorLoss
from utils import get_param


def train_structure_model(train_loader, test_loader, params, normalizer, metadata, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_points = int(metadata["n_points"])
    n_vars = int(metadata["n_vars"])
    input_size = n_points * n_vars
    output_size = n_points * 4
    model = DispVeloPredictor(
        input_size=input_size,
        hidden_size=int(get_param(params, "structure_hidden_size", 512)),
        num_layers=int(get_param(params, "structure_num_layers", 3)),
        output_size=output_size,
        dropout=float(get_param(params, "structure_dropout", 0.5)),
    ).to(device)
    normalizer.to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(get_param(params, "structure_lr", 1e-3)))
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=int(get_param(params, "structure_scheduler_step", 100)),
        factor=float(get_param(params, "scheduler_gamma", 0.5)),
        min_lr=float(get_param(params, "min_lr", 1e-7)),
    )
    loss_fn = RelativeErrorLoss()
    epochs = int(get_param(params, "structure_epochs", 10000))
    save_dir = Path(get_param(params, "weight_path", "weights"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(get_param(params, "structure_weight_saving_interval", 200))
    config_name = get_param(params, "config", "fsi_gno_vit")

    history = []
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            out = apply_structure_constraints(out, normalizer, n_points)
            loss = loss_fn(out, batch_y)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * batch_x.shape[0]
            train_count += batch_x.shape[0]
        train_loss = train_total / max(train_count, 1)

        model.eval()
        test_total = 0.0
        test_count = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                out = model(batch_x)
                out = apply_structure_constraints(out, normalizer, n_points)
                loss = loss_fn(out, batch_y)
                test_total += loss.item() * batch_x.shape[0]
                test_count += batch_x.shape[0]
        test_loss = test_total / max(test_count, 1)
        scheduler.step(test_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        print(
            f"epoch={epoch:04d} structure_train={train_loss:.6e} "
            f"structure_test={test_loss:.6e} lr={optimizer.param_groups[0]['lr']:.3e}"
        )
        if epoch % save_every == 0 or epoch == epochs - 1:
            torch.save(model.state_dict(), save_dir / f"{config_name}_structure_{epoch}.pt")

    return model, history

