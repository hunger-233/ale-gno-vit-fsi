import torch
import torch.nn as nn


class DispVeloPredictor(nn.Module):
    """LSTM surrogate for structural boundary velocity and displacement."""

    def __init__(self, input_size, hidden_size=512, num_layers=3, output_size=None, dropout=0.5):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = self.dropout(lstm_out[:, -1, :])
        return self.fc(last_out)


def apply_structure_constraints(prediction, normalizer, n_points, target_channels=4):
    """Apply root and tip consistency constraints used for the flexible beam.

    Boundary order follows the paper dataset: lower edge 0-25, upper edge 26-51,
    right edge 52-54. Nodes 0 and 26 are clamped, and right-edge corner values
    are tied to their upper/lower edge duplicates.
    """
    pred = prediction.reshape(-1, n_points, target_channels)
    device = pred.device

    mask_fixed = torch.tensor([0, 26], dtype=torch.long, device=device)
    mask_equal_up_down = torch.tensor([25, 51], dtype=torch.long, device=device)
    mask_equal_right = torch.tensor([54, 52], dtype=torch.long, device=device)

    zero = torch.zeros_like(normalizer.mean, device=device)
    fixed_value = normalizer(zero)[[0, 1, 3, 4]].view(1, 1, target_channels)
    if n_points > int(mask_fixed.max()):
        pred[:, mask_fixed, :] = fixed_value.expand(pred.shape[0], mask_fixed.numel(), -1)
    if n_points > int(mask_equal_right.max()):
        pred[:, mask_equal_right, :] = pred[:, mask_equal_up_down, :]
    return pred.reshape(prediction.shape[0], n_points * target_channels)

