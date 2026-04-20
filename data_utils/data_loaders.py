from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset, random_split

from utils import fixed_boundary_indices, get_param, load_mesh, resolve_path


FLUID_CHANNELS = [0, 1, 3, 4, 5, 7, 8]
STRUCTURE_TARGET_CHANNELS = [0, 1, 3, 4]


class Normalizer:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        return (data - self.mean.to(data.device)) / (self.std.to(data.device) + self.eps)

    def denormalize(self, data):
        return data * (self.std.to(data.device) + self.eps) + self.mean.to(data.device)

    def denormalize_coarse(self, data):
        return data * (self.std[:3].to(data.device) + self.eps) + self.mean[:3].to(data.device)

    def denormalize_structure(self, data):
        idx = torch.tensor(STRUCTURE_TARGET_CHANNELS, device=data.device)
        return data * (self.std[idx].to(data.device) + self.eps) + self.mean[idx].to(data.device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class IrregularMeshTensorDataset(Dataset):
    """Fluid sequence dataset on a moving unstructured mesh.

    ``x`` has shape ``(samples, n_input_steps, n_nodes, 7)`` with channels
    ``u, v, p, dx_t, dy_t, dx_t+dt, dy_t+dt``.
    """

    def __init__(self, x, y, transform_x, transform_y, equation, x1, x2, mu, mesh):
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.equation = equation
        self.x1 = x1
        self.x2 = x2
        self.mu = mu
        self.mesh = torch.as_tensor(mesh, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def _static_features(self, d_grid):
        n_grid_points = self.x.shape[2]
        n_variables = self.x.shape[-1] if len(self.equation) > 1 else 3
        position = self.mesh + d_grid.cpu()
        x1 = torch.ones(n_grid_points, 1, dtype=torch.float32) * float(self.x1)
        x2 = torch.ones(n_grid_points, 1, dtype=torch.float32) * float(self.x2)
        return torch.cat([x1, x2, position], dim=-1).repeat(1, n_variables)

    def __getitem__(self, index):
        x = self.x[index].clone()
        y = self.y[index].clone()
        d_grid_x = x[-1, :, 3:5].clone()
        d_grid_y = x[-1, :, 5:7].clone()
        static_features = self._static_features(d_grid_x)

        if self.transform_x is not None:
            x = self.transform_x(x)
        if self.transform_y is not None:
            y = self.transform_y(y)
        if len(self.equation) == 1:
            x = x[:, :, :3]
            y = y[:, :3]

        return {
            "x": x,
            "y": y,
            "d_grid_x": d_grid_x,
            "d_grid_y": d_grid_y,
            "static_features": static_features,
            "equation": self.equation,
        }


class FsiDataset:
    """Reader for the coarse Turek-Hron FSI dataset used in the paper."""

    def __init__(self, data_location, mesh_location, params):
        self.data_location = resolve_path(data_location)
        self.mesh_location = resolve_path(mesh_location)
        self.params = params
        self.mesh = torch.from_numpy(load_mesh(self.mesh_location)).float()

    @staticmethod
    def _param_names(value):
        value = float(value)
        names = [str(value)]
        if value.is_integer():
            names.append(str(int(value)))
        return list(dict.fromkeys(names))

    def _case_candidates(self, mu, x1, x2):
        candidates = []
        for mu_name in self._param_names(mu):
            for x1_name in self._param_names(x1):
                for x2_name in self._param_names(x2):
                    base = self.data_location / f"mu={mu_name}" / f"x1={x1_name}" / f"x2={x2_name}"
                    candidates.extend([base / "Visualization", base / "1" / "Visualization"])
        return candidates

    def _case_path(self, mu, x1, x2):
        for path in self._case_candidates(mu, x1, x2):
            if path.exists():
                return path
        raise FileNotFoundError(f"No FSI case found for mu={mu}, x1={x1}, x2={x2} under {self.data_location}")

    @staticmethod
    def _read_array(path, key):
        with h5py.File(path, "r") as h5f:
            if key in h5f:
                return torch.from_numpy(np.asarray(h5f[key], dtype=np.float32))
            if "VisualisationVector" in h5f:
                values = [None] * len(h5f["VisualisationVector"].keys())
                for item in h5f["VisualisationVector"].keys():
                    values[int(item)] = torch.tensor(np.asarray(h5f["VisualisationVector"][item]), dtype=torch.float32)
                return torch.stack(values, dim=0)
        raise KeyError(f"Cannot find dataset {key!r} in {path}")

    def read_case(self, mu, x1, x2):
        path = self._case_path(mu, x1, x2)
        velocity = self._read_array(path / "velocity.h5", "velocity")
        pressure = self._read_array(path / "pressure.h5", "pressure")
        displacement = self._read_array(path / "displacement.h5", "displacement")
        return velocity, pressure, displacement


def _time_slice(tensor, params):
    start = int(get_param(params, "sample_per_inlet", 0) or 0)
    end = get_param(params, "sample_per_inlet_end", None)
    periodic = bool(get_param(params, "periodic", True))
    if end is not None:
        return tensor[start:int(end)]
    if periodic:
        return tensor[start:]
    return tensor[:start] if start > 0 else tensor


def make_fluid_sequences(velocity, pressure, displacement, params):
    dt = int(get_param(params, "dt", 1))
    n_input_steps = int(get_param(params, "n_input_steps", 1))
    combined = torch.cat(
        [velocity[:-dt], pressure[:-dt], displacement[:-dt], displacement[dt:]],
        dim=-1,
    )[..., FLUID_CHANNELS]
    combined = _time_slice(combined, params)
    if combined.shape[0] <= dt + n_input_steps:
        raise ValueError(f"Not enough time steps after slicing: {combined.shape[0]}")

    input_data = combined[:-dt]
    target_data = combined[dt:]
    xs, ys = [], []
    for idx in range(input_data.shape[0] - n_input_steps):
        xs.append(input_data[idx:idx + n_input_steps])
        ys.append(target_data[idx + n_input_steps])
    return torch.stack(xs), torch.stack(ys)


def _make_fluid_dataset(dataset, params, x2_values, normalizer, shuffle):
    mu_values = get_param(params, "mu_list", [1.0])
    x1_values = get_param(params, "supervised_inlets_x1", [-4.0])
    equation = list(get_param(params, "equation_dict", {"NS": 3, "ES": 4}).keys())
    subsets = []
    for mu in mu_values:
        for x1 in x1_values:
            for x2 in x2_values:
                velocity, pressure, displacement = dataset.read_case(mu, x1, x2)
                x, y = make_fluid_sequences(velocity, pressure, displacement, params)
                if shuffle:
                    indices = torch.randperm(x.shape[0])
                    x = x[indices]
                    y = y[indices]
                subsets.append(IrregularMeshTensorDataset(
                    x=x,
                    y=y,
                    transform_x=normalizer,
                    transform_y=normalizer,
                    equation=equation,
                    x1=x1,
                    x2=x2,
                    mu=mu,
                    mesh=dataset.mesh,
                ))
    return ConcatDataset(subsets)


def build_fluid_case_dataset(params, x2, normalizer, mu=None, x1=None):
    """Build one continuous fluid sequence dataset for long-horizon rollout."""
    dataset = FsiDataset(get_param(params, "data_location"), get_param(params, "input_mesh_location"), params)
    mu = float(mu if mu is not None else get_param(params, "mu_list", [1.0])[0])
    x1 = float(x1 if x1 is not None else get_param(params, "supervised_inlets_x1", [-4.0])[0])
    velocity, pressure, displacement = dataset.read_case(mu, x1, float(x2))
    x, y = make_fluid_sequences(velocity, pressure, displacement, params)
    equation = list(get_param(params, "equation_dict", {"NS": 3, "ES": 4}).keys())
    return IrregularMeshTensorDataset(
        x=x,
        y=y,
        transform_x=normalizer,
        transform_y=normalizer,
        equation=equation,
        x1=x1,
        x2=x2,
        mu=mu,
        mesh=dataset.mesh,
    )


def build_fluid_dataloaders(params):
    dataset = FsiDataset(get_param(params, "data_location"), get_param(params, "input_mesh_location"), params)
    train_x2 = get_param(params, "supervised_inlets_x2_train")
    test_x2 = get_param(params, "supervised_inlets_x2_test")
    if train_x2 is None:
        train_x2 = get_param(params, "supervised_inlets_x2", [-4.0, 0.0, 2.0, 4.0])
    if test_x2 is None:
        test_x2 = train_x2

    train_for_stats = []
    mu_values = get_param(params, "mu_list", [1.0])
    x1_values = get_param(params, "supervised_inlets_x1", [-4.0])
    for mu in mu_values:
        for x1 in x1_values:
            for x2 in train_x2:
                velocity, pressure, displacement = dataset.read_case(mu, x1, x2)
                x, _ = make_fluid_sequences(velocity, pressure, displacement, params)
                train_for_stats.append(x)
    train_for_stats = torch.cat(train_for_stats, dim=0)
    normalizer = Normalizer(
        torch.mean(train_for_stats, dim=(0, 1, 2)),
        torch.var(train_for_stats, dim=(0, 1, 2)).sqrt(),
    )

    train_dataset = _make_fluid_dataset(dataset, params, train_x2, normalizer, shuffle=True)
    test_dataset = _make_fluid_dataset(dataset, params, test_x2, normalizer, shuffle=False)

    ntrain = get_param(params, "ntrain", None)
    if ntrain is not None and ntrain < len(train_dataset):
        train_dataset = random_split(train_dataset, [int(ntrain), len(train_dataset) - int(ntrain)])[0]
    ntest = get_param(params, "ntest", None)
    if ntest is not None and ntest < len(test_dataset):
        test_dataset = random_split(test_dataset, [int(ntest), len(test_dataset) - int(ntest)])[0]
    batch_size = int(get_param(params, "batch_size", 1))
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=1, shuffle=False),
        normalizer,
        dataset.mesh,
    )


def make_structure_sequences(velocity, pressure, displacement, mask_all, params, mu, x1, x2):
    """Return raw structural-boundary windows before normalization."""
    t_in = int(get_param(params, "structure_t_in", 10))
    boundary = torch.cat(
        [velocity[:, mask_all, :2], pressure[:, mask_all, :1], displacement[:, mask_all, :2]],
        dim=-1,
    )
    static = torch.tensor([float(mu), float(x1), float(x2)], dtype=boundary.dtype)
    static = static.view(1, 1, 3).expand(boundary.shape[0], boundary.shape[1], 3)
    boundary = torch.cat([boundary, static], dim=-1)
    boundary = _time_slice(boundary, params)

    xs, ys = [], []
    for idx in range(boundary.shape[0] - t_in):
        xs.append(boundary[idx:idx + t_in])
        ys.append(boundary[idx + t_in])
    return torch.stack(xs), torch.stack(ys)


def build_structure_case_sequences(params, x2, normalizer, mask_all=None, mu=None, x1=None):
    """Build normalized structural-boundary windows for a single rollout case."""
    dataset = FsiDataset(get_param(params, "data_location"), get_param(params, "input_mesh_location"), params)
    if mask_all is None:
        masks = load_boundary_indices(infer_boundary_mask_path(params))
        mask_all = masks["all"]
    mu = float(mu if mu is not None else get_param(params, "mu_list", [1.0])[0])
    x1 = float(x1 if x1 is not None else get_param(params, "supervised_inlets_x1", [-4.0])[0])
    velocity, pressure, displacement = dataset.read_case(mu, x1, float(x2))
    x_raw, y_raw = make_structure_sequences(velocity, pressure, displacement, mask_all, params, mu, x1, x2)
    x_norm = normalizer(x_raw)
    y_norm = normalizer(y_raw)[..., STRUCTURE_TARGET_CHANNELS]
    return x_raw, y_raw, x_norm, y_norm


def load_boundary_indices(mask_path):
    with h5py.File(mask_path, "r") as h5f:
        mask_down = np.asarray(h5f["mask_down_index_c"], dtype=np.int64)
        mask_up = np.asarray(h5f["mask_up_index_c"], dtype=np.int64)
        mask_right = np.asarray(h5f["mask_r_index_c"], dtype=np.int64)
    return {
        "down": mask_down,
        "up": mask_up,
        "right": mask_right,
        "all": np.concatenate([mask_down, mask_up, mask_right], axis=0),
    }


def infer_boundary_mask_path(params):
    explicit = get_param(params, "boundary_mask_path", None)
    if explicit:
        return resolve_path(explicit)
    data_location = resolve_path(get_param(params, "data_location"))
    mu = get_param(params, "mu_list", [1.0])[0]
    x1 = get_param(params, "supervised_inlets_x1", [-4.0])[0]
    x2_values = get_param(params, "supervised_inlets_x2_test", None) or get_param(params, "supervised_inlets_x2", [6.0])
    x2 = x2_values[-1]
    return data_location / f"mu={float(mu)}" / f"x1={float(x1)}" / f"x2={float(x2)}" / "mask_index_points_boundary_all" / "mask_index_points_boundary_all.h5"


def build_structure_dataloaders(params):
    dataset = FsiDataset(get_param(params, "data_location"), get_param(params, "input_mesh_location"), params)
    masks = load_boundary_indices(infer_boundary_mask_path(params))
    mask_all = masks["all"]
    train_x2 = get_param(params, "supervised_inlets_x2_train", [-4.0, 0.0, 2.0, 4.0])
    test_x2 = get_param(params, "supervised_inlets_x2_test", [-2.0, 6.0])
    t_in = int(get_param(params, "structure_t_in", 10))

    def collect(x2_values):
        xs, ys = [], []
        for mu in get_param(params, "mu_list", [1.0]):
            for x1 in get_param(params, "supervised_inlets_x1", [-4.0]):
                for x2 in x2_values:
                    velocity, pressure, displacement = dataset.read_case(mu, x1, x2)
                    x_case, y_case = make_structure_sequences(
                        velocity, pressure, displacement, mask_all, params, mu, x1, x2
                    )
                    xs.append(x_case)
                    ys.append(y_case)
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    x_train_raw, y_train_raw = collect(train_x2)
    mean = torch.mean(x_train_raw, dim=(0, 1, 2))
    std = torch.var(x_train_raw, dim=(0, 1, 2)).sqrt()
    normalizer = Normalizer(mean, std)

    def flatten(x_raw, y_raw):
        x_norm = normalizer(x_raw)
        y_norm = normalizer(y_raw)[..., STRUCTURE_TARGET_CHANNELS]
        n_samples, _, n_points, n_vars = x_norm.shape
        return (
            x_norm.reshape(n_samples, t_in, n_points * n_vars).float(),
            y_norm.reshape(n_samples, n_points * len(STRUCTURE_TARGET_CHANNELS)).float(),
        )

    x_train, y_train = flatten(x_train_raw, y_train_raw)
    x_test_raw, y_test_raw = collect(test_x2)
    x_test, y_test = flatten(x_test_raw, y_test_raw)
    ntrain = get_param(params, "structure_ntrain", None)
    if ntrain is not None:
        x_train = x_train[:int(ntrain)]
        y_train = y_train[:int(ntrain)]
    ntest = get_param(params, "structure_ntest", None)
    if ntest is not None:
        x_test = x_test[:int(ntest)]
        y_test = y_test[:int(ntest)]
    batch_size = int(get_param(params, "structure_batch_size", 1024))
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
        normalizer,
        {
            "n_points": x_train_raw.shape[2],
            "n_vars": x_train_raw.shape[3],
            "t_in": t_in,
            "mask_all": mask_all,
            "fixed_points": fixed_boundary_indices(dataset.mesh),
        },
    )
