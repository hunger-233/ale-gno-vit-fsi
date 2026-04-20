import math
from dataclasses import dataclass

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from data_utils.data_loaders import load_boundary_indices


@dataclass
class TurekHronGeometry:
    length: float = 2.5
    height: float = 0.41
    cylinder_center: tuple = (0.2, 0.2)
    cylinder_radius: float = 0.05
    plate_x: float = 0.23
    plate_y: float = 0.19
    plate_length: float = 0.37
    plate_height: float = 0.02
    plate_region_padding: float = 2.0


def mesh_neighbors(mesh_file):
    mesh = meshio.read(mesh_file)
    points = mesh.points[:, :2].copy()
    if "triangle" in mesh.cells_dict:
        cells = mesh.cells_dict["triangle"]
    elif "quad" in mesh.cells_dict:
        cells = mesh.cells_dict["quad"]
    else:
        raise ValueError("Mesh must contain triangle or quad cells.")

    neighbors = [[] for _ in range(len(points))]
    for cell in cells:
        for idx in range(len(cell)):
            a = int(cell[idx])
            b = int(cell[(idx + 1) % len(cell)])
            if b not in neighbors[a]:
                neighbors[a].append(b)
            if a not in neighbors[b]:
                neighbors[b].append(a)
    return points, cells, neighbors


def fixed_non_plate_indices(points, geometry):
    fixed = []
    cx, cy = geometry.cylinder_center
    radius = geometry.cylinder_radius
    for idx, (x, y) in enumerate(points):
        on_cylinder = abs(math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius) < 1e-3
        on_inlet = abs(x) < 1e-6
        on_outlet = abs(x - geometry.length) < 1e-6
        on_bottom = abs(y) < 1e-6
        on_top = abs(y - geometry.height) < 1e-6
        if on_cylinder or on_inlet or on_outlet or on_bottom or on_top:
            fixed.append(idx)
    return fixed


def plate_region_indices(points, geometry):
    pad = geometry.plate_region_padding
    indices = []
    for idx, (x, y) in enumerate(points):
        if (
            geometry.plate_x - pad <= x <= geometry.plate_x + geometry.plate_length + pad
            and geometry.plate_y - pad <= y <= geometry.plate_y + geometry.plate_height + pad
        ):
            indices.append(idx)
    return indices


def biharmonic_deform(
    current_points,
    neighbors,
    plate_region,
    fixed_indices,
    flex_plate_indices,
    flex_plate_new_positions,
    alpha=1.0,
):
    """Biharmonic extrapolation mesh update used by the ALE rollout."""
    points = np.asarray(current_points, dtype=np.float64)
    n_nodes, n_dims = points.shape
    displacement = np.zeros_like(points)
    fixed_all = set(fixed_indices) | set(flex_plate_indices)

    for idx, new_pos in zip(flex_plate_indices, flex_plate_new_positions):
        displacement[idx] = np.asarray(new_pos) - points[idx]

    rows, cols, values = [], [], []
    eps = 1e-12
    for idx in range(n_nodes):
        weight_sum = 0.0
        for nbr in neighbors[idx]:
            distance = np.linalg.norm(points[idx] - points[nbr])
            weight = 1.0 / (distance ** alpha + eps)
            rows.append(idx)
            cols.append(nbr)
            values.append(weight)
            weight_sum += weight
        rows.append(idx)
        cols.append(idx)
        values.append(-weight_sum)

    laplacian = sp.csr_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes))
    biharmonic = laplacian.dot(laplacian)

    plate_set = set(plate_region)
    outside_plate = set(range(n_nodes)) - plate_set
    known = np.array(sorted((fixed_all | outside_plate) & plate_set))
    unknown = np.array(sorted(plate_set - set(known)))
    updated = displacement.copy()

    if unknown.size:
        for dim in range(n_dims):
            a_uu = biharmonic[unknown, :][:, unknown]
            a_ub = biharmonic[unknown, :][:, known]
            rhs = -a_ub.dot(displacement[known, dim])
            updated[unknown, dim] = spla.spsolve(a_uu, rhs)

    new_points = points.copy()
    for idx in plate_region:
        new_points[idx] = points[idx] + updated[idx]
    return new_points.astype(np.float32)


class ALEMeshUpdater:
    def __init__(
        self,
        reference_points,
        neighbors,
        plate_boundary,
        boundary_order,
        plate_region,
        fixed_non_plate,
        alpha=1.0,
    ):
        self.reference_points = np.asarray(reference_points, dtype=np.float32)
        self.neighbors = neighbors
        self.plate_boundary = list(map(int, plate_boundary))
        self.boundary_order = list(map(int, boundary_order))
        self.plate_region = list(map(int, plate_region))
        self.fixed_non_plate = list(map(int, fixed_non_plate))
        self.alpha = alpha

    @classmethod
    def from_mesh(cls, mesh_file, boundary_mask_path, geometry=None, alpha=1.0):
        geometry = geometry or TurekHronGeometry()
        points, _, neighbors = mesh_neighbors(mesh_file)
        masks = load_boundary_indices(boundary_mask_path)
        boundary_order = masks["all"].tolist()
        plate_boundary = sorted(set(boundary_order))
        return cls(
            reference_points=points,
            neighbors=neighbors,
            plate_boundary=plate_boundary,
            boundary_order=boundary_order,
            plate_region=plate_region_indices(points, geometry),
            fixed_non_plate=fixed_non_plate_indices(points, geometry),
            alpha=alpha,
        )

    def update(self, current_points, boundary_positions):
        index_map = {node: pos for pos, node in enumerate(self.boundary_order)}
        ordered_boundary = np.asarray([boundary_positions[index_map[node]] for node in self.plate_boundary])
        return biharmonic_deform(
            current_points=current_points,
            neighbors=self.neighbors,
            plate_region=self.plate_region,
            fixed_indices=self.fixed_non_plate,
            flex_plate_indices=self.plate_boundary,
            flex_plate_new_positions=ordered_boundary,
            alpha=self.alpha,
        )
