import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the ALE mesh updater used inside long-horizon FSI rollouts."
    )
    parser.add_argument("--mesh", required=True, help="Reference .msh file, e.g. square_with_circle_hole.msh")
    parser.add_argument("--boundary-mask", required=True, help="mask_index_points_boundary_all.h5")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    from mesh.ale import ALEMeshUpdater

    updater = ALEMeshUpdater.from_mesh(args.mesh, args.boundary_mask, alpha=args.alpha)
    print("ALE mesh updater ready")
    print(f"nodes={len(updater.reference_points)}")
    print(f"moving_boundary_nodes={len(updater.plate_boundary)}")
    print(f"deformable_region_nodes={len(updater.plate_region)}")
    print("Use updater.update(current_points, predicted_boundary_positions) inside the coupled rollout loop.")


if __name__ == "__main__":
    main()
