# An ALE-Consistent Hybrid Graph Neural Operator-Transformer Framework for FSI Prediction

### Paper

**ALE-Consistent Hybrid Graph Neural Operator-Transformer Framework for Fluid-Structure Interaction Prediction**

by `<Shihang Zhao>`, `<Martín Saravia>`, `<Haokui Jiang>`, `<Zhiyang Xue>`, `<Shunxiang Cao>`

This repository contains the official implementation of the ALE-consistent GNO-ViT framework for long-horizon fluid-structure interaction prediction on deforming unstructured meshes.

## Model Architecture

The framework couples three components:

```text
Fluid state and next mesh
        |
        v
GNO lifting -> ViT latent operator -> GNO projection
        |
        v
Boundary-corrected fluid prediction
        |
        v
Structure LSTM interface predictor
        |
        v
ALE biharmonic mesh update
        |
        v
Autoregressive FSI rollout
```

The fluid surrogate predicts velocity and pressure on the moving mesh. The structural surrogate predicts boundary velocity and displacement. The ALE updater deforms the fluid mesh from the predicted flexible-boundary motion, enabling coupled long-term rollout.

## Setup and Tutorials

Clone the project:

```bash
git clone https://github.com/hunger-233/ale-gno-vit-fsi.git
cd ale-gno-vit-fsi
```

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate ale-gno-vit-fsi
```

Alternatively, install dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

Install `ipykernel` if you want to run notebooks:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name=ale-gno-vit-fsi
```

## Run Training Scripts

Training scripts are available under the `scripts` folder.

Fluid GNO-ViT:

```bash
python scripts/train_fluid.py --config fluid_nonperiodic
```

Pure GNO ablation:

```bash
python scripts/train_fluid.py --config fluid_gno_ablation
```

No boundary-correction ablation:

```bash
python scripts/train_fluid.py --config fluid_no_boundary_correction
```

Structure LSTM:

```bash
python scripts/train_structure.py --config structure_periodic
```

Coupled long-term fine-tuning:

```bash
python scripts/train_coupled.py --config coupled_nonperiodic \
  --fluid-weight weights/fluid.pt \
  --structure-weight weights/structure.pt
```

## Inference and Coupled Rollout

Run the full GNO-ViT + LSTM + ALE coupled rollout:

```bash
python scripts/evaluate_coupled_rollout.py --config coupled_nonperiodic \
  --fluid-weight weights/fluid.pt \
  --structure-weight weights/structure.pt \
  --x2 -2.0 \
  --output results/coupled_rollout_x2_neg2_200step.npz
```

## Quick Checks

Use these commands to verify that the installation and data paths are working:

```bash
python scripts/train_fluid.py --config fluid_periodic --epochs 1 --ntrain 1 --ntest 1
python scripts/train_structure.py --config structure_periodic --epochs 1 --ntrain 2 --ntest 2
python scripts/train_coupled.py --config coupled_nonperiodic --epochs 1 --horizon 1 --min-step-start 12
```

To check only the ALE mesh updater:

```bash
python scripts/rollout_fsi.py \
  --mesh square_with_circle_hole.msh \
  --boundary-mask path/to/mask_index_points_boundary_all.h5
```

## Repository Structure

```text
configs/          Experiment configs
data_utils/       HDF5 readers, normalization, dataset builders
layers/           GNO, GNN, transformer, regridding layers
mesh/             ALE mesh deformation utilities
models/           Fluid GNO-ViT/GNO and structure LSTM models
scripts/          Training and evaluation entry points
train/            Fluid, structure, and coupled training loops
docs/             Extraction notes and mapping from notebooks
```

## License

This project is released under the MIT License. See `LICENSE` for details.
