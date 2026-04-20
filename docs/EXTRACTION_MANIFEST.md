# Extraction Manifest

Source folder:

`codano-mf-dataset-split-longTermTrain-generation/CoDA-NO`

## Paper Components

| Paper component | Extracted code | Original source |
| --- | --- | --- |
| Fluid GNO-ViT-GNO | `models/vit.py`, `layers/gno_layer.py`, `layers/regular_transformer.py` | `models/vit.py`, `layers/gno_layer.py`, `layers/regular_transformer.py` |
| Pure GNO ablation | `models/gnn.py`, `layers/gnn_layer.py` | `models/gnn.py`, `layers/gnn_layer.py` |
| Structure LSTM | `models/structure_lstm.py`, `train/structure_trainer.py` | `predict_structure_displacements_LSTM_x_y_u_v.ipynb` |
| ALE mesh update | `mesh/ale.py` | `FSI_LongTermTrain*.ipynb`, `plate_flexible_deformation_biharmonic_extrapolation.py` |
| Coupled long-term FSI loop | `train/coupled.py`, `scripts/evaluate_coupled_rollout.py`, `scripts/train_coupled.py` | `FSI_LongTermTrain_nonperiodic_test.ipynb` |
| Boundary correction | `train/trainer.py::_boundary_correct` | `train/trainer.py`, `main_train.py` |
| Fluid dataloading | `data_utils/data_loaders.py` | `data_utils/data_loaders.py`, `main_train.py` |
| Configs for paper runs | `configs/fsi_gno_vit.yaml` | `config/ssl_ns_elastic.yaml` |

## Kept Files

- `YParams.py`
- `baseline_utlis/rigid_neighbor.py`
- `layers/gno_layer.py`
- `layers/gnn_layer.py`
- `layers/regular_transformer.py`
- `layers/regrider.py`
- `layers/lstm.py`
- `models/vit.py`
- `models/gnn.py`
- `models/model_helpers.py`
- `train/new_adam.py`

The copied GNO/GNN layers were lightly patched so grid updates are device-aware and do not hard-code `.cuda()`.

## Rewritten Or Newly Extracted Files

- `data_utils/data_loaders.py`
- `models/get_models.py`
- `models/structure_lstm.py`
- `mesh/ale.py`
- `train/trainer.py`
- `train/structure_trainer.py`
- `train/coupled.py`
- `scripts/train_fluid.py`
- `scripts/train_structure.py`
- `scripts/rollout_fsi.py`
- `scripts/evaluate_coupled_rollout.py`
- `scripts/train_coupled.py`
- `configs/fsi_gno_vit.yaml`

## Excluded From Open-Source Tree

- `wandb/` logs and metadata.
- W&B API key file.
- all `.ipynb` notebooks with embedded outputs.
- large visualization files: `*.gif`, `*.svg`, `*.pptx`.
- generated meshes and local scratch files except `square_with_circle_hole.msh`, which is kept as the ALE reference mesh.
- trained weights in `weights_temp/` and `weights_last/`.
- RB configs and old CoDA-NO pretraining code that is not used by this paper path.
- duplicated sweep scripts.

## Config Mapping

| New config | Purpose | Original config lineage |
| --- | --- | --- |
| `fluid_periodic` | Main GNO-ViT fluid model, periodic rollout regime | `vit_NS_split_1_nsequences1_boundarymask_test` / `vitGNO_LSTM_Combined_FSI_generation` |
| `fluid_nonperiodic` | Non-periodic/transient fluid model | `vit_NS_split_1_nsequences1_boundarymask_lrRLP_randomindex_nonperiodic` / `_generation_nonperiodic` |
| `fluid_gno_ablation` | ViT ablation, pure GNO | `gnn_NS_split_new_test` |
| `fluid_no_boundary_correction` | ALE boundary-correction ablation | main fluid config with `boundarymask: false` |
| `structure_periodic` | LSTM structure/interface model | `predict_structure_displacements_LSTM_x_y_u_v.ipynb` |
| `structure_nonperiodic` | Non-periodic LSTM structure model | same notebook with non-periodic slicing/weight naming |
| `coupled_nonperiodic` | Full GNO-ViT + LSTM + ALE long-term coupled training/evaluation | `vitGNO_LSTM_Combined_FSI_generation_nonperiodic` / `FSI_LongTermTrain_nonperiodic_test.ipynb` |
