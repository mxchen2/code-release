# KnowStroke

Anonymous code release for the ICDM 2026 submission of
**KnowStroke: Bilateral Dual-Channel Graph Learning with Task-Decoupled
Contrastive Supervision for Joint Post-EVT Risk Prediction**.

## Main Result

Five-fold patient-stratified cross-validation, mean ± std (%):

| HT AUC | Edema AUC | Mean AUC | Macro AP | Macro F1 | Macro BACC |
|---:|---:|---:|---:|---:|---:|
| 70.21 ± 5.80 | 74.73 ± 4.89 | 72.47 ± 3.51 | 72.57 ± 3.42 | 66.06 ± 3.05 | 67.52 ± 3.01 |

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
src/
├── knowstroke.py               # main KnowStroke model
├── topological_channel.py      # topology channel (GCN)
├── symmetry_channel.py         # symmetry channel (pair embedding)
├── raw_pair_symmetry.py        # raw-pair symmetry variant
├── channel_fusion.py           # gated dual-channel fusion
├── task_pooling.py             # task-specific attention pooling
├── cross_attention_pooling.py  # cross-attention readout
├── cross_task.py               # stop-gradient cross-task module
├── contrastive.py              # task-specific contrastive losses
├── graph_dataset.py            # graph dataset loading
└── train.py                    # training entry point
configs/
├── run_config.json             # default hyperparameters
└── reproduce.sh                # 5-fold reproduction script
sample_inputs/
└── 1.nii                       # one atlas-registered NCCT sample
```

## Data

The full multi-center cohort, patient features, fold splits, and
atlas specifications are withheld to protect patient privacy. For
input format demonstration, one atlas-registered, brain-extracted
NCCT sample is provided under `sample_inputs/`. Cohort statistics,
preprocessing, atlas construction, and graph definition details are
described in Section III of the paper.

## Training Entry Point

The training script is invoked as a module:

```bash
python -m src.train [args]
```

All hyperparameters used for the main result are listed in
`configs/run_config.json` and as command-line arguments in
`configs/reproduce.sh`.

Note: `--num-gat-layers 2` is a legacy parameter name. Under
`--topo-conv-type gcn` it specifies two layers of GCN message
passing, not GAT.

## Key Hyperparameters

| Parameter | Value |
|---|---:|
| feature_subset | full100 |
| feature_norm | zscore |
| topo_conv_type | gcn |
| num_gat_layers | 2 |
| d_h | 48 |
| dropout | 0.4 |
| lr | 0.0016 |
| weight_decay | 0.0001 |
| batch_size | 16 |
| lambda_ht | 2.0 |
| lambda2 | 0.16 |
| lambda2_warmup_epochs | 10 |
| contrastive_mode | soft_pair_supcon |
| contrastive_loss_weight_hem | 1.5 |
| contrastive_loss_weight_ede | 0.8 |
| fusion_mode | gate |
| fusion_gate_init | 0.85 |
| pooling_type | simple |
| pooling_mode | task_specific |
| head_type | mlp |
| cross_task | enabled |
| cross_gate_init | 0.1 |
| cross_scale | 1.5 |
| cross_scale_edema_to_ht | 1.0 |
| cross_scale_ht_to_edema | 0.5 |
| seed | 21 |
| selection_metric | mean_auc |

## Computing Infrastructure

All experiments were run on a single Linux workstation with:

- GPU: NVIDIA GeForce RTX 3090 (24 GB)
- CPU: AMD EPYC 7502P 32-core
- RAM: 128 GB
- Python 3.10, PyTorch 2.9, PyTorch Geometric 2.7

On this setup, KnowStroke trains at ~0.89 s/epoch and infers at
~6.62 ms per case.

## Notation

In the code, variables prefixed with `hem` correspond to the
Hemorrhagic Transformation (HT) task in the paper, and variables
prefixed with `ede` correspond to the cerebral Edema task.

## License

For double-blind review purposes only.
