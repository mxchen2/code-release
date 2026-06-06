#!/usr/bin/env bash
# Reproduction script for the main KnowStroke 5-fold cross-validation result.
# The training entry point expects patient features and fold splits to be
# accessible at run time; those inputs are not redistributed in this release.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTBASE="${OUTBASE:-outputs/knowstroke_seed21}"

COMMON_ARGS=(
  --feature-subset full100
  --feature-norm zscore
  --epochs 100
  --patience 20
  --seed 21
  --topo-conv-type gcn
  --num-gat-layers 2
  --batch-size 16
  --lr 0.0016
  --weight-decay 0.0001
  --dropout 0.4
  --d-h 48
  --lambda-ht 2.0
  --lambda2 0.16
  --lambda2-warmup-epochs 10
  --cross-scale 1.5
  --cross-scale-edema-to-ht 1.0
  --cross-scale-ht-to-edema 0.5
  --cross-gate-init 0.1
  --fusion-gate-init 0.85
  --contrastive-mode soft_pair_supcon
  --contrastive-loss-weight-hem 1.5
  --contrastive-loss-weight-ede 0.8
  --contrastive-projector-mode mlp
  --temperature-cl 0.2
  --temperature-th 0.07
  --soft-supcon-lambda-barrier 0.8
  --soft-supcon-lambda-var 2.0
  --soft-supcon-lambda-cov 0.08
  --selection-metric mean_auc
  --pooling-mode task_specific
  --head-type mlp
  --pooling-type simple
  --theta-init-from-data
  --theta-init-percentile 70.0
)

echo "OUTBASE=${OUTBASE}"
for fold in 0 1 2 3 4; do
  echo "Running KnowStroke fold ${fold}"
  "${PYTHON_BIN}" -X utf8 -m src.train \
    "${COMMON_ARGS[@]}" \
    --fold "${fold}" \
    --out-dir "${OUTBASE}/fold_${fold}"
done

echo "Finished. Results saved to ${OUTBASE}"
