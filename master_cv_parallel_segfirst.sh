#!/bin/bash
# Approach B run: segment-then-classify cascade experiment
# (--seg-first-cascade), otherwise identical to master_cv_parallel.sh
# (GPU-count-aware batching, RAM-cache staging, same 300-epoch/5-fold/
# class-balanced recipe) so results are directly comparable to the baseline
# and Approach A reruns. Writes to SEPARATE output directories
# (checkpoints_cv_segfirst / gradcam_cv_segfirst / *_segfirst.log) so
# nothing here can overwrite either prior run's artifacts.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -lt 1 ]; then
  NUM_GPUS=1
fi
set -euo pipefail
cd /workspace/minisegcaps/src

CV_ROOT=/workspace/minisegcaps/data/picai_cache_cv
RAMDISK=/dev/shm/picai_cache_cv
CKPT_ROOT=/workspace/minisegcaps/checkpoints_cv_segfirst
LOG_ROOT=/workspace/minisegcaps/logs
GRADCAM_ROOT=/workspace/minisegcaps/gradcam_cv_segfirst
mkdir -p "$CKPT_ROOT" "$LOG_ROOT" "$GRADCAM_ROOT"

echo "=== PREPROCESS START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [ ! -f "${CV_ROOT}/all_manifest.csv" ]; then
  python3 preprocess_picai_cv.py --out-root "$CV_ROOT" > "${LOG_ROOT}/preprocess_picai_cv.log" 2>&1
else
  echo "all_manifest.csv already exists, skipping preprocess"
fi
if [ ! -d "${CV_ROOT}/fold4/valid" ]; then
  python3 build_cv_folds.py --cv-root "$CV_ROOT" > "${LOG_ROOT}/build_cv_folds.log" 2>&1
else
  echo "fold dirs already exist, skipping build_cv_folds"
fi
echo "=== PREPROCESS DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "=== RAMDISK STAGE START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if ! mkdir -p "$RAMDISK" 2>/dev/null; then
  echo "WARNING: could not use /dev/shm -- falling back to the on-disk cache, no RAM speedup"
  RAMDISK="$CV_ROOT"
fi
if [ "$RAMDISK" != "$CV_ROOT" ]; then
  mkdir -p "$RAMDISK"
  for k in 0 1 2 3 4; do
    if [ ! -d "${RAMDISK}/fold${k}" ]; then
      echo "staging fold${k} into tmpfs..."
      cp -r "${CV_ROOT}/fold${k}" "${RAMDISK}/fold${k}"
    fi
  done
fi
echo "=== RAMDISK STAGE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

run_fold () {
  local k=$1
  local gpu=$2
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "=== FOLD $k (GPU $gpu) SEGFIRST TRAIN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  python3 train.py \
    --data-root "${RAMDISK}/fold${k}" \
    --balanced --epochs 300 --batch-size 256 --num-workers 12 \
    --seg-first-cascade \
    --checkpoint-dir "${CKPT_ROOT}/fold${k}" \
    > "${LOG_ROOT}/train_cv_segfirst_fold${k}.log" 2>&1
  echo "=== FOLD $k SEGFIRST TRAIN DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  python3 evaluate.py \
    --data-root "${RAMDISK}/fold${k}" \
    --checkpoint "${CKPT_ROOT}/fold${k}/best.pt" \
    --split valid \
    --seg-first-cascade \
    > "${LOG_ROOT}/evaluate_cv_segfirst_fold${k}.log" 2>&1
  echo "=== FOLD $k SEGFIRST EVAL DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

  python3 run_gradcam_validation.py \
    --data-root "${RAMDISK}/fold${k}/valid" \
    --checkpoint "${CKPT_ROOT}/fold${k}/best.pt" \
    --out-dir "${GRADCAM_ROOT}/fold${k}" \
    --n-examples -1 \
    --seg-first-cascade \
    > "${LOG_ROOT}/gradcam_cv_segfirst_fold${k}.log" 2>&1
  echo "=== FOLD $k SEGFIRST GRADCAM DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
}

echo "NUM_GPUS=$NUM_GPUS -- batching 5 folds into rounds of $NUM_GPUS"
folds=(0 1 2 3 4)
batch_num=0
for ((i=0; i<${#folds[@]}; i+=NUM_GPUS)); do
  batch_num=$((batch_num + 1))
  batch=("${folds[@]:i:NUM_GPUS}")
  echo "=== BATCH $batch_num: folds ${batch[*]} START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  gpu=0
  for k in "${batch[@]}"; do
    run_fold "$k" "$gpu" &
    gpu=$((gpu + 1))
  done
  wait
  echo "=== BATCH $batch_num DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
done

echo "ALL_FOLDS_SEGFIRST_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
