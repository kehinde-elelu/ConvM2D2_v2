#!/usr/bin/env bash
set -euo pipefail

# ---- User Config ----
VENV_PATH="env"                               # virtualenv or conda env dir (adjust)
PYTHON_BIN="$VENV_PATH/bin/python"            # python executable
PREDICT_SCRIPT="src/predict.py"

# Data / lists
TEST_LIST="data/main/DATA/sets/test_mos_list.txt"
WAV_DIR="data/main/DATA/wav"

# Output directory
OUT_DIR="result"

# Checkpoint (head + q_hat from training)
CHECKPOINT_M2D="models/checkpoint_head_m2d.pt"
CHECKPOINT_W2V="models/checkpoint_head_wav2vec.pt"

# M2D weight file
M2D_WEIGHT="/egr/research-deeptech/elelukeh/MOS_project/M2D/m2d/models_m2d/m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth"

# HF cache & model id for wav2vec2
HF_CACHE_DIR="models/hf"
WAV2VEC_MODEL="facebook/wav2vec2-large-960h"

# Common runtime args
BATCH_SIZE=8
NUM_WORKERS=2
COND_BINS=4
USE_TRANSFORMER_HEAD=1   # 1 to enable --use_transformer_head

# ---- Parse optional first arg: upstream ----
UPSTREAM="${1:-m2d}"   # m2d or wav2vec

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN"
  exit 1
fi

[[ -d "$OUT_DIR" ]] || mkdir -p "$OUT_DIR"
[[ -d "$HF_CACHE_DIR" ]] || mkdir -p "$HF_CACHE_DIR"

HEAD_FLAG=""
if [[ "$USE_TRANSFORMER_HEAD" == "1" ]]; then
  HEAD_FLAG="--use_transformer_head"
fi

echo "[INFO] Using upstream=$UPSTREAM"

if [[ "$UPSTREAM" == "m2d" ]]; then
  CHECKPOINT="$CHECKPOINT_M2D"
  CMD=(
    "$PYTHON_BIN" "$PREDICT_SCRIPT"
    --test_list "$TEST_LIST"
    --wav_dir "$WAV_DIR"
    --checkpoint "$CHECKPOINT"
    --weight "$M2D_WEIGHT"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --cond_bins "$COND_BINS"
    --upstream m2d
    $HEAD_FLAG
    --out_result "$OUT_DIR"
  )
elif [[ "$UPSTREAM" == "wav2vec" ]]; then
  CHECKPOINT="$CHECKPOINT_W2V"
  CMD=(
    "$PYTHON_BIN" "$PREDICT_SCRIPT"
    --test_list "$TEST_LIST"
    --wav_dir "$WAV_DIR"
    --checkpoint "$CHECKPOINT"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --cond_bins "$COND_BINS"
    --upstream wav2vec
    --wav2vec_model "$WAV2VEC_MODEL"
    --hf_cache_dir "$HF_CACHE_DIR"
    $HEAD_FLAG
    --out_result "$OUT_DIR"
  )
else
  echo "Unsupported upstream: $UPSTREAM (use m2d or wav2vec)"
  exit 1
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"



# chmod +x predict.sh
# ./predict.sh m2d
# ./predict.sh wav2vec