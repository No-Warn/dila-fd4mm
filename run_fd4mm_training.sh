#!/usr/bin/env bash
# =============================================================================
# FD4MM -> DiLA training launcher
#
# Encapsulates every prerequisite discovered during setup:
#   1. proxy env sanity   (ALL_PROXY=socks:// breaks the hf CLI / httpx)
#   2. HF_HUB_CACHE       (RAE loads DINOv2 offline from ./pretrained)
#   3. asset + data preflight checks
#   4. FD4MM data verification
#   5. accelerate launch
#
# Usage:
#   bash run_fd4mm_training.sh                    # 200-step smoke run
#   NUM_TRAIN_STEPS=30000 BATCH_SIZE=2 EXP_NAME=DiLA-fd4mm \
#     bash run_fd4mm_training.sh                  # full run
#
# Common overrides (all optional):
#   SEQ_LEN=3 BATCH_SIZE=2 VAL_BATCH_SIZE=1 NUM_WORKERS=4
#   NUM_TRAIN_STEPS=200 VAL_EVERY=100 MIXED_PRECISION=bf16
#   WORK_DIR=./checkpoints EXP_NAME=fd4mm-smoke DATA_ROOT=./data/train
#   FRAME_FILL=reflect | SEQUENCE_MODE=acb
#   WANDB=1                 (enable wandb; default disabled)
#   ALLOW_CPU=1             (run on CPU when no GPU is visible)
#   SKIP_VERIFY=1 PROXY=http://127.0.0.1:7897
# =============================================================================
set -euo pipefail

# ----------------------------- parameters ------------------------------------
SEQ_LEN=${SEQ_LEN:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-30000}
VAL_EVERY=${VAL_EVERY:-100}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
WORK_DIR=${WORK_DIR:-./checkpoints}
EXP_NAME=${EXP_NAME:-fd4mm-dila}
DATA_ROOT=${DATA_ROOT:-./data/train}
SEQUENCE_MODE=${SEQUENCE_MODE:-acb}
FRAME_FILL=${FRAME_FILL:-reflect}
WANDB=${WANDB:-0}
ALLOW_CPU=${ALLOW_CPU:-0}
SKIP_VERIFY=${SKIP_VERIFY:-0}
PROXY=${PROXY:-http://127.0.0.1:7897}

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

say() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
die() { printf '\n\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# Pick an interpreter that can actually import torch (prefer the active conda env).
pick_python() {
  local cand
  for cand in "${PY:-}" "${CONDA_PREFIX:-}/bin/python" python python3; do
    [ -n "$cand" ] || continue
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch' >/dev/null 2>&1; then
      command -v "$cand"; return 0
    fi
  done
  return 1
}
PY="$(pick_python)" || die "no Python with torch found. Activate the env first:  conda activate dila"
# keep accelerate from the same environment as PY
if [ -x "$(dirname "$PY")/accelerate" ]; then
  ACCELERATE="$(dirname "$PY")/accelerate"
else
  ACCELERATE="accelerate"
fi

# ----------------------------- 1. environment --------------------------------
say "Step 1/5  environment"
# httpx (used by huggingface_hub) rejects the 'socks://' scheme, and an entry
# like '[::1]' inside NO_PROXY makes it raise InvalidURL. Clean both.
unset ALL_PROXY all_proxy no_proxy NO_PROXY || true
if [ -n "$PROXY" ]; then
  export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY" http_proxy="$PROXY" https_proxy="$PROXY"
  echo "proxy       : $PROXY"
fi
export HF_HUB_CACHE="$REPO_DIR/pretrained"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "alloc conf  : $PYTORCH_CUDA_ALLOC_CONF"

# ----------------------------- 2. preflight ----------------------------------
say "Step 2/5  preflight checks"

[ -d "$DATA_ROOT/frameA" ] || die "missing $DATA_ROOT/frameA"
[ -d "$DATA_ROOT/frameB" ] || die "missing $DATA_ROOT/frameB"
n_a=$(ls "$DATA_ROOT/frameA" | wc -l)
n_b=$(ls "$DATA_ROOT/frameB" | wc -l)
echo "data        : $DATA_ROOT  ($n_a frameA / $n_b frameB)"
[ -d "$DATA_ROOT/frameC" ] && echo "              frameC present -> genuine A->C->B axis available" \
                           || echo "              WARNING: no frameC -> falling back to A->B"

for f in \
  "pretrained/models--facebook--dinov2-with-registers-base" \
  "pretrained/stats/dinov2/wReg_base/imagenet1k/stat.pt" \
  "pretrained/decoders/dinov2/wReg_base/ViTXL/dinov2_decoder.pt" \
  "pretrained/rae/dinov2_stats.pt" \
  "pretrained/rae/dinov2_decoder.pt" ; do
  [ -e "$f" ] || die "missing pretrained asset: $f
  Re-download with the commands in FD4MM_MIGRATION.md section 10.1"
done
echo "weights     : encoder + stats + decoder OK"

CUDA_OK=$($PY - <<'EOF'
import torch
print(1 if torch.cuda.is_available() else 0)
EOF
)
if [ "$CUDA_OK" = "1" ]; then
  $PY - <<'EOF'
import torch
print(f"gpu         : {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory/2**30:.2f} GiB)")
EOF
  LAUNCH_ARGS=(--mixed_precision "$MIXED_PRECISION")
else
  if [ "$ALLOW_CPU" = "1" ]; then
    echo "gpu         : none visible -> ALLOW_CPU=1, running on CPU (slow)"
    LAUNCH_ARGS=(--cpu --num_processes 1 --mixed_precision no)
  else
    die "CUDA is not available.
  * if this shell hides /dev/nvidia*, run from a normal terminal
  * otherwise install a Blackwell-capable build:
      pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
  set ALLOW_CPU=1 to force a (very slow) CPU run."
  fi
fi

# ----------------------------- 3. data verify --------------------------------
if [ "$SKIP_VERIFY" = "1" ]; then
  say "Step 3/5  data verification (skipped)"
else
  say "Step 3/5  data verification"
  $PY verify_fd4mm_dila.py --root "$DATA_ROOT" --seq-len "$SEQ_LEN" \
      --sequence-mode "$SEQUENCE_MODE" --frame-fill "$FRAME_FILL"
fi

# ----------------------------- 4. summary ------------------------------------
say "Step 4/5  run configuration"
cat <<EOF
  data config   : data=fd4mm
  seq_len (T)   : $SEQ_LEN
  sequence_mode : $SEQUENCE_MODE   (acb = genuine A->C->B)
  frame_fill    : $FRAME_FILL      (used only when T > 3)
  batch_size    : $BATCH_SIZE   (val $VAL_BATCH_SIZE)
  steps         : $NUM_TRAIN_STEPS   (validate every $VAL_EVERY)
  precision     : $MIXED_PRECISION
  work_dir      : $WORK_DIR/$EXP_NAME
  wandb         : $([ "$WANDB" = "1" ] && echo enabled || echo disabled)
EOF

# Per-step checkpoint disk usage is ~1.4 GB (model + optimizer states).
mkdir -p "$WORK_DIR"

if [ "${DRY_RUN:-0}" = "1" ]; then
  say "DRY_RUN=1 -> preflight only, not launching training"
  exit 0
fi

# ----------------------------- 5. train --------------------------------------
say "Step 5/5  launching training"
set -x
"$ACCELERATE" launch "${LAUNCH_ARGS[@]}" main.py \
  data=fd4mm \
  seq_len="$SEQ_LEN" \
  batch_size="$BATCH_SIZE" \
  val_batch_size="$VAL_BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  num_train_steps="$NUM_TRAIN_STEPS" \
  val_every_n_steps="$VAL_EVERY" \
  data.multidataset.fd4mm.sequence_mode="$SEQUENCE_MODE" \
  data.multidataset.fd4mm.frame_fill="$FRAME_FILL" \
  RAE.park_decoder_on_cpu=true \
  wandb.enabled=$([ "$WANDB" = "1" ] && echo true || echo false) \
  work_dir="$WORK_DIR" \
  exp_name="$EXP_NAME"
set +x

say "Done"
echo "checkpoints : $WORK_DIR/phase1/${EXP_NAME}*.pt"
echo "recon grids : $WORK_DIR/reconstruction/"
