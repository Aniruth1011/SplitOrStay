#!/usr/bin/env bash
# Untested CARC template. Supply account/partition/GPU type/memory/CPU/time via sbatch CLI.
# Example after verifying TYPE with noderes:
# sbatch --account=ACCOUNT --partition=gpu --nodes=1 --ntasks=1 \
#   --gpus-per-task=TYPE:1 --cpus-per-task=8 --mem=32G --time=00:30:00 slurm/worker.sbatch
set -euo pipefail
: "${SLURM_JOB_ID:?Must execute inside a Slurm job}"
: "${HETEROKV_MODEL:?Set the selected model ID}"
: "${HETEROKV_REVISION:?Set an immutable model commit}"
: "${HETEROKV_ENGINE_PYTHON:?Set the Python executable in the separately pinned GPU environment}"
: "${HETEROKV_ENGINE_VLLM:?Set the vllm executable in the pinned GPU environment}"
: "${HETEROKV_CACHE:?Set model cache on approved project storage}"
: "${HETEROKV_VLLM_VERSION:?Set the exact release validated for this environment}"
export HF_HOME="$HETEROKV_CACHE"
export VLLM_NO_USAGE_STATS=1
"$HETEROKV_ENGINE_PYTHON" -c 'import os,importlib.metadata; assert importlib.metadata.version("vllm") == os.environ["HETEROKV_VLLM_VERSION"]'
bash scripts/discover.sh
# One project process, TP=1, no automatic mixed allocation assumptions.
srun --ntasks=1 "$HETEROKV_ENGINE_VLLM" serve "$HETEROKV_MODEL" \
  --revision "$HETEROKV_REVISION" \
  --tokenizer-revision "$HETEROKV_REVISION" --tensor-parallel-size 1 \
  --dtype bfloat16 --kv-cache-dtype auto --enable-chunked-prefill \
  --no-enable-prefix-caching --gpu-memory-utilization "${HETEROKV_MEMORY_FRACTION:-0.90}" \
  --max-model-len "${HETEROKV_CONTEXT:-4096}" \
  --max-num-batched-tokens "${HETEROKV_BATCH_TOKENS:-4096}" \
  --max-num-seqs "${HETEROKV_MAX_SEQS:-8}" \
  --stream-interval 1 --host "${HETEROKV_HOST:-127.0.0.1}" \
  --port "${HETEROKV_PORT:-8000}"
