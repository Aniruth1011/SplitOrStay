#!/usr/bin/env bash
set -euo pipefail
# Read-only scheduler discovery is permitted on the login node.
sinfo -o '%P %N %G %m %c %l'
if command -v noderes >/dev/null 2>&1; then
  noderes -c -p gpu
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  hostname
  nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv
  nvidia-smi topo -m
else
  echo 'GPU inspection deferred: run this script inside an allocation.'
fi

