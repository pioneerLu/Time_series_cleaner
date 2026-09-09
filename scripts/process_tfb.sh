#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python run.py \
    --input "${INPUT:-data/data.npy}" \
    --output "${OUTPUT:-output}" \
    --mode full \
    --keep_temp_files \
    --dataset_name "${DATASET_NAME:-dataset}"
