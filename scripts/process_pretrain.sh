#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python run.py \
    --input "${INPUT:-data}" \
    --output "${OUTPUT:-output}" \
    --mode pretrain \
    --min_length "${MIN_LENGTH:-32}"
