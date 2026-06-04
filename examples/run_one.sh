#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_DIR="${ROOT}/examples/raw_samples/normal"

cloud="$(find "${SAMPLE_DIR}" -maxdepth 1 -name 'cloud_*.pcd' | head -n 1)"
param="$(find "${SAMPLE_DIR}" -maxdepth 1 -name 'param*.txt' | head -n 1)"
result="$(find "${SAMPLE_DIR}" -maxdepth 1 -name 'result_*.txt' | head -n 1)"

python "${ROOT}/fineloc_infer.py" \
  --cloud "${cloud}" \
  --param "${param}" \
  --result "${result}" \
  --device cpu
