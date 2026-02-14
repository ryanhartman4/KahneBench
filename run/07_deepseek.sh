#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "DeepSeek V3.2..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/deepseek-v3p2 \
  -o results/results_deepseek.json \
  -f results/fingerprint_deepseek.json
echo "Done: results/fingerprint_deepseek.json"
