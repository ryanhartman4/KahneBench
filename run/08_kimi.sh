#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Kimi K2.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/kimi-k2p5 \
  -o results/results_kimi.json \
  -f results/fingerprint_kimi.json
echo "Done: results/fingerprint_kimi.json"
