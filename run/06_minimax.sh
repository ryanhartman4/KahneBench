#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "MiniMax M2P1..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/minimax-m2p1 \
  -o results/results_minimax.json \
  -f results/fingerprint_minimax.json
echo "Done: results/fingerprint_minimax.json"
