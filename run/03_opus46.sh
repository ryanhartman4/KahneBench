#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Opus 4.6..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-opus-4-6 \
  -o results/results_opus46.json \
  -f results/fingerprint_opus46.json
echo "Done: results/fingerprint_opus46.json"
