#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Haiku 4.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-haiku-4-5 \
  -o results/results_haiku45.json \
  -f results/fingerprint_haiku45.json
echo "Done: results/fingerprint_haiku45.json"
