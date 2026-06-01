#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Opus 4.8..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS --verbose \
  -p anthropic -m claude-opus-4-8 \
  -o results/results_opus48.json \
  -f results/fingerprint_opus48.json
echo "Done: results/fingerprint_opus48.json"
