#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Grok 4.1 Fast..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p xai -m grok-4-1-fast-reasoning \
  -o results/results_grok.json \
  -f results/fingerprint_grok.json
echo "Done: results/fingerprint_grok.json"
