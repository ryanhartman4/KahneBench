#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Sonnet 4.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-sonnet-4-5 \
  -o results/results_sonnet45.json \
  -f results/fingerprint_sonnet45.json
echo "Done: results/fingerprint_sonnet45.json"
