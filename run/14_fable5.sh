#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Fable 5..."
uv run kahne-bench evaluate $COMMON_ARGS --verbose \
  -p anthropic -m claude-fable-5 \
  -o results/results_fable5.json \
  -f results/fingerprint_fable5.json
echo "Done: results/fingerprint_fable5.json"
