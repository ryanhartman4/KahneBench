#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Claude Opus 4.7..."
uv run kahne-bench evaluate $COMMON_ARGS --verbose \
  -p anthropic -m claude-opus-4-7 \
  -o results/results_opus47.json \
  -f results/fingerprint_opus47.json
echo "Done: results/fingerprint_opus47.json"
