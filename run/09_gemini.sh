#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "Gemini 3 Pro..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p gemini -m gemini-3-pro-preview \
  -o results/results_gemini.json \
  -f results/fingerprint_gemini.json
echo "Done: results/fingerprint_gemini.json"
