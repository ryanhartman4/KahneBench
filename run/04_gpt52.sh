#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "GPT-5.2..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p openai -m gpt-5.2-2025-12-11 \
  -o results/results_gpt52.json \
  -f results/fingerprint_gpt52.json
echo "Done: results/fingerprint_gpt52.json"
