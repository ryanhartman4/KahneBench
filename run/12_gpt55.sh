#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "GPT-5.5..."
uv run kahne-bench evaluate $COMMON_ARGS --verbose \
  -p openai -m gpt-5.5-2026-04-23 \
  -o results/results_gpt55.json \
  -f results/fingerprint_gpt55.json
echo "Done: results/fingerprint_gpt55.json"
