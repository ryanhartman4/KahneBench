#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source run/common.sh

echo "GLM 4.7..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/glm-4p7 \
  -o results/results_glm47.json \
  -f results/fingerprint_glm47.json
echo "Done: results/fingerprint_glm47.json"
