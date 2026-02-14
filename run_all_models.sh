#!/usr/bin/env bash
# Kahne-Bench Full 8-Model Evaluation Run
# Date: 2026-02-14
# Config: Core tier, 15 biases, 3 trials, 3 intensities (W/M/S), seed 42
# Shared test cases: core_tests.json
#
# Rate-limit mitigation: concurrency=20, 2 retries with 10s delay
# Judge: anthropic claude-haiku-4-5 (default)
#
# Required environment variables:
#   ANTHROPIC_API_KEY  - Claude models + judge
#   OPENAI_API_KEY     - GPT-5.2
#   FIREWORKS_API_KEY  - GLM, MiniMax, DeepSeek, Kimi
#   GOOGLE_API_KEY     - Gemini 3 Pro
#   XAI_API_KEY        - Grok 4.1 Fast

set -euo pipefail

INPUT="core_tests.json"
TRIALS=3
CONCURRENT=30
RETRIES=2
RETRY_DELAY=5
TIER="core"
COMMON_ARGS="-i $INPUT -n $TRIALS -c $CONCURRENT --rate-limit-retries $RETRIES --rate-limit-retry-delay $RETRY_DELAY --tier $TIER"

mkdir -p results

echo "============================================"
echo "Kahne-Bench Full Model Run"
echo "Input: $INPUT"
echo "Trials: $TRIALS | Concurrency: $CONCURRENT"
echo "Rate-limit retries: $RETRIES (${RETRY_DELAY}s delay)"
echo "============================================"
echo ""

# --- Anthropic Models ---

echo "[1/10] Claude Sonnet 4.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-sonnet-4-5 \
  -o results/results_sonnet45.json \
  -f results/fingerprint_sonnet45.json
echo "  Done."
echo ""

echo "[2/10] Claude Haiku 4.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-haiku-4-5 \
  -o results/results_haiku45.json \
  -f results/fingerprint_haiku45.json
echo "  Done."
echo ""

echo "[3/10] Claude Opus 4.6..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p anthropic -m claude-opus-4-6 \
  -o results/results_opus46.json \
  -f results/fingerprint_opus46.json
echo "  Done."
echo ""

# --- OpenAI ---

echo "[4/10] GPT-5.2..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p openai -m gpt-5.2-2025-12-11 \
  -o results/results_gpt52.json \
  -f results/fingerprint_gpt52.json
echo "  Done."
echo ""

# --- Fireworks (OpenAI-compatible) ---

echo "[5/10] GLM 4.7..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/glm-4p7 \
  -o results/results_glm47.json \
  -f results/fingerprint_glm47.json
echo "  Done."
echo ""

echo "[6/10] MiniMax M2P1..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/minimax-m2p1 \
  -o results/results_minimax.json \
  -f results/fingerprint_minimax.json
echo "  Done."
echo ""

echo "[7/10] DeepSeek V3.2..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/deepseek-v3p2 \
  -o results/results_deepseek.json \
  -f results/fingerprint_deepseek.json
echo "  Done."
echo ""

echo "[8/10] Kimi K2.5..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p fireworks -m accounts/fireworks/models/kimi-k2p5 \
  -o results/results_kimi.json \
  -f results/fingerprint_kimi.json
echo "  Done."
echo ""

# --- Google ---

echo "[9/10] Gemini 3 Pro..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p gemini -m gemini-3-pro-preview \
  -o results/results_gemini.json \
  -f results/fingerprint_gemini.json
echo "  Done."
echo ""

# --- xAI ---

echo "[10/10] Grok 4.1 Fast..."
PYTHONPATH=src uv run kahne-bench evaluate $COMMON_ARGS \
  -p xai -m grok-4-1-fast-reasoning \
  -o results/results_grok.json \
  -f results/fingerprint_grok.json
echo "  Done."
echo ""

echo "============================================"
echo "All 10 models complete!"
echo "Results in: results/"
echo "============================================"

# Quick audit check
echo ""
echo "Running audit checks..."
for f in results/results_*.json; do
  model=$(basename "$f" .json | sed 's/results_//')
  errors=$(python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
results = data.get('results', data) if isinstance(data, dict) else data
errors = [r for r in results if r.get('model_response','').startswith('ERROR:') and r.get('is_biased') is not None]
print(len(errors))
")
  echo "  $model: $errors error responses scored (should be 0)"
done
echo ""
echo "Audit complete."
