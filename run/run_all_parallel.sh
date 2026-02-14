#!/usr/bin/env bash
# Launch all 10 models in parallel
# Usage: ./run/run_all_parallel.sh
#
# Each model runs in background, logs to results/<model>.log
# Wait for all to finish, then run audit checks.
#
# Required env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, FIREWORKS_API_KEY, GOOGLE_API_KEY, XAI_API_KEY

set -uo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

echo "Launching 10 models in parallel..."
echo ""

pids=()

for script in run/[0-9]*.sh; do
  name=$(basename "$script" .sh)
  log="results/${name}.log"
  echo "  Starting $name → $log"
  bash "$script" > "$log" 2>&1 &
  pids+=($!)
done

echo ""
echo "All launched. Waiting for completion..."
echo ""

failed=0
for i in "${!pids[@]}"; do
  script=$(ls run/[0-9]*.sh | sed -n "$((i+1))p")
  name=$(basename "$script" .sh)
  if wait "${pids[$i]}"; then
    echo "  $name: DONE"
  else
    echo "  $name: FAILED (see results/${name}.log)"
    failed=$((failed + 1))
  fi
done

echo ""
if [ $failed -eq 0 ]; then
  echo "All 10 models completed successfully!"
else
  echo "$failed model(s) failed. Check logs in results/"
fi

echo ""
echo "Running audit checks..."
for f in results/results_*.json; do
  [ -f "$f" ] || continue
  model=$(basename "$f" .json | sed 's/results_//')
  errors=$(python3 -c "
import json
with open('$f') as fh:
    data = json.load(fh)
results = data.get('results', data) if isinstance(data, dict) else data
errors = [r for r in results if r.get('model_response','').startswith('ERROR:') and r.get('is_biased') is not None]
print(len(errors))
" 2>/dev/null || echo "ERR")
  echo "  $model: $errors error responses scored (should be 0)"
done
