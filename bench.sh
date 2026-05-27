#!/bin/bash
set -euo pipefail

export MODEL_NAME="Qwen3.6-35B-A3B-UD-IQ2_XXS"
export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BENCHY_DIR="$SCRIPT_DIR/benchy"
export TRANSFORMERS_VERBOSITY=debug

RESULT_FILE="$SCRIPT_DIR/bench_results/${MODEL_NAME}_bench.json"

cd "$BENCHY_DIR"
.venv/bin/llama-benchy \
  --base-url http://localhost:8080/v1 \
  --model "$MODEL_NAME" \
  --tokenizer Qwen/Qwen3.6-35B-A3B \
  --pp 2048 \
  --tg 2048 \
  --depth 4096 \
  --runs 1 \
  --post-run-cmd "curl -s -X POST http://localhost:8080/v1/cache/clear -H 'Content-Type: application/json' -d '{}' > /dev/null" \
  --latency-mode generation \
  --no-warmup \
  --enable-prefix-caching \
  --concurrency 4 \
  --save-result "$RESULT_FILE" \
  --format json \
  --exit-on-first-fail \
  --no-results-on-fail

SUMMARY="$(jq -r '
  "TG (generation):         \([.benchmarks[].tg_throughput.mean] | add / length | . * 100 | round / 100) tok/s",
  "PP (prefill):            \([.benchmarks[].pp_throughput.mean] | add / length | . * 100 | round / 100) tok/s",
  "Latency:                 \(.latency_ms * 100 | round / 100) ms/tok",
  "TTFR:                    \([.benchmarks[].ttfr.mean] | add / length | . * 100 | round / 100) ms"
' "$RESULT_FILE")"

# Extract launch params from turboquant.sh
MOE=$(sed -n 's/.*--n-cpu-moe \([0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")
CTX=$(sed -n 's/.* -c \([0-9][0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")
UBS=$(sed -n 's/.*--ubatch-size \([0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")
BS=$(sed -n 's/.*--batch-size \([0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")
TH=$(sed -n 's/.*--threads \([0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")
THB=$(sed -n 's/.*--threads-batch \([0-9]*\).*/\1/p' "$SCRIPT_DIR/turboquant.sh")

MODEL_LABEL="$(basename "$RESULT_FILE" _bench.json)"

OUTPUT="=== $MODEL_LABEL ===
$SUMMARY
Params: --n-cpu-moe $MOE -c $CTX --ubatch-size $UBS --batch-size $BS --threads $TH --threads-batch $THB
"

echo ""
echo "$OUTPUT"

# Append to results.txt
mkdir -p "$SCRIPT_DIR/bench_results"
{
  echo ""
  echo "$OUTPUT"
} >> "$SCRIPT_DIR/bench_results/results.txt"
