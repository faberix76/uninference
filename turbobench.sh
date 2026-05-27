#!/bin/bash
# ==============================================================
# turbobench.sh – llama-bench inside Docker container, sweep
#
# Arrays in testa → modifica liberamente per escludere valori.
# Ogni combinazione gira in un container Docker separato.
# Il file risultati viene salvato in bench_results/.
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/models/Qwen3.6-35B-A3B-UD-IQ3_S.gguf}"
IMAGE="${IMAGE:-llama-cpp-turboquant:latest}"
RESULTS_FILE="${SCRIPT_DIR}/bench_results/turbobench_results.txt"

mkdir -p "$(dirname "$RESULTS_FILE")"
: > "$RESULTS_FILE"

# ═══════════════════════════════════════════════════════════════
# PARAMETER ARRAYS
# ═══════════════════════════════════════════════════════════════
# Riduci gli array per esecuzioni più rapide (es. un valore solo).
# Lascia commentando con # per escludere.

# --n-cpu-moe: CPU core dedicati alla moltiplicazione MoE
NCPU_MOE=(26 14)

# -t / --threads: thread CPU per inferenza
THREADS=(16 8)

# --threads-batch: thread CPU per batch processing
THREADS_BATCH=(16 8)

# --ubatch-size: micro-batch per step GPU
UBATCH_SIZE=(256 512)

# -b / --batch-size: batch totale per prompt processing
BATCH_SIZE=(4096 8192)

# -p: numero token di prompt
PROMPT_SIZE=(512 2048)

# -n: numero token da generare
GEN_SIZE=(256)

# ═══════════════════════════════════════════════════════════════
# DO NOT EDIT BELOW THIS LINE
# ═══════════════════════════════════════════════════════════════

TOTAL=$(( ${#NCPU_MOE[@]} * ${#THREADS[@]} * ${#THREADS_BATCH[@]} * ${#UBATCH_SIZE[@]} * ${#BATCH_SIZE[@]} * ${#PROMPT_SIZE[@]} * ${#GEN_SIZE[@]} ))
COUNT=0

cat > /tmp/turbobench_header.$$ <<EOF
╔══════════════════════════════════════════════════════════════╗
║  turbobench.sh — $TOTAL combinazioni
║  Model:  $MODEL
║  Image:  $IMAGE
╚══════════════════════════════════════════════════════════════╝

EOF
cat /tmp/turbobench_header.$$ | tee "$RESULTS_FILE"
rm /tmp/turbobench_header.$$

for ncmoe in "${NCPU_MOE[@]}"; do
  for th in "${THREADS[@]}"; do
    for thb in "${THREADS_BATCH[@]}"; do
      for ub in "${UBATCH_SIZE[@]}"; do
        for bs in "${BATCH_SIZE[@]}"; do
          for pp in "${PROMPT_SIZE[@]}"; do
            for gn in "${GEN_SIZE[@]}"; do
              COUNT=$((COUNT + 1))

              SEP="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              LABEL="[$COUNT/$TOTAL] ncmoe=$ncmoe  th=$th  thb=$thb  ub=$ub  bs=$bs  pp=$pp  gn=$gn"

              echo "$SEP" | tee -a "$RESULTS_FILE"
              echo "  $LABEL" | tee -a "$RESULTS_FILE"
              echo "$SEP" | tee -a "$RESULTS_FILE"

              docker run --rm \
                -v "${SCRIPT_DIR}/models:/models" \
                --gpus all \
                --entrypoint /bin/sh \
                "$IMAGE" \
                -c "exec llama-bench \
                  -m '${MODEL}' \
                  --n-cpu-moe ${ncmoe} \
                  -t ${th} \
                  --threads-batch ${thb} \
                  --ubatch-size ${ub} \
                  -b ${bs} \
                  -p ${pp} \
                  -n ${gn} \
                  -fa 1" \
                | tee -a "$RESULTS_FILE"

              echo "  ✓ [$COUNT/$TOTAL] done" | tee -a "$RESULTS_FILE"
              echo "" | tee -a "$RESULTS_FILE"
            done
          done
        done
      done
    done
  done
done

# ─── Final summary ────────────────────────────────────────────
{
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║              TURBOBENCH COMPLETE                            ║"
  echo "╠══════════════════════════════════════════════════════════════╣"
  printf "║  Total:  %-5d combos                                        ║\n" "$TOTAL"
  printf "║  Model:  %-45s║\n" "$MODEL"
  printf "║  File:   %-45s║\n" "$RESULTS_FILE"
  echo "╚══════════════════════════════════════════════════════════════╝"
} | tee -a "$RESULTS_FILE"
