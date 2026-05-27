#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      MODEL="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ "$(basename "$MODEL")" == *MTP* ]]; then
  MTP_ARGS="--spec-type draft-mtp --spec-draft-n-max 1 --spec-draft-p-min 1"
else
  MTP_ARGS=""
fi

if docker ps -a --format '{{.Names}}' | grep -q '^turboquant$'; then
  echo "Rimuovo container esistente..."
  docker rm -f turboquant
fi

docker run  \
  --name turboquant \
  --restart unless-stopped \
  -v "${SCRIPT_DIR}/models:/models" \
  -e GGML_OP_OFFLOAD_MIN_BATCH=999999 \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --gpus all \
  -p 8080:8080 \
  --log-driver json-file \
  --log-opt max-size=50m \
  --log-opt max-file=3 \
  --entrypoint /bin/sh \
    llama-cpp-turboquant:latest \
  -c "exec taskset -c 0-15 llama-server \
    --port 8080 \
    --host 0.0.0.0 \
    -m ${MODEL} \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    --no-mmap \
    --no-warmup \
    --mlock \
    --n-cpu-moe 26 \
    --flash-attn on \
    --ubatch-size 512 \
    --verbose \
    --batch-size 8192 \
    -c 131000 \
    -ngl 999 \
    --threads 16 \
    --threads-batch 16 \
    --temp 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --presence-penalty 0.0 \
    --min-p 0.00 \
    ${MTP_ARGS} \
    --chat-template-kwargs \"{\\\"enable_thinking\\\":true}\" \
    --webui-mcp-proxy"
