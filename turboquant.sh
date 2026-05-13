#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if docker ps -a --format '{{.Names}}' | grep -q '^turboquant$'; then
  echo "Rimuovo container esistente..."
  docker rm -f turboquant
fi

docker run -d \
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
  uhao-llama-cpp:0.4.150 \
  -c 'exec llama-server \
    --port 8080 \
    --host 0.0.0.0 \
    --models-dir /models \
    --models-max 4 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    --n-cpu-moe 35 \
    --no-mmap \
    --mlock \
    --jinja \
    --flash-attn on \
    --ubatch-size 256 \
    --batch-size 8192 \
    -c 131072 \
    -ngl 999 \
    --threads 8 \
    --threads-batch 24 \
    --no-context-shift \
    --swa-full \
    -nkvo \
    --no-warmup
    --webui-mcp-proxy'
