#!/usr/bin/env bash
# build-turboquant.sh – build container llama-cpp-turboquant
# Uso:
#   ./build-turboquant.sh               # tag: llama-cpp-turboquant:latest
#   CUDA_ARCH=86 ./build-turboquant.sh  # arch diversa (86 = RTX 30xx)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CUDA_ARCH="${CUDA_ARCH:-89}"
IMAGE_NAME="${IMAGE_NAME:-llama-cpp-turboquant}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="${ROOT}/llama-cpp-turboquant"
DOCKERFILE="${ROOT}/docker-files/Dockerfile"
TAG="${IMAGE_NAME}:latest"

echo "========================================================"
echo ""
echo "  Aggiorno il repo ufficiale llama-cpp-turboquant..."
echo ""
echo "========================================================"
echo ""
git -C "${SCRIPT_DIR}" pull origin feature/turboquant-kv-cache
echo ""

echo "========================================================"
echo "  Immagine  : ${TAG}"
echo "  CUDA arch : sm_${CUDA_ARCH}"
echo "  Contesto  : ${SCRIPT_DIR}"
echo "========================================================"
echo ""

docker build \
    --progress=plain \
    --file "${DOCKERFILE}" \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --tag "${TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "========================================================"
echo ""
echo "  Build completata: ${TAG}"
echo ""
echo "========================================================"
