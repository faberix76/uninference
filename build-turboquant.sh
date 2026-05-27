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
echo "  Status Docker before prune"
echo ""
echo "========================================================"
echo ""
docker system df
echo "========================================================"
echo ""
echo "  Clean Docker build cache e layers"
echo ""
echo "========================================================"
echo ""
docker image prune -f
echo ""
echo "========================================================"
echo ""
echo "  Status Docker after prune"
echo ""
echo "========================================================"
echo ""
docker system df

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
