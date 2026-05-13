#!/usr/bin/env bash
# build.sh – build container uninference con auto-incremento patch
# Uso:
#   ./build.sh                          # tag automatico: uninference:X.Y.Z+1
#   CUDA_ARCH=86 ./build.sh             # arch diversa (86 = RTX 30xx)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-uninference}"
CUDA_ARCH="${CUDA_ARCH:-89}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Trova l'ultima minor version ─────────────────────────────────────────────
LATEST_TAG=$(docker images "${IMAGE_NAME}" --format '{{.Tag}}' \
  | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' \
  | sort -t. -k1,1n -k2,2n -k3,3n \
  | tail -1 || true)

if [ -z "$LATEST_TAG" ]; then
  NEW_TAG="${IMAGE_NAME}:0.1.0"
else
  MAJOR="${LATEST_TAG%%.*}"
  REST="${LATEST_TAG#*.}"
  MINOR="${REST%%.*}"
  PATCH="${REST#*.}"
  PATCH="${PATCH%%.*}"
  NEW_PATCH=$((PATCH + 1))
  NEW_TAG="${IMAGE_NAME}:${MAJOR}.${MINOR}.${NEW_PATCH}"
fi

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
echo "  Immagine  : ${NEW_TAG}"
echo "  CUDA arch : sm_${CUDA_ARCH}"
echo "  Contesto  : ${SCRIPT_DIR}"
echo "========================================================"
echo ""

docker build \
    --progress=plain \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --tag "${NEW_TAG}" \
    "${SCRIPT_DIR}"

echo ""
echo "========================================================"
echo ""
echo "  Build completata: ${NEW_TAG}"
echo ""
echo "========================================================"
