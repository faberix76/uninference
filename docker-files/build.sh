#!/usr/bin/env bash
# build.sh – build container uninference, auto-incremento patch da .version
# Uso:
#   ./build.sh                          # tag: uninference:X.Y.Z (da .version)
#   CUDA_ARCH=86 ./build.sh             # arch diversa (86 = RTX 30xx)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-uninference}"
CUDA_ARCH="${CUDA_ARCH:-89}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION_FILE="${SCRIPT_DIR}/.version"
DOCKER_VERSION_FILE="$(cd "${SCRIPT_DIR}/../docker-files" && pwd)/.version"

# ── Legge versione da .version (source of truth) ────────────────────────────
if [ ! -f "$VERSION_FILE" ]; then
  echo "ERROR: .version file not found at ${VERSION_FILE}"
  exit 1
fi

CUR_VERSION="$(cat "${VERSION_FILE}" | tr -d '[:space:]')"
if [ -z "$CUR_VERSION" ]; then
  echo "ERROR: .version file is empty"
  exit 1
fi

# ── Incrementa patch ────────────────────────────────────────────────────────
MAJOR="${CUR_VERSION%%.*}"
REST="${CUR_VERSION#*.}"
MINOR="${REST%%.*}"
PATCH="${REST#*.}"
PATCH="${PATCH%%.*}"
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="${MAJOR}.${MINOR}.${NEW_PATCH}"
NEW_TAG="${IMAGE_NAME}:${NEW_VERSION}"

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

# ── Aggiorna .version (source of truth) ──────────────────────────────────────
echo "${NEW_VERSION}" > "${VERSION_FILE}"
echo "${NEW_VERSION}" > "${DOCKER_VERSION_FILE}"
echo "  ${VERSION_FILE} → ${NEW_VERSION}"
