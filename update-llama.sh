#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Cloning llama-cpp-turboquant ==="
git clone https://github.com/TheTom/llama-cpp-turboquant "$ROOT/llama-cpp-turboquant"

echo "=== Removing .git ==="
rm -rf "$ROOT/llama-cpp-turboquant/.git"

echo "=== Emptying uninference/ ==="
rm -rf "$ROOT/uninference/" && mkdir "$ROOT/uninference/"

echo "=== Moving contents to uninference/ ==="
rsync -a "$ROOT/llama-cpp-turboquant/" "$ROOT/uninference/"

echo "=== Removing source folder ==="
rm -rf "$ROOT/llama-cpp-turboquant"

echo "=== Removing unwanted files from uninference/ ==="
(cd "$ROOT/uninference" && rm -rf \
    .pre-commit-config.yaml \
    .gemini \
    .github \
    .docs \
    .gitignore \
    .git \
    .gitattributes \
    .gitmodules \
    AGENTS.md \
    CLAUDE.md \
    AUTHORS \
    bench* \
    CONTRIBUTING.md \
    LICENSE \
    README.md \
    SECURITY.md)

echo "=== Copying docker-files/ ==="
rsync -a "$ROOT/docker-files/" "$ROOT/uninference/"

echo "=== Done ==="
