#!/usr/bin/env bash
set -euo pipefail

git add -A
git commit -m "${*:-fix and improvement}"
git push origin main
