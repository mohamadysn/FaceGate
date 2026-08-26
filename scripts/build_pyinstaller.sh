#!/usr/bin/env bash
# Build FaceGate with PyInstaller (Linux / macOS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m pip install -q -e ".[build]"
rm -rf build dist/FaceGate
pyinstaller --noconfirm packaging/facegate.spec
echo "Built: $ROOT/dist/FaceGate/"
