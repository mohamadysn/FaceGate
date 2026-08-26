#!/usr/bin/env bash
# Thin wrapper kept for older docs / shortcuts — launches the desktop app.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "$HERE/FaceGate" "$@"
