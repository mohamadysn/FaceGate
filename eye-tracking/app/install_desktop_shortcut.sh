#!/usr/bin/env bash
# Install Desktop / app-menu shortcut (delegates to cross-platform Python installer).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EYE="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$EYE/.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$EYE/.venv/bin/python" ]]; then
  PYTHON="$EYE/.venv/bin/python"
else
  PYTHON="$(command -v python3 || command -v python)"
fi
exec "$PYTHON" "$HERE/install_desktop_shortcut.py"
