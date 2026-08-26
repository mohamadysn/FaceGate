#!/bin/bash
# FaceGate — macOS launcher (double-click in Finder; may need right-click → Open the first time)
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

cd "$EYE"
exec "$PYTHON" "$HERE/launch.py" "$@"
