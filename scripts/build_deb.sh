#!/usr/bin/env bash
# Wrap dist/FaceGate into a .deb (requires fpm + prior PyInstaller build).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${FACEGATE_VERSION:-0.1.0}"
STAGE="$ROOT/dist/deb-stage"
APP_SRC="$ROOT/dist/FaceGate"

if [[ ! -d "$APP_SRC" ]]; then
  echo "Missing $APP_SRC — run scripts/build_pyinstaller.sh first." >&2
  exit 1
fi
if ! command -v fpm >/dev/null 2>&1; then
  echo "fpm not found. Install: gem install fpm  (or use your distro package)" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/opt/facegate" "$STAGE/usr/local/bin"
cp -a "$APP_SRC/." "$STAGE/opt/facegate/"
cat > "$STAGE/usr/local/bin/facegate" <<'EOF'
#!/bin/sh
exec /opt/facegate/FaceGate "$@"
EOF
chmod +x "$STAGE/usr/local/bin/facegate" "$STAGE/opt/facegate/FaceGate"

fpm -s dir -t deb -n facegate -v "$VERSION" \
  --description "FaceGate desktop face recognition" \
  --url "https://github.com/mohamadysn/FaceGate" \
  --license MIT \
  -C "$STAGE" \
  opt/facegate usr/local/bin

mkdir -p "$ROOT/dist"
mv -f facegate_"${VERSION}"*.deb "$ROOT/dist/" 2>/dev/null || mv -f ./*.deb "$ROOT/dist/"
echo "Debian package under $ROOT/dist/"
