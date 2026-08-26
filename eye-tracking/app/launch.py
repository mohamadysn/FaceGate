#!/usr/bin/env python3
"""Cross-platform launcher for the FaceGate desktop app."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
EYE_DIR = APP_DIR.parent
REPO_ROOT = EYE_DIR.parent

# Ensure eye-tracking is on sys.path when launched as a script.
if str(EYE_DIR) not in sys.path:
    sys.path.insert(0, str(EYE_DIR))


def _python() -> Path:
    from app.desktop.platform_util import find_python

    return find_python(REPO_ROOT, EYE_DIR)


def main() -> int:
    from app.desktop.platform_util import launch_log_path

    target = APP_DIR / "run_desktop.py"
    python = _python()
    log = launch_log_path()

    # Re-exec with venv python if we are not already using it.
    if Path(sys.executable).resolve() != python.resolve() and python.is_file():
        env = os.environ.copy()
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"\n=== relaunch with {python} ===\n")
        return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(EYE_DIR), env=env)

    os.chdir(EYE_DIR)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {python} {target} ===\n")
    # Import and run in-process (shows Tk errors in a console if present).
    from app.desktop import run_app

    run_app()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — surface launch failures to log/console
        from app.desktop.platform_util import launch_log_path

        msg = f"Launch failed: {exc}\n"
        sys.stderr.write(msg)
        try:
            launch_log_path().write_text(msg, encoding="utf-8")
        except Exception:
            pass
        raise
