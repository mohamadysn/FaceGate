#!/usr/bin/env python3
"""
Install a Desktop / Start-menu shortcut for FaceGate (Linux / Windows / macOS).

    python app/install_desktop_shortcut.py
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
EYE_DIR = APP_DIR.parent
REPO_ROOT = EYE_DIR.parent
ICON = APP_DIR / "desktop" / "assets" / "face_recog.png"
LAUNCH_PY = APP_DIR / "launch.py"
APP_NAME = "FaceGate"


def _python() -> Path:
    sys.path.insert(0, str(EYE_DIR))
    from app.desktop.platform_util import find_python

    return find_python(REPO_ROOT, EYE_DIR)


def _desktop() -> Path:
    sys.path.insert(0, str(EYE_DIR))
    from app.desktop.platform_util import desktop_directory

    return desktop_directory()


def install_linux() -> None:
    desktop = _desktop()
    apps = Path.home() / ".local" / "share" / "applications"
    apps.mkdir(parents=True, exist_ok=True)
    python = _python()
    exec_line = f'"{python}" "{LAUNCH_PY}"'
    body = f"""[Desktop Entry]
Version=1.0
Type=Application
Name={APP_NAME}
Comment=FaceGate — enroll and recognize faces from webcam or photos
Exec={exec_line}
Icon={ICON}
Path={EYE_DIR}
Terminal=false
StartupNotify=true
Categories=Utility;
StartupWMClass=Tk
"""
    entry = apps / "facegate.desktop"
    entry.write_text(body, encoding="utf-8")
    entry.chmod(entry.stat().st_mode | stat.S_IEXEC)
    # Remove old shortcut names if present
    for old in (
        apps / "face-recognition.desktop",
        desktop / "Face Recognition.desktop",
        desktop / "face-recognition.desktop",
    ):
        if old.is_file():
            old.unlink()
    if desktop.is_dir():
        dest = desktop / f"{APP_NAME}.desktop"
        shutil.copy2(entry, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC)
        try:
            subprocess.run(
                ["gio", "set", str(dest), "metadata::trusted", "true"],
                check=False,
                capture_output=True,
            )
        except FileNotFoundError:
            pass
        print(f"Desktop shortcut: {dest}")
        print("If it opens as text: right-click → Allow Launching")
    subprocess.run(["update-desktop-database", str(apps)], check=False, capture_output=True)
    print(f"App menu entry: {entry}")


def install_windows() -> None:
    desktop = _desktop()
    desktop.mkdir(parents=True, exist_ok=True)
    python = _python()
    bat = desktop / f"{APP_NAME}.bat"
    bat.write_text(
        "\r\n".join(
            [
                "@echo off",
                f'cd /d "{EYE_DIR}"',
                f'"{python}" "{LAUNCH_PY}"',
                "if errorlevel 1 pause",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Desktop launcher: {bat}")
    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if start_menu.is_dir():
        lnk = start_menu / f"{APP_NAME}.lnk"
        ps = f"""
$W = New-Object -ComObject WScript.Shell
$S = $W.CreateShortcut('{lnk}')
$S.TargetPath = '{python}'
$S.Arguments = '"{LAUNCH_PY}"'
$S.WorkingDirectory = '{EYE_DIR}'
$S.IconLocation = '{ICON}'
$S.Save()
"""
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            check=False,
            capture_output=True,
        )
        print(f"Start Menu shortcut: {lnk}")


def install_macos() -> None:
    desktop = _desktop()
    desktop.mkdir(parents=True, exist_ok=True)
    python = _python()
    command = desktop / f"{APP_NAME}.command"
    command.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                f'cd "{EYE_DIR}"',
                f'exec "{python}" "{LAUNCH_PY}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    command.chmod(command.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Desktop launcher: {command}")
    print("First run: right-click → Open (Gatekeeper may block unknown scripts).")


def main() -> None:
    if not LAUNCH_PY.is_file():
        raise SystemExit(f"Missing launcher: {LAUNCH_PY}")
    if sys.platform.startswith("linux"):
        install_linux()
    elif sys.platform == "win32":
        install_windows()
    elif sys.platform == "darwin":
        install_macos()
    else:
        raise SystemExit(f"Unsupported platform: {sys.platform}")
    print("Done.")


if __name__ == "__main__":
    main()
