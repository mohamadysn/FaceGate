"""Cross-platform helpers for the desktop app (Linux / Windows / macOS)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def ui_font(size: int, bold: bool = False) -> Tuple:
    """Return a Tk font tuple that exists on the current OS."""
    if is_macos():
        family = "Helvetica Neue"
    elif is_windows():
        family = "Segoe UI"
    else:
        family = "DejaVu Sans"
    return (family, size, "bold") if bold else (family, size)


def project_paths(from_file: Path | None = None) -> Tuple[Path, Path, Path]:
    """
    Resolve ``(app_dir, eye_tracking_dir, repo_root)``.

    ``from_file`` should be a file under ``eye-tracking/app/`` (default: this module's callers).
    """
    app_dir = Path(from_file).resolve().parent if from_file else Path(__file__).resolve().parent
    # If called from desktop/*.py, parents[1] is app/; callers pass run_desktop path.
    eye = app_dir.parent if app_dir.name == "app" else app_dir.parents[1]
    if (eye / "app").is_dir() and eye.name != "eye-tracking" and (app_dir / "run_desktop.py").exists():
        eye = app_dir.parent
    root = eye.parent
    return app_dir, eye, root


def find_python(repo_root: Path, eye_dir: Path) -> Path:
    """Prefer the project virtualenv interpreter when present."""
    candidates = []
    if is_windows():
        candidates.extend(
            [
                repo_root / ".venv" / "Scripts" / "python.exe",
                eye_dir / ".venv" / "Scripts" / "python.exe",
            ]
        )
    else:
        candidates.extend(
            [
                repo_root / ".venv" / "bin" / "python",
                eye_dir / ".venv" / "bin" / "python",
            ]
        )
    for path in candidates:
        if path.is_file():
            return path
    return Path(sys.executable)


def desktop_directory() -> Path:
    """User Desktop folder (supports French Ubuntu ``Bureau``)."""
    home = Path.home()
    env = os.environ.get("XDG_DESKTOP_DIR") or os.environ.get("USERPROFILE")
    if env:
        # XDG may be like $HOME/Desktop
        raw = Path(os.path.expandvars(env)).expanduser()
        if raw.is_dir() and raw.name.lower() in {"desktop", "bureau"}:
            return raw
        desktop = raw / "Desktop"
        if desktop.is_dir():
            return desktop
    for name in ("Desktop", "Bureau"):
        candidate = home / name
        if candidate.is_dir():
            return candidate
    return home / "Desktop"


def launch_log_path() -> Path:
    return Path(tempfile.gettempdir()) / "facegate-launch.log"


def user_data_dir() -> Path:
    """Per-user FaceGate data root (gallery, future settings)."""
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        path = base / "FaceGate"
    elif is_macos():
        path = Path.home() / "Library" / "Application Support" / "FaceGate"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        path = base / "FaceGate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_gallery_dir(eye_tracking_root: Path) -> Path:
    return eye_tracking_root / "face-recognition" / "gallery"


def _is_installed_package(eye_tracking_root: Path) -> bool:
    root = str(eye_tracking_root).lower()
    return "site-packages" in root or "dist-packages" in root


def _migrate_gallery_if_needed(src: Path, dst: Path) -> None:
    """Copy repo gallery into user folder once if user gallery is still empty."""
    if (dst / "gallery.json").is_file():
        return
    if not (src / "gallery.json").is_file():
        return
    import shutil

    for name in ("gallery.json", "embeddings.npy"):
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)


def resolve_gallery_dir(eye_tracking_root: Path) -> Path:
    """
    Return the gallery directory for the desktop app.

    - ``FACEGATE_GALLERY`` env var overrides everything.
    - pip / PyInstaller install → ``%LOCALAPPDATA%\\FaceGate\\gallery`` (Windows),
      ``~/.local/share/FaceGate/gallery`` (Linux), etc.
    - Git clone / editable dev → ``eye-tracking/face-recognition/gallery``.
    """
    override = os.environ.get("FACEGATE_GALLERY")
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    repo_gallery = repo_gallery_dir(eye_tracking_root)
    user_gallery = user_data_dir() / "gallery"
    user_gallery.mkdir(parents=True, exist_ok=True)

    if _is_installed_package(eye_tracking_root):
        _migrate_gallery_if_needed(repo_gallery, user_gallery)
        return user_gallery

    if repo_gallery.is_dir():
        return repo_gallery
    return user_gallery
