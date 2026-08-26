"""Unit tests for desktop platform helpers (no GUI required)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.desktop.platform_util import (
    desktop_directory,
    find_python,
    is_linux,
    is_macos,
    is_windows,
    launch_log_path,
    ui_font,
)


class PlatformUtilTests(unittest.TestCase):
    def test_os_flags_exclusive(self) -> None:
        flags = [is_windows(), is_macos(), is_linux()]
        self.assertEqual(sum(1 for f in flags if f), 1)

    def test_ui_font_shape(self) -> None:
        regular = ui_font(12)
        bold = ui_font(12, bold=True)
        self.assertEqual(len(regular), 2)
        self.assertEqual(bold[-1], "bold")
        with mock.patch("app.desktop.platform_util.sys.platform", "win32"):
            self.assertEqual(ui_font(10)[0], "Segoe UI")
        with mock.patch("app.desktop.platform_util.sys.platform", "darwin"):
            self.assertEqual(ui_font(10)[0], "Helvetica Neue")
        with mock.patch("app.desktop.platform_util.sys.platform", "linux"):
            self.assertEqual(ui_font(10)[0], "DejaVu Sans")

    def test_find_python_falls_back_to_sys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eye = root / "eye-tracking"
            eye.mkdir()
            py = find_python(root, eye)
            self.assertEqual(py.resolve(), Path(sys.executable).resolve())

    def test_find_python_prefers_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eye = root / "eye-tracking"
            eye.mkdir()
            if sys.platform == "win32":
                venv_py = root / ".venv" / "Scripts" / "python.exe"
            else:
                venv_py = root / ".venv" / "bin" / "python"
            venv_py.parent.mkdir(parents=True)
            venv_py.write_text("#!/bin/sh\n", encoding="utf-8")
            venv_py.chmod(0o755)
            found = find_python(root, eye)
            self.assertEqual(found, venv_py)

    def test_launch_log_path_name(self) -> None:
        self.assertEqual(launch_log_path().name, "facegate-launch.log")

    def test_desktop_directory_returns_path(self) -> None:
        path = desktop_directory()
        self.assertIsInstance(path, Path)


if __name__ == "__main__":
    unittest.main()
