# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for FaceGate desktop app."""

from pathlib import Path

block_cipher = None
ROOT = Path(SPECPATH).resolve().parent
EYE = ROOT / "eye-tracking"
ENTRY = EYE / "app" / "run_desktop.py"

a = Analysis(
    [str(ENTRY)],
    pathex=[str(EYE), str(ROOT)],
    binaries=[],
    datas=[
        (str(EYE / "app" / "desktop" / "assets"), "app/desktop/assets"),
    ],
    hiddenimports=[
        "insightface",
        "onnxruntime",
        "PIL",
        "cv2",
        "numpy",
        "app.desktop",
        "app.desktop.app",
        "app.desktop.services",
        "common.face_recognition",
        "common.camera_io",
        "common.profiles",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FaceGate",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FaceGate",
)
