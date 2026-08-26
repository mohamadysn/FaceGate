@echo off
REM Build FaceGate with PyInstaller (Windows).
set ROOT=%~dp0..
cd /d "%ROOT%"

python -m pip install -q -e ".[build]"
if exist build rmdir /s /q build
if exist dist\FaceGate rmdir /s /q dist\FaceGate
pyinstaller --noconfirm packaging\facegate.spec
echo Built: %ROOT%\dist\FaceGate\
