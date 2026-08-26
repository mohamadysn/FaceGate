; Inno Setup script — wrap PyInstaller output into a Windows installer.
; 1. Run scripts\build_pyinstaller.bat
; 2. Open this file in Inno Setup Compiler and Build
; For a true .msi, use WiX Toolset against the same dist\FaceGate folder.

#define MyAppName "FaceGate"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Mohamad Yassine"
#define MyAppURL "https://github.com/mohamadysn/FaceGate"
#define MyAppExeName "FaceGate.exe"

[Setup]
AppId={{A3F8C2D1-9E4B-4F07-9C1A-FaceGate01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=..\dist
OutputBaseFilename=FaceGate-Setup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\FaceGate\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
