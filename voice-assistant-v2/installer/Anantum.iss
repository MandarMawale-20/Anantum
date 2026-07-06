; Inno Setup script for packaging the Tauri shell and Python backend runtime.
; Use when a customized installation flow is required.

#define AppName "Anantum"
#define AppVersion "0.1.0"
#define Publisher "Anantum"
#define ExeName "anantum_siri_widget.exe"

[Setup]
AppId={{A4CD13A0-7D35-4B42-8D2A-6CEAD6C25A7B}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=output
OutputBaseFilename=Anantum-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; Tauri release binaries.
Source: "..\frontend\src-tauri\target\release\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\frontend\src-tauri\target\release\*.dll"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Packaged Python backend runtime.
Source: "..\frontend\dist\backend\*"; DestDir: "{app}\backend"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#ExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"

[Run]
Filename: "{app}\{#ExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
