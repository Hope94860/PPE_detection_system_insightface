[Setup]
AppName=SiteSecureVision
AppVersion=1.0.0
AppPublisher=Invisible Fiction
PrivilegesRequired=admin
DefaultDirName={autopf}\SiteSecureVision
DefaultGroupName=SiteSecureVision
OutputDir=.\Installer
OutputBaseFilename=SiteSecureVision_Installer
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "setup_prerequisites.ps1"; DestDir: "{tmp}"; Flags: deleteafterinstall
Source: "dist\SiteSecureVision\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SiteSecureVision"; Filename: "{app}\SiteSecureVision.exe"
Name: "{autodesktop}\SiteSecureVision"; Filename: "{app}\SiteSecureVision.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -WindowStyle Hidden -File ""{tmp}\setup_prerequisites.ps1"""; StatusMsg: "Checking and installing MongoDB dependencies. This may take a few minutes..."; Flags: waituntilterminated runhidden
Filename: "{app}\SiteSecureVision.exe"; Parameters: "--init-db"; StatusMsg: "Initializing Database and Collections..."; Flags: waituntilterminated runhidden
Filename: "{app}\SiteSecureVision.exe"; Description: "Launch SiteSecureVision"; Flags: nowait postinstall skipifsilent
