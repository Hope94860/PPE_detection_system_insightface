@echo off
echo ==============================================
echo Building SiteSecureVision Installer
echo ==============================================

echo [1/3] Cleaning previous build...
rmdir /s /q build
rmdir /s /q dist
rmdir /s /q Installer

echo [2/3] Building executable using PyInstaller...
.\insight_env\Scripts\python.exe -m PyInstaller -y sitesecurevision.spec

if %ERRORLEVEL% neq 0 (
    echo PyInstaller build failed!
    exit /b %ERRORLEVEL%
)

echo [3/3] Compiling Inno Setup Script...
:: Search for Inno Setup compiler in common locations
set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
)

if not exist "%ISCC%" (
    echo Inno Setup 6 compiler ^(ISCC.exe^) not found.
    echo Please ensure Inno Setup 6 is installed.
    exit /b 1
)

"%ISCC%" installer.iss

if %ERRORLEVEL% neq 0 (
    echo Inno Setup compilation failed!
    exit /b %ERRORLEVEL%
)

echo.
echo ==============================================
echo Build Completed Successfully!
echo Installer can be found in the "Installer" folder.
echo ==============================================
pause
