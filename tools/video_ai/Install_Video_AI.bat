@echo off
setlocal EnableExtensions

set "CRV_ROOT=%LOCALAPPDATA%\CRV"
set "VENV_DIR=%CRV_ROOT%\venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "CRV_EXE=%VENV_DIR%\Scripts\crv.exe"
set "INSTALL_LOG=%CRV_ROOT%\install.log"

if /i "%~1"=="--worker" goto :worker

if not exist "%CRV_ROOT%" mkdir "%CRV_ROOT%"
if errorlevel 1 (
    echo ERROR: Could not create "%CRV_ROOT%".
    exit /b 1
)

call "%~f0" --worker > "%INSTALL_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"
type "%INSTALL_LOG%"
if not "%EXIT_CODE%"=="0" echo Installation failed. See "%INSTALL_LOG%".
exit /b %EXIT_CODE%

:worker
echo Video AI installation started at %DATE% %TIME%
echo Installation log: "%INSTALL_LOG%"
echo.

where py >nul 2>&1
if errorlevel 1 (
    echo Python launcher not found.
    call :install_python
    if errorlevel 1 exit /b 1
)

py -3.12 --version
if errorlevel 1 (
    echo Python 3.12 not found.
    call :install_python
    if errorlevel 1 exit /b 1
)

py -3.12 --version
if errorlevel 1 (
    echo ERROR: Python 3.12 is still unavailable after installation.
    exit /b 1
)

ffmpeg -version
set "FFMPEG_MISSING=%ERRORLEVEL%"
ffprobe -version
set "FFPROBE_MISSING=%ERRORLEVEL%"
if not "%FFMPEG_MISSING%"=="0" goto :install_ffmpeg
if not "%FFPROBE_MISSING%"=="0" goto :install_ffmpeg
goto :ffmpeg_ready

:install_ffmpeg
where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: FFmpeg or FFprobe is unavailable and winget was not found.
    exit /b 1
)
echo Installing FFmpeg with winget...
winget install -e --id Gyan.FFmpeg
if errorlevel 1 (
    echo ERROR: winget could not install Gyan.FFmpeg.
    exit /b 1
)

:ffmpeg_ready
ffmpeg -version
if errorlevel 1 (
    echo ERROR: ffmpeg is still unavailable. Open a new terminal after installation and rerun this installer.
    exit /b 1
)
ffprobe -version
if errorlevel 1 (
    echo ERROR: ffprobe is still unavailable. Open a new terminal after installation and rerun this installer.
    exit /b 1
)

if not exist "%CRV_ROOT%" mkdir "%CRV_ROOT%"
if errorlevel 1 (
    echo ERROR: Could not create "%CRV_ROOT%".
    exit /b 1
)

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
    if errorlevel 1 (
        echo ERROR: Existing environment at "%VENV_DIR%" is not Python 3.12.
        echo Delete "%VENV_DIR%" and rerun this installer.
        exit /b 1
    )
) else (
    if exist "%VENV_DIR%" (
        echo ERROR: "%VENV_DIR%" exists but is not a usable Python environment.
        echo Delete "%VENV_DIR%" and rerun this installer.
        exit /b 1
    )
    echo Creating Python 3.12 environment at "%VENV_DIR%"...
    py -3.12 -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo ERROR: Could not create the Python 3.12 environment.
        exit /b 1
    )
)

echo Upgrading pip in the isolated environment...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed.
    exit /b 1
)

echo Installing pinned claude-real-video package...
"%VENV_PY%" -m pip install --upgrade "claude-real-video==0.10.3"
if errorlevel 1 (
    echo ERROR: claude-real-video installation failed.
    exit /b 1
)

echo.
echo Verifying installation...
"%VENV_PY%" --version
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip show claude-real-video
if errorlevel 1 exit /b 1
"%CRV_EXE%" --help
if errorlevel 1 exit /b 1

"%VENV_PY%" -c "import importlib.metadata as m, sys; version = m.version('claude-real-video'); print('Verified CRV version:', version); sys.exit(0 if version == '0.10.3' else 1)"
if errorlevel 1 (
    echo ERROR: Installed claude-real-video version is not 0.10.3.
    exit /b 1
)
set "CRV_VERSION=0.10.3"

echo.
echo Installation succeeded.
echo Python launcher:
where py
echo Python environment: "%VENV_PY%"
"%VENV_PY%" --version
echo FFmpeg path:
where ffmpeg
ffmpeg -version | findstr /b /c:"ffmpeg version"
echo FFprobe path:
where ffprobe
ffprobe -version | findstr /b /c:"ffprobe version"
echo CRV executable: "%CRV_EXE%"
echo CRV version: %CRV_VERSION%
exit /b 0

:install_python
where winget >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 is unavailable and winget was not found.
    exit /b 1
)
echo Installing Python 3.12 with winget...
winget install -e --id Python.Python.3.12
if errorlevel 1 (
    echo ERROR: winget could not install Python.Python.3.12.
    exit /b 1
)
exit /b 0
