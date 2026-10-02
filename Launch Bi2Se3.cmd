@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot open the repository directory.
    pause
    endlocal & exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The repository Python environment is missing.
    echo Run this command from the repository directory:
    echo     uv sync --frozen --extra visualization
    set "launch_exit=1"
    goto failed
)
".venv\Scripts\python.exe" -B "interactive\detector_viewer.py" --execution-backend cpu --presentation-backend matplotlib %*
set "launch_exit=%errorlevel%"
if "%launch_exit%"=="0" goto finished
echo.
echo ERROR: The Bi2Se3 viewer exited with code %launch_exit%.
echo If dependencies are missing, run from the repository directory:
echo     uv sync --frozen --extra visualization
:failed
pause
:finished
popd
endlocal & exit /b %launch_exit%
