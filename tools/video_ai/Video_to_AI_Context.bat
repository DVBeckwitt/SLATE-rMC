@echo off
setlocal EnableExtensions

set "CRV_NO_MEMORY=1"
set "CRV_NO_HINT=1"
set "CRV_ROOT=%LOCALAPPDATA%\CRV"
set "VENV_PY=%CRV_ROOT%\venv\Scripts\python.exe"
set "CRV_EXE=%CRV_ROOT%\venv\Scripts\crv.exe"

if not exist "%CRV_EXE%" (
    echo The local Video AI tool is not installed.
    choice /c YN /n /m "Run the installer now? [Y/N] "
    if errorlevel 2 exit /b 1
    call "%~dp0Install_Video_AI.bat"
    if errorlevel 1 exit /b 1
)

ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: ffmpeg is unavailable. Run Install_Video_AI.bat, then open a new terminal and try again.
    exit /b 1
)
ffprobe -version >nul 2>&1
if errorlevel 1 (
    echo ERROR: ffprobe is unavailable. Run Install_Video_AI.bat, then open a new terminal and try again.
    exit /b 1
)

set "VIDEO_PATH="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -STA -Command "Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Title = 'Select a video for local AI preprocessing'; $dialog.Filter = 'Video files|*.mp4;*.mov;*.mkv;*.avi;*.wmv;*.m4v;*.webm;*.mpeg;*.mpg|All files|*.*'; $dialog.Multiselect = $false; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileName }"`) do set "VIDEO_PATH=%%F"

if not defined VIDEO_PATH (
    echo No video selected. Nothing was changed.
    exit /b 0
)

echo.
echo Choose an extraction profile:
echo   1. Routine
echo   2. Detailed
echo   3. Audit and tune
echo.
echo Detailed is best for fast hand movements, brief apparatus states,
echo small objects, or dense visual content.
echo Audit and tune keeps rejected candidates in a dropped folder and uses more disk space.
choice /c 123 /n /m "Profile [1/2/3]: "
if errorlevel 3 goto :audit
if errorlevel 2 goto :detailed
goto :routine

:routine
set "PROFILE_NAME=Routine"
set "FPS_FLOOR=1"
set "MAX_FRAMES=300"
set "FRAME_WIDTH=960"
set "DEDUP_THRESHOLD="
set "DEDUP_ARG="
set "REPORT_ARG="
goto :run

:detailed
set "PROFILE_NAME=Detailed"
set "FPS_FLOOR=0.5"
set "MAX_FRAMES=600"
set "FRAME_WIDTH=1280"
set "DEDUP_THRESHOLD=5"
set "DEDUP_ARG=--dedup-threshold 5"
set "REPORT_ARG="
goto :run

:audit
set "PROFILE_NAME=Audit and tune"
set "FPS_FLOOR=1"
set "MAX_FRAMES=300"
set "FRAME_WIDTH=960"
set "DEDUP_THRESHOLD="
set "DEDUP_ARG="
set "REPORT_ARG=--report"
goto :run

:run
for %%F in ("%VIDEO_PATH%") do (
    set "VIDEO_DIR=%%~dpF"
    set "VIDEO_BASE=%%~nF"
)

:choose_output
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "STAMP=%%T"
set "OUTPUT_DIR=%VIDEO_DIR%%VIDEO_BASE%_AI_%STAMP%"
if exist "%OUTPUT_DIR%" (
    powershell -NoProfile -Command "Start-Sleep -Seconds 1"
    goto :choose_output
)
mkdir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create output folder "%OUTPUT_DIR%".
    exit /b 1
)

set "RUN_LOG=%OUTPUT_DIR%\run.log"
set "WHY_TEXT=Describe visible actions, intermediate state changes, and outcomes in chronological order"
echo Video AI run started at %DATE% %TIME%> "%RUN_LOG%"
echo Source: "%VIDEO_PATH%">> "%RUN_LOG%"
echo Profile: %PROFILE_NAME%>> "%RUN_LOG%"

echo Command: "%CRV_EXE%" "%VIDEO_PATH%" -o "%OUTPUT_DIR%" --no-transcribe --adaptive --fps-floor %FPS_FLOOR% --dedup-window 1 %DEDUP_ARG% --max-frames %MAX_FRAMES% --frame-width %FRAME_WIDTH% --viewer --grid --why "%WHY_TEXT%" %REPORT_ARG%
echo Command: "%CRV_EXE%" "%VIDEO_PATH%" -o "%OUTPUT_DIR%" --no-transcribe --adaptive --fps-floor %FPS_FLOOR% --dedup-window 1 %DEDUP_ARG% --max-frames %MAX_FRAMES% --frame-width %FRAME_WIDTH% --viewer --grid --why "%WHY_TEXT%" %REPORT_ARG%>> "%RUN_LOG%"
powershell -NoProfile -Command "$argsList = @($env:VIDEO_PATH, '-o', $env:OUTPUT_DIR, '--no-transcribe', '--adaptive', '--fps-floor', $env:FPS_FLOOR, '--dedup-window', '1'); if ($env:DEDUP_THRESHOLD) { $argsList += @('--dedup-threshold', $env:DEDUP_THRESHOLD) }; $argsList += @('--max-frames', $env:MAX_FRAMES, '--frame-width', $env:FRAME_WIDTH, '--viewer', '--grid', '--why', $env:WHY_TEXT); if ($env:REPORT_ARG) { $argsList += '--report' }; & $env:CRV_EXE @argsList 2>&1 | ForEach-Object { $_; $_ | Out-File -LiteralPath $env:RUN_LOG -Append -Encoding utf8 }; exit $LASTEXITCODE"
set "RUN_EXIT=%ERRORLEVEL%"

:after_run
if not "%RUN_EXIT%"=="0" (
    echo ERROR: CRV frame-extraction command failed with exit code %RUN_EXIT%.
    echo ERROR: CRV frame-extraction command failed with exit code %RUN_EXIT%.>> "%RUN_LOG%"
    echo See "%RUN_LOG%".
    exit /b %RUN_EXIT%
)

set "TRANSCRIPT_PATH="
if exist "%VIDEO_DIR%%VIDEO_BASE%.srt" set "TRANSCRIPT_PATH=%VIDEO_DIR%%VIDEO_BASE%.srt"
if not defined TRANSCRIPT_PATH if exist "%VIDEO_DIR%%VIDEO_BASE%.vtt" set "TRANSCRIPT_PATH=%VIDEO_DIR%%VIDEO_BASE%.vtt"
if not defined TRANSCRIPT_PATH if exist "%VIDEO_DIR%%VIDEO_BASE%.json" set "TRANSCRIPT_PATH=%VIDEO_DIR%%VIDEO_BASE%.json"
if not defined TRANSCRIPT_PATH if exist "%VIDEO_DIR%%VIDEO_BASE%.txt" set "TRANSCRIPT_PATH=%VIDEO_DIR%%VIDEO_BASE%.txt"
if not defined TRANSCRIPT_PATH if exist "%VIDEO_PATH%.srt" set "TRANSCRIPT_PATH=%VIDEO_PATH%.srt"
if not defined TRANSCRIPT_PATH if exist "%VIDEO_PATH%.vtt" set "TRANSCRIPT_PATH=%VIDEO_PATH%.vtt"

if defined TRANSCRIPT_PATH (
    for %%T in ("%TRANSCRIPT_PATH%") do copy /y "%TRANSCRIPT_PATH%" "%OUTPUT_DIR%\external_transcript%%~xT" >nul
    if errorlevel 1 (
        echo ERROR: External transcript copy failed.
        echo ERROR: External transcript copy failed.>> "%RUN_LOG%"
        echo See "%RUN_LOG%".
        exit /b 1
    )
    echo Copied external transcript from "%TRANSCRIPT_PATH%".
    echo Copied external transcript from "%TRANSCRIPT_PATH%".>> "%RUN_LOG%"
) else (
    echo No matching external transcript was found; none was copied.
    echo No matching external transcript was found; none was copied.>> "%RUN_LOG%"
)

set "VERSION_FILE=%TEMP%\crv-version-%RANDOM%-%RANDOM%.txt"
"%VENV_PY%" -c "import importlib.metadata as m; print(m.version('claude-real-video'))" > "%VERSION_FILE%" 2>> "%RUN_LOG%"
if errorlevel 1 (
    echo ERROR: CRV version lookup failed.
    echo ERROR: CRV version lookup failed.>> "%RUN_LOG%"
    echo See "%RUN_LOG%".
    exit /b 1
)
set /p "CRV_VERSION="< "%VERSION_FILE%"
del /q "%VERSION_FILE%"

for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "Get-Date -Format o"`) do set "RUN_TIME=%%T"
set "INPUTS_FILE=%OUTPUT_DIR%\INPUTS.txt"
powershell -NoProfile -Command "$lines = @('source video path: ' + $env:VIDEO_PATH, 'selected profile: ' + $env:PROFILE_NAME, 'CRV version: ' + $env:CRV_VERSION, 'run date and time: ' + $env:RUN_TIME, 'external transcript path: ' + $(if ($env:TRANSCRIPT_PATH) { $env:TRANSCRIPT_PATH } else { '(none)' }), 'output folder: ' + $env:OUTPUT_DIR); [IO.File]::WriteAllLines($env:INPUTS_FILE, $lines, [Text.UTF8Encoding]::new($false))"
if errorlevel 1 (
    echo ERROR: INPUTS.txt creation failed.
    echo ERROR: INPUTS.txt creation failed.>> "%RUN_LOG%"
    echo See "%RUN_LOG%".
    exit /b 1
)

set "SUMMARY_FILE=%TEMP%\crv-summary-%RANDOM%-%RANDOM%.txt"
powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; $required = @('MANIFEST.txt', 'frames.json', 'viewer.html'); foreach ($name in $required) { $path = Join-Path $env:OUTPUT_DIR $name; if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Get-Item -LiteralPath $path).Length -eq 0) { throw ('Missing or empty required output: ' + $path) } }; $framesDir = Join-Path $env:OUTPUT_DIR 'frames'; if (-not (Test-Path -LiteralPath $framesDir -PathType Container)) { throw ('Missing frames directory: ' + $framesDir) }; $data = Get-Content -LiteralPath (Join-Path $env:OUTPUT_DIR 'frames.json') -Raw | ConvertFrom-Json; $frames = @($data.frames); if ($frames.Count -lt 1) { throw 'frames.json contains no frames' }; $previous = [double]::NegativeInfinity; foreach ($frame in $frames) { if ($null -eq $frame.timestamp_sec -or $frame.timestamp_sec -is [string] -or $frame.timestamp_sec -is [bool]) { throw 'Frame timestamp is not a JSON number' }; $timestamp = [double]$frame.timestamp_sec; if ([double]::IsNaN($timestamp) -or [double]::IsInfinity($timestamp) -or $timestamp -lt $previous) { throw 'Frame timestamps are not numeric and monotonically increasing' }; $previous = $timestamp; $fileName = [string]$frame.file; if ([IO.Path]::GetFileName($fileName) -ne $fileName) { throw ('Unsafe frame filename in frames.json: ' + $fileName) }; $imagePath = Join-Path $framesDir $fileName; if (-not (Test-Path -LiteralPath $imagePath -PathType Leaf) -or (Get-Item -LiteralPath $imagePath).Length -eq 0) { throw ('Missing frame image: ' + $imagePath) } }; $gridsDir = Join-Path $env:OUTPUT_DIR 'grids'; $gridCount = @(Get-ChildItem -LiteralPath $gridsDir -File -ErrorAction Stop).Count; if ($gridCount -lt 1) { throw ('No contact sheet found in ' + $gridsDir) }; $summary = '{0}|{1:R}|{2:R}' -f $frames.Count, [double]$frames[0].timestamp_sec, [double]$frames[-1].timestamp_sec; [IO.File]::WriteAllText($env:SUMMARY_FILE, $summary, [Text.UTF8Encoding]::new($false))" >> "%RUN_LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: Output validation failed after CRV extraction.
    echo ERROR: Output validation failed after CRV extraction.>> "%RUN_LOG%"
    echo See "%RUN_LOG%".
    exit /b 1
)

for /f "usebackq tokens=1-3 delims=|" %%A in ("%SUMMARY_FILE%") do (
    set "FRAME_COUNT=%%A"
    set "FIRST_TIMESTAMP=%%B"
    set "LAST_TIMESTAMP=%%C"
)
del /q "%SUMMARY_FILE%"

echo.
echo Success: retained %FRAME_COUNT% frames.
echo First timestamp: %FIRST_TIMESTAMP% seconds
echo Last timestamp: %LAST_TIMESTAMP% seconds
echo Output folder: "%OUTPUT_DIR%"
echo Success: retained %FRAME_COUNT% frames.>> "%RUN_LOG%"
echo First timestamp: %FIRST_TIMESTAMP% seconds>> "%RUN_LOG%"
echo Last timestamp: %LAST_TIMESTAMP% seconds>> "%RUN_LOG%"
echo Output folder: "%OUTPUT_DIR%">> "%RUN_LOG%"

start "" explorer.exe "%OUTPUT_DIR%"
start "" "%OUTPUT_DIR%\viewer.html"
exit /b 0
