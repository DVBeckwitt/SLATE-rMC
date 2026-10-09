param(
    [Parameter(Mandatory = $true)][string]$CompilerRoot,
    [Parameter(Mandatory = $true)][string]$GemmiRoot,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$destination = [IO.Path]::GetFullPath($OutputDirectory)
if ($destination.StartsWith($repository + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or $destination -eq $repository) {
    throw 'Build outputs must be outside the repository.'
}
$compiler = Join-Path $CompilerRoot 'bin/i686-w64-mingw32-clang.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Use the documented LLVM-MinGW-XP compiler distribution.' }
$gemmiVersion = Get-Content -LiteralPath (Join-Path $GemmiRoot 'include/gemmi/version.hpp') -Raw
if ($gemmiVersion -notmatch '#define GEMMI_VERSION "0\.7\.5"') { throw 'Use the documented Gemmi 0.7.5 source.' }
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$common = @(
    '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
    '-D_WIN32_WINNT=0x0501', '-DWINVER=0x0501', '-march=pentium4',
    '-msse2', '-mfpmath=sse', '-ffp-contract=off'
)
$objects = @()
foreach ($source in @('hbn_core', 'hbn_io', 'osc_app', 'osc_view', 'osc_output', 'osc_analysis', 'osc_analysis_io', 'osc_analysis_ui', 'osc_angle_view', 'osc_cif_ui')) {
    $object = Join-Path $destination ($source + '.o')
    & $compiler @common '-std=c11' '-c' (Join-Path $PSScriptRoot ($source + '.c')) '-o' $object
    if ($LASTEXITCODE -ne 0) { throw "C compilation failed: $source" }
    $objects += $object
}
$cpp = Join-Path $CompilerRoot 'bin/i686-w64-mingw32-clang++.exe'
foreach ($source in @((Join-Path $PSScriptRoot 'cif_peaks.cpp'), (Join-Path $GemmiRoot 'src/symmetry.cpp'), (Join-Path $GemmiRoot 'src/sprintf.cpp'))) {
    $object = Join-Path $destination ([IO.Path]::GetFileNameWithoutExtension($source) + '.o')
    & $cpp @common '-std=c++17' '-isystem' (Join-Path $GemmiRoot 'include') '-c' $source '-o' $object
    if ($LASTEXITCODE -ne 0) { throw "C++ compilation failed: $source" }
    $objects += $object
}
$arguments = @('-static', '-mwindows') + $objects + @(
    '-Wl,--subsystem,windows:5.1', '-Wl,--major-os-version,5', '-Wl,--minor-os-version,1',
    '-Wl,--no-insert-timestamp', '-Wl,--strip-all', '-lcomdlg32', '-lcomctl32', '-lgdi32', '-luser32', '-lm',
    '-o', (Join-Path $destination 'SLATE-OSC-XP.exe')
)
& $cpp @arguments
if ($LASTEXITCODE -ne 0) { throw 'Link failed.' }
& $Python '-B' (Join-Path $PSScriptRoot 'export_cif_data.py') (Join-Path $destination 'cif_scattering.bin')
if ($LASTEXITCODE -ne 0) { throw 'Scattering table generation failed.' }
foreach ($object in $objects) { Remove-Item -LiteralPath $object }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'instrument.ini') -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $destination 'README.txt')
Copy-Item -LiteralPath (Join-Path $CompilerRoot 'LICENSE.TXT') -Destination (Join-Path $destination 'LLVM-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $CompilerRoot 'i686-w64-mingw32/share/mingw32/COPYING.MinGW-w64-runtime.txt') -Destination $destination
Copy-Item -LiteralPath (Join-Path $GemmiRoot 'LICENSE.txt') -Destination (Join-Path $destination 'Gemmi-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $GemmiRoot 'include/gemmi/third_party/tao/LICENSE') -Destination (Join-Path $destination 'PEGTL-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'CIF-NOTICES.txt') -Destination $destination
Get-FileHash -LiteralPath (Join-Path $destination 'SLATE-OSC-XP.exe') -Algorithm SHA256
