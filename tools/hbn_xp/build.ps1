param(
    [Parameter(Mandatory = $true)][string]$CompilerRoot,
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
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$arguments = @(
    '-std=c11', '-O2', '-Wall', '-Wextra', '-Wpedantic', '-Werror',
    '-D_WIN32_WINNT=0x0501', '-DWINVER=0x0501', '-march=pentium4',
    '-msse2', '-mfpmath=sse', '-ffp-contract=off', '-static', '-mwindows',
    (Join-Path $PSScriptRoot 'hbn_core.c'), (Join-Path $PSScriptRoot 'hbn_io.c'),
    (Join-Path $PSScriptRoot 'osc_app.c'),
    '-Wl,--subsystem,windows:5.1', '-Wl,--major-os-version,5', '-Wl,--minor-os-version,1',
    '-Wl,--no-insert-timestamp', '-Wl,--strip-all', '-lcomdlg32', '-lgdi32', '-luser32', '-lm',
    '-o', (Join-Path $destination 'SLATE-OSC-XP.exe')
)
& $compiler @arguments
if ($LASTEXITCODE -ne 0) { throw 'C compilation failed.' }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'instrument.ini') -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $destination 'README.txt')
Copy-Item -LiteralPath (Join-Path $CompilerRoot 'LICENSE.TXT') -Destination (Join-Path $destination 'LLVM-LICENSE.txt')
Copy-Item -LiteralPath (Join-Path $CompilerRoot 'i686-w64-mingw32/share/mingw32/COPYING.MinGW-w64-runtime.txt') -Destination $destination
Get-FileHash -LiteralPath (Join-Path $destination 'SLATE-OSC-XP.exe') -Algorithm SHA256
