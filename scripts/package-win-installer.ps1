$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$package = Get-Content (Join-Path $root 'package.json') -Raw | ConvertFrom-Json
if ($package.version -notmatch '^\d+\.\d+\.\d+$') { throw 'MSI packaging requires a three-part release version.' }
$app = Join-Path $root 'release/win/TokenSmith'
if (-not (Test-Path (Join-Path $app 'TokenSmith.exe'))) { throw 'Run npm run package:win first.' }
$wixBin = if ($env:WIX) { Join-Path $env:WIX 'bin' } else { Join-Path ${env:ProgramFiles(x86)} 'WiX Toolset v3.14/bin' }
$build = Join-Path $root 'release/msi-build'
New-Item -ItemType Directory -Force $build | Out-Null
function Invoke-Wix([string]$tool, [string[]]$arguments) {
    & (Join-Path $wixBin "$tool.exe") @arguments
    if ($LASTEXITCODE -ne 0) { throw "$tool failed with exit code $LASTEXITCODE" }
}
# Harvest only the already-packaged app. Preserve upstream executables and signatures.
Invoke-Wix 'heat' @('dir', $app, '-nologo', '-gg', '-srd', '-sreg', '-scom', '-dr', 'INSTALLFOLDER', '-cg', 'AppFiles', '-var', 'var.AppDir', '-out', (Join-Path $build 'files.wxs'))
Invoke-Wix 'candle' @('-nologo', '-arch', 'x64', "-dAppDir=$app", "-dAppVersion=$($package.version)", '-out', "$build\", (Join-Path $root 'build-resources/windows-installer.wxs'), (Join-Path $build 'files.wxs'))
# ICE38/64 flag per-user file components without registry key paths; these components
# intentionally track files and directories rather than creating thousands of registry values.
Invoke-Wix 'light' @('-nologo', '-sice:ICE38', '-sice:ICE64', '-out', (Join-Path $root "release/TokenSmith-$($package.version)-win-x64.msi"), (Join-Path $build 'windows-installer.wixobj'), (Join-Path $build 'files.wixobj'))
