$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$installers = @(Get-ChildItem (Join-Path $root 'release/distribution/*.msi'))
if ($installers.Count -ne 1) { throw 'Expected exactly one installer for the smoke test.' }
$destination = Join-Path $env:RUNNER_TEMP 'TokenSmith-installer-test'
$log = Join-Path $env:RUNNER_TEMP 'TokenSmith-msi-install.log'
$process = Start-Process msiexec.exe -ArgumentList @('/i', "`"$($installers[0].FullName)`"", '/qn', '/norestart', "INSTALLFOLDER=`"$destination`"", '/l*v', "`"$log`"") -Wait -PassThru
if ($process.ExitCode -notin @(0, 3010)) {
    Get-Content $log -Tail 100
    throw "Installer failed with exit code $($process.ExitCode)"
}
try {
    $packaged = Join-Path $root 'release/win/TokenSmith/TokenSmith.exe'
    $installed = Join-Path $destination 'TokenSmith.exe'
    if ((Get-FileHash $packaged).Hash -ne (Get-FileHash $installed).Hash) { throw 'Installed executable differs from the packaged executable.' }
    $python = Join-Path $destination 'resources/app/app_runtime/python/python.exe'
    & $python -c 'import faiss, numpy, pypdfium2, PIL, onnxruntime, cv2; from rapidocr import RapidOCR'
    if ($LASTEXITCODE -ne 0) { throw 'Installed Python runtime failed to load.' }
} finally {
    $uninstall = Start-Process msiexec.exe -ArgumentList @('/x', "`"$($installers[0].FullName)`"", '/qn', '/norestart') -Wait -PassThru
    if ($uninstall.ExitCode -notin @(0, 3010)) { throw "Uninstaller failed with exit code $($uninstall.ExitCode)" }
}
if (Test-Path (Join-Path $destination 'TokenSmith.exe')) { throw 'Uninstaller left the application executable behind.' }
Write-Output 'Installer, installed Python runtime, and uninstaller passed.'
