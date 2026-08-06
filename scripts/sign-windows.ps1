<#
.SYNOPSIS
  Authenticode-sign a Windows binary (main exe or NSIS setup) with the
  OpenSquilla code-signing certificate.

.DESCRIPTION
  Used by:
    1. tauri.conf.json -> bundle.windows.signCommand (signs the app's main
       exe while `cargo tauri build` packages it). The bundler invokes:
         powershell -NoProfile -ExecutionPolicy Bypass -File
           ../scripts/sign-windows.ps1 <path-to-binary>
    2. .github/workflows/rust-release.yml (signs the outer NSIS setup.exe
       after tauri-action finishes).

  The certificate arrives as a base64-encoded .pfx via the environment:
    WINDOWS_CERTIFICATE  - base64 of the .pfx file (GitHub secret)
    WINDOWS_CERT_PASSWORD - .pfx export password (GitHub secret)

  When WINDOWS_CERTIFICATE is not set (local builds, unsigned CI), the script
  is a no-op so `cargo tauri build` still succeeds.

.NOTES
  Requires signtool.exe (Windows SDK, preinstalled on windows-latest runners).
  Timestamps via DigiCert RFC 3161.
#>

param(
  [Parameter(Mandatory = $false, Position = 0)]
  [string]$BinaryPath
)

$ErrorActionPreference = 'Stop'

# --- no-op fast paths --------------------------------------------------------

if (-not $BinaryPath) {
  Write-Host 'sign-windows: no binary path supplied; nothing to sign.'
  exit 0
}
try {
  if (-not (Test-Path -LiteralPath $BinaryPath -PathType Leaf)) {
    Write-Host "sign-windows: binary not found at '$BinaryPath'; skipping."
    exit 0
  }
} catch {
  # The path contains characters that are illegal in a path (e.g. an
  # unexpanded `%1` placeholder carrying literal quotes). Nothing to sign.
  Write-Host "sign-windows: invalid path '$BinaryPath'; skipping."
  exit 0
}

$certB64 = $env:WINDOWS_CERTIFICATE
$certPass = $env:WINDOWS_CERT_PASSWORD
if (-not $certB64) {
  Write-Host "sign-windows: WINDOWS_CERTIFICATE is not set; skipping Authenticode signing for '$BinaryPath'."
  exit 0
}

# --- locate signtool ---------------------------------------------------------

function Find-Signtool {
  $kitRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
  if (Test-Path $kitRoot) {
    $versions = Get-ChildItem -Directory $kitRoot | Sort-Object Name -Descending
    foreach ($ver in $versions) {
      $candidates = @(
        Join-Path $ver.FullName 'x64\signtool.exe'
        Join-Path $ver.FullName 'arm64\signtool.exe'
        Join-Path $ver.FullName 'x86\signtool.exe'
      )
      foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
      }
    }
  }
  # Fall back to PATH.
  return (Get-Command signtool.exe -ErrorAction SilentlyContinue).Source
}

$signtool = Find-Signtool
if (-not $signtool) {
  Write-Host 'sign-windows: signtool.exe not found (Windows SDK missing); skipping signing.'
  exit 0
}

# --- write cert --------------------------------------------------------------

$tempDir = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$certPath = Join-Path $tempDir 'opensquilla-signing.pfx'
if (-not (Test-Path -LiteralPath $certPath)) {
  try {
    $bytes = [Convert]::FromBase64String($certB64)
    [IO.File]::WriteAllBytes($certPath, $bytes)
    Write-Host "sign-windows: wrote certificate to '$certPath'."
  } catch {
    Write-Host "sign-windows: could not decode WINDOWS_CERTIFICATE as base64: $($_.Exception.Message)"
    exit 1
  }
}

# --- sign --------------------------------------------------------------------

$timestamp = 'http://timestamp.digicert.com'
$arguments = @(
  'sign', '/fd', 'SHA256',
  '/f', $certPath,
  '/p', $certPass,
  '/tr', $timestamp, '/td', 'SHA256',
  '/v',
  $BinaryPath
)

Write-Host "sign-windows: signing '$BinaryPath'..."
& $signtool $arguments
if ($LASTEXITCODE -ne 0) {
  Write-Error "sign-windows: signtool failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "sign-windows: OK — signed '$BinaryPath'."
exit 0
