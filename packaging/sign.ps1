<#
.SYNOPSIS
  Authenticode-signs ClusterLM binaries and installers with signtool, or says loudly that it did not.

.DESCRIPTION
  Signs every .exe/.dll/.msi under -Path when ALL of these are present (GitHub secrets / environment):
    CLUSTERLM_SIGN_CERT_BASE64      base64 of the .pfx
    CLUSTERLM_SIGN_CERT_PASSWORD    its password
    CLUSTERLM_SIGN_TIMESTAMP_URL    RFC 3161 timestamp server (e.g. the one your CA documents)
  Otherwise NOTHING is signed: the script prints a prominent warning (and a GitHub Actions ::warning:: annotation),
  writes status "unsigned" to the status file and exits 0, so build-msi.ps1 puts UNSIGNED into the artifact name.
  It never creates or uses a self-signed or test certificate: an unsigned package must look unsigned.

  -Require turns "not configured" into a failure (release builds).
  -Probe only reports whether signing is configured ("signed" | "unsigned" on the pipeline) and signs nothing.
#>
[CmdletBinding()]
param(
  [string[]]$Path = @(),
  [switch]$Require,
  [switch]$Probe,
  [string]$StatusFile = ''
)
$ErrorActionPreference = 'Stop'

function Test-SigningConfigured {
  return (-not [string]::IsNullOrWhiteSpace($env:CLUSTERLM_SIGN_CERT_BASE64)) -and
         (-not [string]::IsNullOrWhiteSpace($env:CLUSTERLM_SIGN_CERT_PASSWORD)) -and
         (-not [string]::IsNullOrWhiteSpace($env:CLUSTERLM_SIGN_TIMESTAMP_URL))
}

function Write-Status([bool]$Signed, [string[]]$Files) {
  if ([string]::IsNullOrEmpty($StatusFile)) { return }
  $dir = Split-Path -Parent $StatusFile
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
  [ordered]@{ signed = $Signed; files = @($Files); tool = 'signtool'; algorithm = 'sha256' } |
    ConvertTo-Json | Set-Content -Encoding UTF8 -Path $StatusFile
}

function Find-SignTool {
  $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $roots = @("${env:ProgramFiles(x86)}\Windows Kits\10\bin", "${env:ProgramFiles}\Windows Kits\10\bin")
  foreach ($r in $roots) {
    if (-not (Test-Path $r)) { continue }
    $hit = Get-ChildItem -Path $r -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
           Where-Object { $_.FullName -match '\\x64\\' } | Sort-Object FullName -Descending | Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  throw 'signtool.exe not found (install the Windows SDK).'
}

if ($Probe) {
  if (Test-SigningConfigured) { Write-Output 'signed' } else { Write-Output 'unsigned' }
  return
}

if (-not (Test-SigningConfigured)) {
  $msg = 'CODE SIGNING NOT CONFIGURED (CLUSTERLM_SIGN_CERT_BASE64 / CLUSTERLM_SIGN_CERT_PASSWORD / CLUSTERLM_SIGN_TIMESTAMP_URL). ' +
         'Binaries and installers are UNSIGNED. Windows SmartScreen will warn. Do not distribute as a release.'
  Write-Warning $msg
  if ($env:GITHUB_ACTIONS -eq 'true') { Write-Host "::warning title=UNSIGNED BUILD::$msg" }
  Write-Host ('=' * 78); Write-Host "  $msg"; Write-Host ('=' * 78)
  if ($Require) { throw 'Signing is required (-Require) but not configured.' }
  Write-Status $false @()
  return
}

$files = @()
foreach ($p in $Path) {
  if (Test-Path -PathType Container $p) {
    $files += Get-ChildItem -Path $p -Recurse -File | Where-Object { $_.Extension -in '.exe', '.dll', '.msi' } |
              ForEach-Object { $_.FullName }
  } elseif (Test-Path -PathType Leaf $p) {
    $files += (Resolve-Path $p).Path
  } else {
    throw "Path not found: $p"
  }
}
if ($files.Count -eq 0) { throw 'No .exe/.dll/.msi files to sign.' }

$signtool = Find-SignTool
$pfx = Join-Path ([System.IO.Path]::GetTempPath()) ("clusterlm-sign-" + [guid]::NewGuid().ToString('N') + '.pfx')
try {
  [System.IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:CLUSTERLM_SIGN_CERT_BASE64))
  foreach ($f in $files) {
    # Already signed by a previous call (e.g. a third-party DLL): keep that signature.
    $sig = Get-AuthenticodeSignature -FilePath $f
    if ($sig.Status -eq 'Valid') { Write-Host "already signed: $f"; continue }
    & $signtool sign /f $pfx /p $env:CLUSTERLM_SIGN_CERT_PASSWORD /fd SHA256 /tr $env:CLUSTERLM_SIGN_TIMESTAMP_URL /td SHA256 `
        /d 'ClusterLM' $f
    if ($LASTEXITCODE -ne 0) { throw "signtool sign failed for $f" }
    & $signtool verify /pa /q $f
    if ($LASTEXITCODE -ne 0) { throw "signtool verify failed for $f" }
    Write-Host "signed: $f"
  }
} finally {
  if (Test-Path $pfx) { Remove-Item -Force $pfx }
}
Write-Status $true $files
