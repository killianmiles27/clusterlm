<#
.SYNOPSIS
  Builds the ClusterLM Father and Node MSI packages from CMake install stages with WiX v5.

.DESCRIPTION
  Prerequisites (CI installs them; nothing here downloads anything):
    dotnet tool install --global wix --version 5.0.2
    wix extension add --global WixToolset.Util.wixext/5.0.2
  Inputs: -StageRoot holds father\ and node\ (cmake --install <build> --component father|node --prefix ...).
  Steps per package: add the OpenSSL license text, sign the stage's EXE/DLL files (sign.ps1), wix build, sign the MSI,
  hash it. Unsigned packages carry "-UNSIGNED" in the file name and "Signed: no" in the MSI summary.
  Writes <OutDir>\packaging-manifest.json (name, file, version, sha256, signed, stage contents).
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$StageRoot,
  [string]$OutDir = (Join-Path $PSScriptRoot 'out'),
  [string]$Version = '',
  [string]$OpenSslLicense = '',   # vcpkg: <root>\installed\x64-windows-static\share\openssl\copyright
  [switch]$RequireSigning,
  [string[]]$Packages = @('Father', 'Node')
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$wixExt = 'WixToolset.Util.wixext'

if ([string]::IsNullOrEmpty($Version)) {
  $cmake = Get-Content (Join-Path $repo 'CMakeLists.txt') -Raw
  if ($cmake -notmatch 'project\(clusterlm VERSION ([0-9]+\.[0-9]+\.[0-9]+)') { throw 'cannot read the project version from CMakeLists.txt' }
  $Version = $Matches[1]
}
if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?$') { throw "MSI version must be numeric x.y.z[.w]: $Version" }

if (-not (Get-Command wix -ErrorAction SilentlyContinue)) { throw 'wix not found: dotnet tool install --global wix --version 5.0.2' }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path

$signing = (& (Join-Path $PSScriptRoot 'sign.ps1') -Probe) -eq 'signed'
if ($RequireSigning -and -not $signing) { throw 'Signing is required (-RequireSigning) but CLUSTERLM_SIGN_* are not set.' }
$suffix = if ($signing) { '' } else { '-UNSIGNED' }
$signedText = if ($signing) { 'yes' } else { 'no' }
if (-not $signing) {
  & (Join-Path $PSScriptRoot 'sign.ps1') -Path @() -StatusFile (Join-Path $OutDir 'signing-status.json')   # prints the loud warning
}

$manifest = @()
foreach ($pkg in $Packages) {
  $lc = $pkg.ToLowerInvariant()
  $stage = Join-Path $StageRoot $lc
  if (-not (Test-Path (Join-Path $stage 'bin'))) { throw "stage for $pkg not found: $stage\bin (run cmake --install --component $lc)" }
  $stage = (Resolve-Path $stage).Path

  # Licenses: the OpenSSL text comes from the port that built the static libraries.
  $licDir = Join-Path $stage 'licenses'
  New-Item -ItemType Directory -Force -Path $licDir | Out-Null
  if ($OpenSslLicense -and (Test-Path $OpenSslLicense)) {
    Copy-Item -Force $OpenSslLicense (Join-Path $licDir 'OpenSSL-LICENSE.txt')
  } else {
    Write-Warning 'OpenSSL license text not supplied (-OpenSslLicense): this package is NOT release-complete.'
  }

  $hasUi = if (Test-Path (Join-Path $stage "bin\clusterlm-$lc-ui.exe")) { '1' } else { '0' }
  Write-Host "== $pkg $Version (UI executable: $hasUi, signed: $signedText)"

  if ($signing) { & (Join-Path $PSScriptRoot 'sign.ps1') -Path $stage -Require -StatusFile (Join-Path $OutDir "signing-$lc-files.json") }

  $msi = Join-Path $OutDir "ClusterLM-$pkg-$Version-x64$suffix.msi"
  & wix build -arch x64 -ext $wixExt `
      -d "StageDir=$stage" -d "Version=$Version" -d "Signed=$signedText" -d "HasUi=$hasUi" `
      (Join-Path $PSScriptRoot "wix\$pkg.wxs") -o $msi
  if ($LASTEXITCODE -ne 0) { throw "wix build failed for $pkg" }

  if ($signing) { & (Join-Path $PSScriptRoot 'sign.ps1') -Path $msi -Require }

  $files = Get-ChildItem -Recurse -File $stage | ForEach-Object { $_.FullName.Substring($stage.Length + 1).Replace('\', '/') } | Sort-Object
  $manifest += [ordered]@{
    package = "ClusterLM $pkg"; file = (Split-Path -Leaf $msi); version = $Version
    sha256 = (Get-FileHash -Algorithm SHA256 $msi).Hash.ToLowerInvariant(); signed = $signing; ui_included = ($hasUi -eq '1')
    stage_files = @($files)
  }
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $OutDir 'packaging-manifest.json')
Write-Host "Packages written to $OutDir"
