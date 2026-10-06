<#
.SYNOPSIS
  Starts the pinned llama.cpp rpc-server on a Node for the P0-A baseline (HQ-P0A-01), bound to the wired interface,
  with the persistent tensor cache DISABLED (no -c).

.DESCRIPTION
  The RPC protocol has no authentication or encryption: run this only on an isolated benchmark network, never on a
  LAN shared with real data or the ClusterLM control plane (docs/backends/llama-rpc.md).
  The server is started with an explicit argument list (no shell). LLAMA_CACHE is pointed at a fresh empty directory
  so any cache write is visible; on exit the script writes a JSON report of that directory that
  `clusterlm-bench baseline llama-rpc --node-fs-report` reads.

.PARAMETER Server   Path to ggml-rpc-server.exe built from the pinned llama.cpp (target clusterlm-llama-rpc-server).
.PARAMETER BindAddress  The wired interface address to listen on (not 0.0.0.0, not a Wi-Fi address).
.PARAMETER Port     TCP port (default 50052).
.PARAMETER Threads  CPU threads for the CPU device.
.PARAMETER Report   Where to write the filesystem report when the server exits.
.PARAMETER NodeName Name recorded in the report.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Server,
  [Parameter(Mandatory = $true)][string]$BindAddress,
  [int]$Port = 50052,
  [int]$Threads = 4,
  [string]$Report = "rpc-node-report.json",
  [string]$NodeName = $env:COMPUTERNAME
)
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Server -PathType Leaf)) { throw "rpc-server not found: $Server" }
if ($BindAddress -in @('0.0.0.0', '::')) { throw "Refusing to bind to every interface; give the wired interface address." }
if ($Port -lt 1 -or $Port -gt 65535) { throw "bad port $Port" }

$cacheRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("clusterlm-rpc-cache-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $cacheRoot | Out-Null
$env:LLAMA_CACHE = $cacheRoot        # where -c would write <cache>\rpc; stays empty without -c
$env:GGML_RPC_NO_RDMA = '1'          # plain TCP, matching the baseline harness' byte accounting

# Explicit argv, no -c. Start-Process receives the list as separate, quoted elements.
$argList = @('-H', $BindAddress, '-p', "$Port", '-t', "$Threads")
Write-Host "Starting $Server $($argList -join ' ')  (cache disabled; LLAMA_CACHE=$cacheRoot)"
$proc = Start-Process -FilePath $Server -ArgumentList $argList -NoNewWindow -PassThru
try {
  $proc.WaitForExit()
} finally {
  if (-not $proc.HasExited) { $proc.Kill() }
  $entries = @(Get-ChildItem -LiteralPath $cacheRoot -Recurse -Force -ErrorAction SilentlyContinue)
  $bytes = ($entries | Where-Object { -not $_.PSIsContainer } | Measure-Object -Property Length -Sum).Sum
  if ($null -eq $bytes) { $bytes = 0 }
  [ordered]@{
    node            = $NodeName
    cache_dir       = $cacheRoot
    cache_flag_used = $false
    cache_entries   = $entries.Count
    cache_bytes     = $bytes
  } | ConvertTo-Json | Set-Content -LiteralPath $Report -Encoding UTF8
  Write-Host "Wrote $Report ($($entries.Count) cache entries)"
  Remove-Item -LiteralPath $cacheRoot -Recurse -Force -ErrorAction SilentlyContinue
}
