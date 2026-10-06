<#
.SYNOPSIS
  Fails if any staged ClusterLM executable/DLL imports a DLL that is not part of Windows itself.

.DESCRIPTION
  End-user machines have no compiler, Python, Git, CUDA toolkit or Visual C++ Redistributable, and the installers
  contain no OpenSSL DLL: everything is linked statically (vcpkg x64-windows-static, /MT). Uses dumpbin from the MSVC
  environment. Prints the full import table so a failure names the offending DLL.
#>
[CmdletBinding()]
param([Parameter(Mandatory)][string[]]$Stage)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command dumpbin.exe -ErrorAction SilentlyContinue)) { throw 'dumpbin.exe not found (run inside a Visual Studio developer environment)' }

# DLLs present on every supported Windows 11 installation (system32), plus API sets and the Universal CRT forwarders.
$allowed = '^(kernel32|kernelbase|user32|advapi32|ws2_32|mswsock|crypt32|ole32|oleaut32|shell32|shlwapi|bcrypt|ncrypt|' +
           'secur32|sechost|rpcrt4|ntdll|dxgi|powrprof|wtsapi32|userenv|iphlpapi|version|win32u|gdi32|imm32|msvcrt|ucrtbase|' +
           'netapi32|wininet|winhttp|cfgmgr32|setupapi|api-ms-win-.*|ext-ms-win-.*)\.dll$'
$bad = @()
foreach ($root in $Stage) {
  foreach ($f in Get-ChildItem -Recurse -File $root | Where-Object { $_.Extension -in '.exe', '.dll' }) {
    $out = & dumpbin.exe /nologo /dependents $f.FullName
    $deps = $out | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^[A-Za-z0-9_.\-]+\.dll$' }
    Write-Host ("{0}: {1}" -f $f.Name, ($deps -join ', '))
    foreach ($d in $deps) {
      if ($d.ToLowerInvariant() -notmatch $allowed) { $bad += "$($f.Name) -> $d" }
    }
  }
}
if ($bad.Count -gt 0) {
  $bad | ForEach-Object { Write-Error $_ -ErrorAction Continue }
  throw "Executables import non-system DLLs (static linking broken, or ship them deliberately and update docs/packaging.md): $($bad -join '; ')"
}
Write-Host 'OK: every import is a Windows system DLL.'
