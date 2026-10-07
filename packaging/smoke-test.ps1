<#
.SYNOPSIS
  Install / verify / uninstall smoke test of one ClusterLM MSI on a Windows machine (CI runner or HQ-INSTALL-01).

.DESCRIPTION
  Node:   msiexec /i /qn; service ClusterLMNode exists, LocalService, delayed auto start, recovery 5/30/60 s, SID type,
          preshutdown; firewall rule scoped to the installed clusterlm-node.exe, TCP, Private+Domain, no Public; helper Run
          value; the service reaches Running; staging ACL; a planted lease-store orphan under staging; msiexec /x; then
          service, firewall rules, Run value, install directory, staging directory and processes are gone.
  Father: msiexec /i /qn; executables, catalog, licenses present; agent Run value; PATH; the CLI starts with no runtime
          DLL installed; msiexec /x; all gone.
  Needs an elevated PowerShell. Every check prints PASS/FAIL; the script exits 1 if any FAIL (after uninstalling).
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][ValidateSet('Node', 'Father')][string]$Package,
  [Parameter(Mandatory)][string]$Msi,
  [string]$LogDir = (Join-Path $PSScriptRoot 'out\logs'),
  [int]$Port = 47600
)
$ErrorActionPreference = 'Stop'
$Msi = (Resolve-Path $Msi).Path
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogDir = (Resolve-Path $LogDir).Path
$failures = New-Object System.Collections.Generic.List[string]

function Check([string]$Name, [bool]$Ok, [string]$Detail = '') {
  if ($Ok) { Write-Host "PASS  $Name" } else { Write-Host "FAIL  $Name  $Detail"; $script:failures.Add($Name) }
}
function Run-Msi([string[]]$MsiArgs, [string]$Tag) {
  $log = Join-Path $LogDir "$Package-$Tag.log"
  $p = Start-Process -FilePath msiexec.exe -ArgumentList (@($MsiArgs) + @('/qn', '/norestart', '/l*v', "`"$log`"")) -Wait -PassThru
  Write-Host "msiexec $Tag exit code $($p.ExitCode) (log: $log)"
  return $p.ExitCode
}
function Sc-Out([string[]]$ScArgs) { return ((& sc.exe @ScArgs 2>&1) | Out-String) }
$runKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run'
function Run-Value([string]$Name) { return (Get-ItemProperty -Path $runKey -Name $Name -ErrorAction SilentlyContinue).$Name }

# The Node data directories are owner+SYSTEM-only on purpose (ADR 0133): even an elevated Administrator gets Access
# denied, and Test-Path would report False for a directory that exists. Everything under %ProgramData%\ClusterLM\Node
# is therefore inspected as SYSTEM, through a one-shot scheduled task.
function Invoke-AsSystem([string]$Script) {
  $id = [guid]::NewGuid().ToString('N')
  $out = Join-Path $LogDir "system-$id.txt"
  $ps1 = Join-Path $LogDir "system-$id.ps1"
  Set-Content -Path $ps1 -Encoding UTF8 -Value "& { $Script } *>&1 | Out-File -Encoding utf8 -FilePath '$out'; Add-Content -Path '$out' -Value '#done'"
  $task = "clusterlm-smoke-$id"
  $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$ps1`""
  $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
  Register-ScheduledTask -TaskName $task -Action $action -Principal $principal | Out-Null
  try {
    Start-ScheduledTask -TaskName $task
    for ($i = 0; $i -lt 90; $i++) {
      if ((Test-Path $out) -and ((Get-Content $out -Raw) -match '#done')) { break }
      Start-Sleep -Seconds 1
    }
  } finally { Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue }
  if (-not (Test-Path $out)) { return 'ERROR: the SYSTEM task produced no output' }
  return ((Get-Content $out -Raw) -replace '#done', '').Trim()
}

$installDir = Join-Path $env:ProgramFiles "ClusterLM $Package"
$programData = Join-Path $env:ProgramData 'ClusterLM\Node'
$staging = Join-Path $programData 'staging'

# ---------------------------------------------------------------- install
$rc = Run-Msi @('/i', "`"$Msi`"") 'install'
Check "$Package msiexec /i succeeds" ($rc -eq 0 -or $rc -eq 3010) "exit code $rc"
Check 'install directory exists' (Test-Path $installDir) $installDir

if ($Package -eq 'Node') {
  $svc = 'ClusterLMNode'
  $exe = Join-Path $installDir 'bin\clusterlm-node-service.exe'
  $worker = Join-Path $installDir 'bin\clusterlm-node.exe'
  foreach ($f in 'clusterlm-node-service.exe', 'clusterlm-node.exe', 'clusterlm-node-helper.exe') {
    Check "file bin\$f" (Test-Path (Join-Path $installDir "bin\$f"))
  }
  Check 'file licenses\THIRD-PARTY-NOTICES.txt' (Test-Path (Join-Path $installDir 'licenses\THIRD-PARTY-NOTICES.txt'))

  $qc = Sc-Out @('qc', $svc)
  Write-Host $qc
  Check 'service exists' ($qc -match 'SERVICE_NAME') $qc
  Check 'service account is LocalService' ($qc -match 'SERVICE_START_NAME\s*:\s*NT AUTHORITY\\LocalService')
  Check 'service start type is AUTO_START (DELAYED)' ($qc -match 'AUTO_START\s+\(DELAYED\)')
  Check 'service binary is the installed exe' ($qc -match [regex]::Escape($exe))
  $qf = Sc-Out @('qfailure', $svc)
  Write-Host $qf
  Check 'recovery: restart at 5000/30000/60000 ms' (($qf -match 'RESTART -- Delay = 5000') -and ($qf -match 'RESTART -- Delay = 30000') -and ($qf -match 'RESTART -- Delay = 60000')) $qf
  Check 'recovery: reset period 86400 s' ($qf -match 'RESET_PERIOD[^:]*:\s*86400')
  $sid = Sc-Out @('qsidtype', $svc)
  Check 'service SID type is UNRESTRICTED' ($sid -match 'UNRESTRICTED') $sid
  # SERVICE_CONFIG_PRESHUTDOWN_INFO is stored as the service key's PreshutdownTimeout (DWORD, ms); sc.exe has no
  # query verb that prints it on every Windows build, so read the value the SCM persisted.
  $pre = (Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\$svc" -Name PreshutdownTimeout -ErrorAction SilentlyContinue).PreshutdownTimeout
  Check 'preshutdown timeout is 15000 ms' ($pre -eq 15000) "PreshutdownTimeout=$pre"

  # Firewall: the rule the service created with the same code as `clusterlm-node-service --print-firewall-specs`.
  $rule = Get-NetFirewallRule -DisplayName 'ClusterLM Node data (TCP-In)' -ErrorAction SilentlyContinue
  Check 'firewall rule exists' ($null -ne $rule)
  if ($rule) {
    $app = $rule | Get-NetFirewallApplicationFilter
    $portf = $rule | Get-NetFirewallPortFilter
    $addr = $rule | Get-NetFirewallAddressFilter
    $profiles = [string]$rule.Profile
    Write-Host "rule: direction=$($rule.Direction) action=$($rule.Action) profile=$profiles program=$($app.Program) proto=$($portf.Protocol) port=$($portf.LocalPort) remote=$($addr.RemoteAddress) edge=$($rule.EdgeTraversalPolicy)"
    Check 'firewall rule is inbound allow' ($rule.Direction -eq 'Inbound' -and $rule.Action -eq 'Allow')
    Check 'firewall rule scoped to the installed worker program' ($app.Program -ieq $worker) $app.Program
    Check 'firewall rule is TCP on the Node port' ($portf.Protocol -eq 'TCP' -and [string]$portf.LocalPort -eq [string]$Port)
    Check 'firewall rule excludes the Public profile' ($profiles -notmatch 'Public' -and $profiles -notmatch 'Any')
    Check 'firewall rule remote scope is not Any' ([string]$addr.RemoteAddress -ne 'Any')
    Check 'firewall rule has edge traversal off' ($rule.EdgeTraversalPolicy -eq 'Block')
  }
  Check 'helper Run value points at the installed helper' ((Run-Value 'ClusterLMNodeHelper') -like '*clusterlm-node-helper.exe*')

  # The service starts the worker and creates its data directories (owner+SYSTEM-only, ADR 0133).
  $running = $false
  for ($i = 0; $i -lt 30; $i++) {
    if ((Get-Service $svc -ErrorAction SilentlyContinue).Status -eq 'Running') { $running = $true; break }
    Start-Sleep -Seconds 1
  }
  Check 'service reaches Running within 30 s' $running (Sc-Out @('query', $svc))
  if ($running) {
    Start-Sleep -Seconds 2
    Check 'staging directory created by the service' ((Invoke-AsSystem "Test-Path '$staging'") -eq 'True')
    $acl = Invoke-AsSystem "icacls.exe '$staging'"
    Write-Host $acl
    Check 'staging ACL names no Users, Administrators or Everyone' ($acl -notmatch 'BUILTIN\\Users' -and $acl -notmatch 'BUILTIN\\Administrators' -and $acl -notmatch 'Everyone') $acl
    Check 'staging ACL has no inherited entries' ($acl -notmatch '\(I\)') $acl
  }
  # Stop the service and plant a lease-store orphan: uninstall must delete it (model fragments must not survive).
  Sc-Out @('stop', $svc) | Out-Null
  for ($i = 0; $i -lt 30 -and (Get-Service $svc -ErrorAction SilentlyContinue).Status -ne 'Stopped'; $i++) { Start-Sleep -Seconds 1 }
  $plant = "New-Item -ItemType Directory -Force -Path '$staging\leases\9' | Out-Null; " +
           "[System.IO.File]::WriteAllBytes('$staging\leases\9\obj-0.part', (New-Object byte[] 4096)); Test-Path '$staging\leases\9\obj-0.part'"
  Check 'planted a lease-store orphan under staging' ((Invoke-AsSystem $plant) -eq 'True')
}
else {
  foreach ($f in 'clusterlm-father-agent.exe', 'clusterlm-father.exe', 'clusterlm-bench.exe') {
    Check "file bin\$f" (Test-Path (Join-Path $installDir "bin\$f"))
  }
  # Present only when the build contained the target (model inspector, UI): informational.
  foreach ($f in 'clusterlm-model-inspect.exe', 'clusterlm-father-ui.exe') {
    $state = if (Test-Path (Join-Path $installDir "bin\$f")) { 'installed' } else { 'not in this build' }
    Write-Host "info  bin\$f $state"
  }
  Check 'shipped catalog installed' (Test-Path (Join-Path $installDir 'catalog\clusterlm-catalog.json'))
  Check 'licenses installed' (Test-Path (Join-Path $installDir 'licenses\THIRD-PARTY-NOTICES.txt'))
  Check 'agent autostart Run value' ((Run-Value 'ClusterLMFatherAgent') -like '*clusterlm-father-agent.exe*')
  $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  Check 'CLI directory on the machine PATH' ($machinePath -like "*$installDir\bin*")
  $cli = Start-Process -FilePath (Join-Path $installDir 'bin\clusterlm-father.exe') -ArgumentList '--help' -PassThru -NoNewWindow `
         -RedirectStandardOutput (Join-Path $LogDir 'father-help.txt') -RedirectStandardError (Join-Path $LogDir 'father-help-err.txt')
  if (-not $cli.WaitForExit(20000)) { $cli.Kill() }
  Write-Host "clusterlm-father --help exit code $($cli.ExitCode) (must run with no runtime DLL installed)"
  Check 'clusterlm-father starts (not a missing-DLL 0xC0000135 / 0xC000007B)' ($cli.ExitCode -ne -1073741515 -and $cli.ExitCode -ne -1073741701)
}

# ---------------------------------------------------------------- uninstall
$rc = Run-Msi @('/x', "`"$Msi`"") 'uninstall'
Check "$Package msiexec /x succeeds" ($rc -eq 0 -or $rc -eq 3010) "exit code $rc"
Check 'install directory removed' (-not (Test-Path $installDir)) $installDir
if ($Package -eq 'Node') {
  Check 'service removed' ((Sc-Out @('query', 'ClusterLMNode')) -match '1060')
  Check 'firewall rules removed' ($null -eq (Get-NetFirewallRule -DisplayName 'ClusterLM *' -ErrorAction SilentlyContinue))
  Check 'helper Run value removed' ($null -eq (Run-Value 'ClusterLMNodeHelper'))
  Check 'staging directory removed (no model fragments left)' ((Invoke-AsSystem "Test-Path '$staging'") -eq 'False') $staging
  Check 'no clusterlm-node processes left' ($null -eq (Get-Process -Name 'clusterlm-node*' -ErrorAction SilentlyContinue))
  $left = Invoke-AsSystem "if (Test-Path '$programData') { (Get-ChildItem -Force '$programData' | ForEach-Object Name) -join ', ' } else { '(nothing)' }"
  Write-Host "left under ProgramData\ClusterLM\Node (kept by design unless CLUSTERLM_PURGE=1): $left"
}
else {
  Check 'agent autostart Run value removed' ($null -eq (Run-Value 'ClusterLMFatherAgent'))
  Check 'no clusterlm-father processes left' ($null -eq (Get-Process -Name 'clusterlm-father*' -ErrorAction SilentlyContinue))
  $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
  Check 'CLI directory removed from the machine PATH' ($machinePath -notlike "*$installDir\bin*")
}

if ($failures.Count -gt 0) {
  Write-Host "`n$($failures.Count) check(s) failed: $($failures -join '; ')"
  Get-ChildItem $LogDir -Filter "$Package-*.log" | ForEach-Object {
    Write-Host "---- tail of $($_.Name)"; Get-Content $_.FullName -Tail 60 -ErrorAction SilentlyContinue
  }
  exit 1
}
Write-Host "`nAll $Package smoke checks passed."
# Explicit: the CI shell wrapper exits with $LASTEXITCODE of the last native command (here sc.exe query, which
# correctly reports 1060 "service does not exist" after uninstall), not with the result of the checks.
exit 0
