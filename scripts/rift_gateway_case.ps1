$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# 等 riftsim 预热完成
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
  try {
    $h = Invoke-RestMethod -Uri 'http://127.0.0.1:7862/healthz' -Method Get -TimeoutSec 3
    if ($h.ok) { $ok = $true; break }
  } catch { Start-Sleep -Seconds 2 }
}
if (-not $ok) { Write-Error 'riftsim not healthy'; exit 1 }
'riftsim healthy'

$question = '我方【天角牧者】位于战场A，对方【迅捷蟹】位于战场B。我方打出【我流奥义！隼舞】选择对【迅捷蟹】造成2点伤害并将【天角牧者】移动至战场B，此时【迅捷蟹】的绝念技能和【天角牧者】的移动技能的结算顺序是什么？'
$payload = @{ query = $question; mode = 'agent'; trace = $true }
$body = [System.Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Compress))
$resp = Invoke-WebRequest -Uri 'http://127.0.0.1:7860/api/rift/query' -Method Post -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 240
"gateway status: $([int]$resp.StatusCode)"
$resp.Content | Out-File -Encoding utf8 'scripts\rift_gateway_case.out.json'
'saved scripts\rift_gateway_case.out.json'
