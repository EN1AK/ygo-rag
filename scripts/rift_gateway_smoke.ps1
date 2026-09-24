$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$base = 'http://127.0.0.1:7860'  # kami-man 将 RIFT_RAG_BASE_URL 指向 $base/api/rift（拼 {base}/api/query）

function Post-Query($payload, $outFile) {
  $body = [System.Text.Encoding]::UTF8.GetBytes(($payload | ConvertTo-Json -Compress))
  try {
    $resp = Invoke-WebRequest -Uri "$base/api/rift/query" -Method Post -ContentType 'application/json; charset=utf-8' -Body $body -TimeoutSec 240
    $status = [int]$resp.StatusCode
    $text = $resp.Content
  } catch {
    $status = [int]$_.Exception.Response.StatusCode
    $reader = [System.IO.StreamReader]::new($_.Exception.Response.GetResponseStream(), [System.Text.Encoding]::UTF8)
    $text = $reader.ReadToEnd()
  }
  @($status, $text) | Out-File -Encoding utf8 $outFile
  "saved $outFile (status=$status)"
}

Post-Query @{ query = '反制堆叠有上限吗' } 'scripts\rift_gateway_smoke_200.out.json'
Post-Query @{ query = ' ' }                'scripts\rift_gateway_smoke_400.out.json'
'done'
