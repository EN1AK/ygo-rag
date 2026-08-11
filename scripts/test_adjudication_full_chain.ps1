param(
    [switch]$Llm,
    [switch]$NoTrace,
    [switch]$RawOutput,
    [switch]$Progress,
    [string]$OutFile = "",
    [string]$Scenario = "",
    [string]$ScenarioPath = "",
    [string]$ExpectedPath = "",
    [string]$CaseDir = "",
    [string]$EmbeddingModel = "C:\Users\86136\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181",
    [string]$EmbeddingDevice = "cuda",
    [int]$TotalTurnLimit = 0,
    [int]$RetrievalLimit = 0,
    [int]$CritiqueLimit = 0,
    [int]$SubmitLimit = 0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

. (Join-Path $PSScriptRoot "utf8.ps1")

$env:RAG_EMBEDDING_MODEL = $EmbeddingModel
$env:RAG_EMBEDDING_DEVICE = $EmbeddingDevice
$env:RAG_EMBEDDING_MAX_LENGTH = "256"
$env:HF_HUB_OFFLINE = "1"
$env:RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT = if ($TotalTurnLimit -gt 0) {
    [string]$TotalTurnLimit
} elseif ($env:RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT) {
    $env:RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT
} else { "24" }
$env:RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT = if ($RetrievalLimit -gt 0) {
    [string]$RetrievalLimit
} elseif ($env:RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT) {
    $env:RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT
} else { "14" }
$env:RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT = if ($CritiqueLimit -gt 0) {
    [string]$CritiqueLimit
} elseif ($env:RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT) {
    $env:RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT
} else { "8" }
$env:RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT = if ($SubmitLimit -gt 0) {
    [string]$SubmitLimit
} elseif ($env:RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT) {
    $env:RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT
} else { "3" }

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$CardsDb = Join-Path $RepoRoot "data\cards.cdb"
$YgocdbCache = Join-Path $RepoRoot "data\ygocdb"
$RulesEpub = Join-Path $RepoRoot "ocg-rule-readthedocs-io-zh-cn-latest.epub"
$DefaultScenarioPath = Join-Path $PSScriptRoot "adjudication_sample_scenario.txt"
$DefaultExpectedPath = Join-Path $PSScriptRoot "adjudication_sample_scenario.expected.json"
$SummaryScript = Join-Path $PSScriptRoot "summarize_adjudication_output.py"

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}
if (-not (Test-Path $CardsDb)) {
    throw "cards.cdb not found: $CardsDb"
}
if (-not (Test-Path $YgocdbCache)) {
    throw "ygocdb cache not found: $YgocdbCache"
}
if (-not (Test-Path $RulesEpub)) {
    throw "rules EPUB not found: $RulesEpub"
}

Write-Host ""
Write-Host "== Inspect ruling indexes =="
& $Python -m rag_agent inspect-ruling-index --json
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

function ConvertTo-WindowsCommandLineArgument {
    param([string]$Value)

    if ($null -eq $Value) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    $Result = '"'
    $Backslashes = 0
    foreach ($Char in $Value.ToCharArray()) {
        if ($Char -eq '\') {
            $Backslashes += 1
            continue
        }
        if ($Char -eq '"') {
            $Result += ('\' * (($Backslashes * 2) + 1))
            $Result += '"'
            $Backslashes = 0
            continue
        }
        if ($Backslashes -gt 0) {
            $Result += ('\' * $Backslashes)
            $Backslashes = 0
        }
        $Result += $Char
    }
    if ($Backslashes -gt 0) {
        $Result += ('\' * ($Backslashes * 2))
    }
    $Result += '"'
    return $Result
}

function Invoke-AdjudicationCase {
    param(
        [string]$CaseScenario,
        [string]$CaseName,
        [string]$CaseOutFile,
        [string]$CaseExpectedPath
    )

    Write-Host ""
    Write-Host "== Case: $CaseName =="
    $Args = @(
        "-m", "rag_agent", "adjudicate", $CaseScenario,
        "--db", $CardsDb,
        "--ygocdb-cache", $YgocdbCache,
        "--rules-epub", $RulesEpub,
        "--agent"
    )

    if ($Llm) {
        if (-not $env:DEEPSEEK_API_KEY) {
            throw "DEEPSEEK_API_KEY is not set. Set it first or run without -Llm."
        }
        $Args += "--llm"
    }
    if (-not $NoTrace) {
        $Args += "--trace"
    }
    if ($Progress) {
        $Args += @("--progress", "--output-format", "json")
    }

    if ($RawOutput) {
        & $Python @Args
        return $LASTEXITCODE
    }

    $StdoutFile = "$CaseOutFile.stdout.tmp"
    $StderrFile = "$CaseOutFile.stderr.tmp"
    $QuotedArgs = ($Args | ForEach-Object { ConvertTo-WindowsCommandLineArgument $_ }) -join " "
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $QuotedArgs `
        -RedirectStandardOutput $StdoutFile `
        -RedirectStandardError $StderrFile `
        -NoNewWindow `
        -Wait `
        -PassThru
    $ExitCode = $Process.ExitCode
    $Stdout = ""
    $Stderr = ""
    if (Test-Path $StdoutFile) {
        $Stdout = Get-Content -LiteralPath $StdoutFile -Encoding UTF8 -Raw
        Remove-Item -LiteralPath $StdoutFile -Force
    }
    if (Test-Path $StderrFile) {
        $Stderr = Get-Content -LiteralPath $StderrFile -Encoding UTF8 -Raw
        Remove-Item -LiteralPath $StderrFile -Force
    }
    if ($Progress) {
        $ProgressFile = [IO.Path]::ChangeExtension($CaseOutFile, ".progress.log")
        $Stderr | Set-Content -LiteralPath $ProgressFile -Encoding UTF8
        Write-Host "Progress log saved to: $ProgressFile"
    }
    (($Stdout, $Stderr) -join "") | Set-Content -LiteralPath $CaseOutFile -Encoding UTF8

    Write-Host "Full output saved to: $CaseOutFile"
    Write-Host "Exit code: $ExitCode"
    if ($CaseExpectedPath -and (Test-Path $CaseExpectedPath)) {
        & $Python $SummaryScript $CaseOutFile --expected $CaseExpectedPath |
            ForEach-Object { Write-Host $_ }
    }
    else {
        & $Python $SummaryScript $CaseOutFile |
            ForEach-Object { Write-Host $_ }
    }
    $SummaryExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) {
        return $ExitCode
    }
    return $SummaryExitCode
}

$OutputDir = Join-Path $RepoRoot "data\full_chain_demo"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Failures = 0

if ($CaseDir) {
    $Cases = Get-ChildItem -LiteralPath $CaseDir -Filter "*.txt" | Sort-Object Name
    if (-not $Cases) {
        throw "No .txt cases found in: $CaseDir"
    }
    foreach ($Case in $Cases) {
        $CaseScenario = (Get-Content -LiteralPath $Case.FullName -Encoding UTF8 -Raw).Trim()
        $CaseName = [IO.Path]::GetFileNameWithoutExtension($Case.Name)
        $CaseOutFile = Join-Path $OutputDir "$CaseName-$Stamp.json"
        $CaseExpectedPath = Join-Path $Case.DirectoryName "$CaseName.expected.json"
        $Result = Invoke-AdjudicationCase $CaseScenario $CaseName $CaseOutFile $CaseExpectedPath
        if ($Result -ne 0) {
            $Failures += 1
        }
    }
}
else {
    if (-not $Scenario) {
        if (-not $ScenarioPath) {
            $ScenarioPath = $DefaultScenarioPath
        }
        if (-not (Test-Path $ScenarioPath)) {
            throw "scenario file not found: $ScenarioPath"
        }
        $Scenario = (Get-Content -LiteralPath $ScenarioPath -Encoding UTF8 -Raw).Trim()
    }
    if (-not $ExpectedPath -and $ScenarioPath -eq $DefaultScenarioPath) {
        $ExpectedPath = $DefaultExpectedPath
    }
    if (-not $OutFile) {
        $OutFile = Join-Path $OutputDir "adjudication-full-chain-$Stamp.json"
    }
    $Result = Invoke-AdjudicationCase $Scenario "single" $OutFile $ExpectedPath
    if ($Result -ne 0) {
        $Failures += 1
    }
}

Write-Host ""
if ($Failures -eq 0) {
    Write-Host "All cases passed."
    exit 0
}
Write-Host "$Failures case(s) failed."
exit 1
