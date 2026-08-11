# Yu-Gi-Oh! Card RAG Agent

## Encoding Policy

This repository uses UTF-8 everywhere. Before running PowerShell commands on Windows, load the project UTF-8 setup:

```powershell
.\scripts\utf8.ps1
```

Project rules:

- Text files are UTF-8.
- Python text I/O must pass `encoding="utf-8"` explicitly.
- Python processes should run with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`.
- PowerShell sessions should set console input/output and `$OutputEncoding` to UTF-8.
- Git text handling is defined in `.gitattributes`; `scripts/utf8.ps1` also configures local Git UTF-8 display settings.
- Do not rely on system-default encodings.

本项目用于基于 `cards.cdb` 查询效果相近的游戏王卡片。

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Use Existing CUDA

当前机器已有可用 CUDA PyTorch：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

如果 `.venv` 安装到 CPU 版 PyTorch，可以复用全局环境里的 CUDA 版 torch，而不是重新下载 CUDA wheel：

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch

$src='C:\Users\86136\anaconda3\Lib\site-packages'
$dst='D:\workspace\rag\.venv\Lib\site-packages'
Copy-Item -LiteralPath (Join-Path $src 'torch') -Destination $dst -Recurse -Force
Get-ChildItem -LiteralPath $src -Filter 'torch-2.7.1*dist-info' | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $dst -Recurse -Force }
Get-ChildItem -LiteralPath $src -Filter 'functorch*' | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $dst -Recurse -Force }
Get-ChildItem -LiteralPath $src -Filter 'torchgen*' | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $dst -Recurse -Force }

.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

启用 GPU：

```powershell
$env:RAG_DEVICE="cuda"
$env:HF_HUB_OFFLINE="1"
```

说明：

- Chroma 本身不使用 GPU；GPU 用于 `bge-m3` embedding 和 `bge-reranker-v2-m3` rerank。
- `device=cuda` 时，embedding adapter 会绕开 `sentence-transformers`，直接用 `transformers` + PyTorch CUDA 加载 `bge-m3`，避免全局 Anaconda 环境的 `pyarrow/pandas` 崩溃问题。
- 当 `RAG_DEVICE=cuda` 且启用 `--rerank` 时，reranker 默认在隔离子进程中运行，并使用 `auto/CPU` 设备，避免 `FlagEmbedding` 与 CUDA embedding 同进程触发 Windows `python.exe - 应用程序错误` / access violation。
- 如果确实要实验 GPU reranker，需要显式设置 `$env:RAG_RERANKER_DEVICE="cuda"`；不建议作为默认路径。
- RTX 3060 Laptop 6GB 显存可运行当前小批量构建；全量索引建议从 `--batch-size 10` 或 `--batch-size 16` 开始。

可选环境变量：

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
$env:DEEPSEEK_TIMEOUT_SECONDS="60"
$env:RAG_EMBEDDING_MODEL="BAAI/bge-m3"
$env:RAG_RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
$env:CHROMA_PERSIST_DIR="data/chroma"
```

## Download Database

```powershell
.\.venv\Scripts\python.exe -m rag_agent download-db --url "https://cdn02.moecube.com:444/ygopro-database/zh-CN/cards.cdb"
```

## Inspect Database

```powershell
.\.venv\Scripts\python.exe -m rag_agent inspect-db --db data/cards.cdb
```

## Build Chroma Index

需要本地可加载 `bge-m3`。

```powershell
.\.venv\Scripts\python.exe -m rag_agent build-index --db data/cards.cdb
```

首次运行会从 Hugging Face 下载 `BAAI/bge-m3`。下载完成后，建议启用离线缓存模式，避免每次启动都尝试联网检查：

```powershell
$env:HF_HUB_OFFLINE="1"
```

调试时可以先构建小索引：

```powershell
.\.venv\Scripts\python.exe -m rag_agent build-index --db data/cards.cdb --limit 20 --batch-size 10 --reset
```

GPU 构建示例：

```powershell
$env:RAG_DEVICE="cuda"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -m rag_agent build-index --db data/cards.cdb --batch-size 10 --reset
```

注意：如果 Chroma 索引数量少于 `cards.cdb` 中的卡片数量，`query --semantic` 会跳过 dense retrieval 并输出 warning，避免 partial index 污染结果。需要完整 semantic retrieval 时，必须不带 `--limit` 构建全量索引。

注意：结构化过滤的 dense metadata filter 依赖 Chroma 文档里的新版 metadata（例如 `is_xyz`、`rank`、`decoded_level`）。如果你的索引是在该功能加入前构建的，需要重建：

```powershell
.\.venv\Scripts\python.exe -m rag_agent build-index --db data/cards.cdb --batch-size 10 --reset
```

否则查询仍会 fallback 到旧的 dense 检索后过滤逻辑，并输出 warning。

## Query

离线 sparse baseline：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --db data/cards.cdb
```

启用 Chroma dense retrieval：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --db data/cards.cdb --semantic
```

启用完整 RAG：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --db data/cards.cdb --semantic --rerank --llm
```

使用 DeepSeek 作为 LLM judge/rerank，避免加载本地 reranker：

```powershell
$env:DEEPSEEK_API_KEY="..."
.\.venv\Scripts\python.exe -m rag_agent query "有没有效果类似“我身作盾”的卡" --db data/cards.cdb --semantic --llm-rerank
```

说明：

- `--rerank` 使用本地 `bge-reranker-v2-m3`。
- `--llm-rerank` 使用 DeepSeek 对候选卡做 judge/rerank，不等于最终回答生成。
- `--llm-rerank` 可以不搭配 `--llm` 使用；此时最终输出仍是检索结果格式，但排序来自 LLM judge。
- 同时使用 `--llm-rerank --llm` 时，DeepSeek 会先用于候选排序，再用于最终回答生成，API 延迟和 token 成本都会增加。
- LLM rerank 默认最多评估 20 个候选，可用 `$env:RAG_LLM_RERANK_MAX_CANDIDATES="10"` 调整。
- DeepSeek 调用默认超时 60 秒，可用 `$env:DEEPSEEK_TIMEOUT_SECONDS="30"` 调整。
- 如果 LLM judge 超时、网络失败或返回格式无法解析，查询会回退到 hybrid 候选顺序并输出 warning。

PowerShell 中如果查询文本包含中文弯引号，建议用单引号包裹整个 query：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query '有没有效果类似“我身作盾”的卡' --db data/cards.cdb --rerank
```

## Unified Agent

`agent` 是项目的上层任务入口。它通过中文 Planner prompt 在严格的工具 schema 中选择能力，现有业务实现仍由各自的 service 负责：

- `search_cards` -> `query_service.execute_query()`
- `translate_text` -> `translation_service.execute_translation()`
- `adjudicate` -> `adjudication_service.execute_adjudication()` 的 Agent 路径
- `finish` -> 根据已有工具结果生成最终自然语言回答

单项查卡：

```powershell
.\.venv\Scripts\python.exe -m rag_agent agent '查找效果类似“我身作盾”的卡' --semantic
```

组合任务会在同一个有界 Agent loop 中依次调用多个工具：

```powershell
.\.venv\Scripts\python.exe -m rag_agent agent '查找“我身作盾”，并把这句翻译成中文：発動を無効にし破壊する。' `
  --semantic `
  --trace `
  --progress
```

运行策略由 CLI 或服务端配置控制，Planner 不能自行开启网络、修改数据库路径或传入 API key。`--semantic`、`--rerank` 和 `--llm-rerank` 只影响查卡 specialist。`--max-steps` 限制上层工具轮次，默认 6。

Bot 或其他客户端可以调用统一 API：

```http
POST /api/agent
Content-Type: application/json

{
  "text": "查找我身作盾并翻译它的效果",
  "semantic": true,
  "max_steps": 6,
  "trace": false
}
```

响应固定包含 `answer`、`warnings`、`exhausted` 和 `structured.blocks`。每个 block 保留实际调用的工具及其结构化结果；只有请求 `trace: true` 时才返回上层 Agent trace。原有 `/api/query`、`/api/translate` 和 `/api/adjudicate` 接口继续保留。

## AI Ruling Adjudication

The `adjudicate` command answers concrete ruling scenarios with evidence-grounded output. It keeps this workflow separate from similar-card retrieval:

```powershell
.\.venv\Scripts\python.exe -m rag_agent adjudicate "Can Alpha Dragon destroy Beta Serpent? Can Beta Serpent apply instead?" --db data/cards.cdb
```

Optional evidence sources:

```powershell
.\.venv\Scripts\python.exe -m rag_agent adjudicate "..." `
  --db data/cards.cdb `
  --ygocdb-cache data/ygocdb `
  --rules-epub ocg-rule-readthedocs-io-zh-cn-latest.epub
```

Evidence handling:

- `ygocdb` FAQ data is read from cached API responses shaped like `https://ygocdb.com/api/v0/card/{id}?show=all`.
- `faqs` entries are treated as FAQ ruling evidence and deduplicated by `fid`.
- `faqs[*].refer` builds a reverse index for related-card rulings. These are labeled as related evidence, not direct rulings.
- `supplement` entries are lower-priority card supplement evidence, not concrete FAQ rulings.
- FAQ and supplement HTML is preserved for audit metadata while plain text is used for retrieval and display.
- The local OCG rules EPUB is parsed through `nav.xhtml` or `toc.ncx` into section evidence ids like `ocgrules:{xhtml_path}#{anchor}`.
- Rule passages and ygocdb rulings are complementary. The adjudicator attempts both when available.

The first implementation is document-grounded. `--simulate` is wired through the response shape, but without a concrete `ygoenv` adapter it reports simulator availability/encoding warnings instead of blocking the answer.

LangGraph adjudication agent:

```powershell
.\.venv\Scripts\python.exe -m rag_agent adjudicate "..." `
  --db data/cards.cdb `
  --ygocdb-cache data/ygocdb `
  --rules-epub ocg-rule-readthedocs-io-zh-cn-latest.epub `
  --agent `
  --trace
```

- `--agent` enables the LangGraph path. The default path remains the original one-shot adjudicator.
- The agent keeps explicit state for extracted scenario data, fuzzy card candidates, resolved cards, evidence, coverage, warnings, and final JSON.
- Card resolution tries exact name matching, normalized matching such as `破械冥官·篁` -> `破械冥官 篁`, and sparse card retrieval over card name, text, metadata, and effect tags.
- Evidence tools collect card effects, direct ygocdb FAQ, refer-based related FAQ, supplements, analogous FAQ, and OCG rules passages.
- Coverage checks look for unresolved or ambiguous cards, missing evidence, and hypothetical-branch gaps before final answering. Retrieval is retried up to `RAG_ADJUDICATION_AGENT_MAX_ITERATIONS`.
- `--trace` includes compact node/tool trace data in the JSON output. Without it, the service can still return `agent_coverage` while omitting trace by default.
- Current limitation: ygoenv simulation is not implemented in the agent path yet; the agent is retrieval and evidence grounded.

Agentic card resolution:

- The adjudication agent resolves card names before ruling retrieval: LLM extraction of possible card-name strings, local card candidate retrieval, rule-based high-confidence selection, optional constrained LLM selection for ambiguous candidates, then bounded targeted retries.
- Local card data is the only source of card ids. LLM extraction output is accepted only as strings/context, and LLM selection can choose only ids already returned by local retrieval.
- Malformed LLM extraction or selection output falls back to deterministic extraction/selection and is recorded in the card-resolution trace.
- Ruling FAQ/rules retrieval consumes only the final resolved card ids after retries; unresolved and ambiguous mentions remain visible in coverage and trace summaries.
- Recommended defaults are `RAG_ADJUDICATION_CARD_RESOLUTION_TOP_K=5`, `RAG_ADJUDICATION_CARD_RESOLUTION_RETRY_LIMIT=1`, `RAG_ADJUDICATION_LLM_CARD_EXTRACTION=1`, and `RAG_ADJUDICATION_LLM_CARD_SELECTION=0`. Enable LLM selection only when the card database retrieval candidates are already strong enough to constrain the choice.

Persistent ruling indexes:

```powershell
# 1. Download or refresh local ygocdb payloads first.
.\.venv\Scripts\python.exe -m rag_agent download-rulings `
  --db data/cards.cdb `
  --out data/ygocdb

# 2. Build FAQ/supplement and OCG rules vector indexes from local files.
.\.venv\Scripts\python.exe -m rag_agent build-ruling-index `
  --db data/cards.cdb `
  --ygocdb-cache data/ygocdb `
  --rules-epub ocg-rule-readthedocs-io-zh-cn-latest.epub `
  --reset

# 3. Inspect index manifests and collection counts.
.\.venv\Scripts\python.exe -m rag_agent inspect-ruling-index --json

# 4. Test ruling evidence retrieval directly.
.\.venv\Scripts\python.exe -m rag_agent query-rulings "target was not destroyed later special summon" `
  --db data/cards.cdb `
  --semantic `
  --json
```

- `build-ruling-index --faq-only` updates only ygocdb FAQ/supplement evidence.
- `build-ruling-index --rules-only` updates only OCG rules evidence.
- `query-rulings --rerank` applies the local bge reranker to retrieved evidence.
- `query-rulings --llm-rerank` applies DeepSeek LLM judge rerank and requires `DEEPSEEK_API_KEY`.
- If a persistent ruling index is missing or stale, adjudication keeps the existing runtime fallback from `data/ygocdb` and the EPUB, with warnings/trace metadata where available.
- Persistent indexes are derived artifacts. The raw sources remain `data/ygocdb/*.json`, `data/cards.cdb`, and the configured OCG rules EPUB.

## Web UI

启动本地 Web 页面：

```powershell
$env:RAG_DEVICE="cuda"
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -m rag_agent web --host 127.0.0.1 --port 7860
```

然后打开：

```text
http://127.0.0.1:7860
```

说明：

- 页面默认只监听本机地址 `127.0.0.1`。
- `cards.cdb` 路径、Top K、semantic、本地 rerank、LLM judge rerank、LLM 回答开关可以在页面上调整。
- DeepSeek API Key 不在页面填写，仍然通过环境变量 `DEEPSEEK_API_KEY` 读取。
- 如果勾选 LLM judge rerank 或 LLM 回答，需要当前启动 Web 服务的终端里已经设置好 `DEEPSEEK_API_KEY`。

## HTTP API / Bot Integration

Web 服务启动后，其他服务可以调用：

```text
POST http://127.0.0.1:7860/api/query
```

请求示例：

```json
{
  "query": "有没有效果类似暗黑武装龙且是暗属性的卡",
  "db_path": "data/cards.cdb",
  "top_k": 5,
  "rerank_candidates": 5,
  "semantic": true,
  "rerank": false,
  "llm_rerank": true,
  "llm": false,
  "structured_max_block_chars": 500
}
```

响应会保留原有字段：

```json
{
  "answer": "...",
  "results": [],
  "warnings": []
}
```

同时额外返回适合 QQ bot 按卡片分段发送的 `structured`：

```json
{
  "structured": {
    "version": 1,
    "summary": {
      "result_count": 1,
      "warning_count": 0,
      "warnings": []
    },
    "blocks": [
      {
        "type": "card",
        "index": 1,
        "card_id": 65192027,
        "name": "暗黑武装龙",
        "score": 0.95,
        "text": "1. 暗黑武装龙\nID: 65192027\nScore: 0.9500\n效果: ...\n理由: ...",
        "truncated": false,
        "fields": {
          "card_id": 65192027,
          "name": "暗黑武装龙",
          "score": 0.95,
          "source_text": "...",
          "reason": "..."
        }
      }
    ]
  }
}
```

Bot 可以直接逐条发送：

```text
structured.blocks[*].text
```

如果 QQ 单条消息长度有限，传入 `structured_max_block_chars`，超长卡片块会被截断并标记：

```json
{
  "truncated": true
}
```

## Translation API / Bot Integration

The browser Web UI stays focused on card RAG search, but the same local service also exposes a backend translation endpoint for bots:

```text
POST http://127.0.0.1:7860/api/translate
```

Request example:

```json
{
  "text": "Once per turn: You can target 1 card on the field; destroy it.",
  "source_lang": "en",
  "target_lang": "zh-CN",
  "structured_max_block_chars": 500
}
```

`source_lang` defaults to `auto`, and `target_lang` defaults to `zh-CN`. If a user prompt clearly asks for another direction, pass the requested target language:

```json
{
  "text": "你好",
  "target_lang": "en"
}
```

Response shape:

```json
{
  "translation": "...",
  "source_lang": "en",
  "target_lang": "zh-CN",
  "warnings": [],
  "structured": {
    "version": 1,
    "summary": {
      "block_count": 1,
      "split": false,
      "truncated": false
    },
    "blocks": [
      {
        "type": "translation",
        "index": 1,
        "text": "...",
        "truncated": false,
        "split": false,
        "fields": {
          "source_lang": "en",
          "target_lang": "zh-CN",
          "part": 1,
          "total_parts": 1
        }
      }
    ]
  }
}
```

Bot callers can send:

```text
structured.blocks[*].text
```

DeepSeek credentials are still read from the server environment. Do not send API keys in the request body.

## Adjudication API / Bot Integration

The Web service also exposes adjudication:

```text
POST http://127.0.0.1:7860/api/adjudicate
```

Request example:

```json
{
  "scenario": "Can Alpha Dragon destroy Beta Serpent? Can Beta Serpent apply instead?",
  "db_path": "data/cards.cdb",
  "ygocdb_cache_path": "data/ygocdb",
  "rules_epub_path": "ocg-rule-readthedocs-io-zh-cn-latest.epub",
  "llm": false,
  "simulate": false
}
```

Response fields include `scenario`, `resolved_cards`, `evidence`, `reasoning`, `answer`, `confidence`, `warnings`, and `simulation`. Evidence items are labeled by kind and relationship so callers can distinguish direct ygocdb FAQ rulings, `refer` related rulings, analogous rulings, supplement evidence, card text, rule passages, and simulation observations.

## Translate CLI

The translation service can also be tested from the command line:

```powershell
.\.venv\Scripts\python.exe -m rag_agent translate "Once per turn: You can target 1 card on the field; destroy it."
```

By default this translates to Chinese. Explicit language direction is supported:

```powershell
.\.venv\Scripts\python.exe -m rag_agent translate "你好" --source-lang zh-CN --target-lang en
```

## Structured Filters

查询中包含明确的卡片结构条件时，系统会先按 `cards.cdb` 元数据过滤候选卡，再做 sparse / Chroma / rerank 排序。例如：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query "效果是除外或者回收对手墓地的卡的四星超量怪兽" --db data/cards.cdb --semantic --llm-rerank
```

当前支持的高置信条件包括：

- 属性：`暗属性`、`光属性`、`地属性`、`水属性`、`炎属性`、`风属性`、`神属性`
- 大类：`怪兽`、`魔法`、`陷阱`
- 怪兽类型：`超量/XYZ`、`融合`、`同调/同步`、`连接/LINK`、`仪式`、`效果怪兽`、`通常怪兽`
- 数值：`四星`、`4星`、`四阶`、`4阶`、`Rank 4`、`LINK-2`

例如 `四星超量怪兽` 会按实际需求宽松解释为 `rank=4` 的超量怪兽。CLI 会在命中结构化条件时额外输出 `Structured filters` 诊断，API 也会返回 `structured_query` 和 `filter_diagnostics`。

## Test

当前全局 Anaconda 环境可能自动加载不兼容 pytest 插件，建议禁用插件自动加载：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
```
