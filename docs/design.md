# ygo-rag 设计文档（接口与集成）

> 2026-10-12：当前部署已升级为统一聊天/游戏王/符文 Agent，见 [unified-agent.md](unified-agent.md)。下文原网关拓扑保留为历史设计；生产不再依赖独立 7862 服务。

本文档面向后续开发，说明 ygo-rag 的整体结构、对外接口契约和集成方式。
使用示例见 [README.md](../README.md)；接口行为级规范以 `openspec/specs/` 为准；进行中的改造见 `openspec/changes/`。

## 1. 系统定位与拓扑

ygo-rag 是本机**统一 RAG 服务入口**（默认 `127.0.0.1:7860`），为 kami-man QQ bot 等本地客户端提供全部 RAG 能力。

```text
kami-man bot (QQ)
  ├─ ygo_rag_qa        ──> POST 7860 /api/query      （游戏王相似卡检索）
  ├─ ygo_rag_translate ──> POST 7860 /api/translate  （文本翻译）
  ├─ (agent 客户端)     ──> POST 7860 /api/agent      （统一 Agent 入口）
  └─ rift_rag_qa       ──> POST 7860 /api/rift/query （符文战场规则问答，待实现）
                                │
                                ▼ （HTTP 代理）
                    riftsim rift-rag 服务（127.0.0.1:7862）
```

| 组件 | 仓库 | 默认地址 | 角色 |
|---|---|---|---|
| ygo-rag web 服务 | 本仓库 | `127.0.0.1:7860` | 统一入口：YGO 检索/翻译/裁定/Agent + Rift 网关 |
| riftsim rift-rag | `../riftsim` | `127.0.0.1:7862` | Riftbound 规则 RAG 上游服务（ygo-rag 只是其 HTTP 客户端） |
| kami-man bot | `../kami-man` | — | HTTP 客户端，插件化消费上述接口 |

约定：

- 所有服务只监听 `127.0.0.1`，本机本地优先，不做多用户/鉴权。
- DeepSeek 等密钥**只能**配在服务端环境变量；请求体中出现 `api_key`/`DEEPSEEK_API_KEY`/`authorization` 等键会被 400 拒绝（`/api/translate`、`/api/agent`）。
- 全部文本 I/O 为 UTF-8（见 [AGENTS.md](../AGENTS.md)）。

## 2. 代码结构（`rag_agent/`）

| 模块 | 职责 |
|---|---|
| `__main__.py` | CLI 入口（argparse 子命令） |
| `web.py` | 裸 ASGI 应用：路由、请求解析、JSON 响应；handler 可注入（测试友好） |
| `config.py` | `Settings.from_env()`：全部运行时配置（环境变量 + 默认值） |
| `query_service.py` | `/api/query` 业务：相似卡检索 + `structured` 分块响应 |
| `translation_service.py` | `/api/translate` 业务：DeepSeek 翻译 |
| `adjudication.py` / `adjudication_agent.py` / `adjudication_service.py` | `/api/adjudicate` 业务：裁定（one-shot / LangGraph agent 双路径） |
| `assistant_agent.py` | `/api/agent` 统一 Agent：planner → 工具（search_cards / translate_text / adjudicate / finish） |
| `cards.py` / `db.py` / `card_metadata.py` | cards.cdb 读取与元数据 |
| `retrieval/` | sparse（中文分词 tf） / vector（Chroma + bge-m3） / hybrid（RRF） / reranker（bge-reranker、LLM judge、子进程隔离） |
| `ruling_index.py` | 裁定证据持久索引（ygocdb FAQ + OCG rules EPUB） |
| `structured_query.py` | 查询中的结构化条件（属性/星级/卡片种类）预过滤 |
| `embedding_worker.py` / `rerank_worker.py` | 模型子进程 worker（与网络进程隔离，防 CUDA/FlagEmbedding 冲突） |
| `llm.py` | DeepSeek（langchain-openai）封装 |

> 待实现：`rift_gateway.py`（Riftbound 网关客户端），见 openspec change `add-riftbound-rules-gateway`。

## 3. HTTP API 契约

通用行为（`web.py` 全局）：

- 解析失败/`ValueError` → `400 {"error": str}`；未知路径 → `404 {"error"}`；未预期异常 → `500 {"error": str}`。
- 除 `GET /`（HTML 页面）外均为 `POST` JSON 接口，响应 `application/json; charset=utf-8`。

### 3.1 `POST /api/query` — 相似卡检索

请求：

```json
{
  "query": "有没有效果类似暗黑武装龙的卡",
  "db_path": "data/cards.cdb",
  "top_k": 5,
  "rerank_candidates": 20,
  "semantic": true,
  "rerank": false,
  "llm_rerank": true,
  "llm": false,
  "structured_max_block_chars": 500
}
```

- 必填：`query`（非空）。`top_k` 1..50（默认 10），`rerank_candidates` 1..200（默认 20）。
- `structured_max_block_chars` 可选 1..4000，超长卡片块截断并标 `truncated: true`。

响应：`{answer, results, warnings, structured_query, filter_diagnostics, structured}`。
其中 `structured.blocks[*]` 按卡片分块（`type: "card"`），bot 可直接逐条发送 `blocks[*].text`（shape 见 README「HTTP API / Bot Integration」）。

### 3.2 `POST /api/translate` — 文本翻译

请求：`{text, source_lang = auto, target_lang = zh-CN, structured_max_block_chars?}`。
响应：`{translation, source_lang, target_lang, warnings, structured}`（block `type: "translation"`，长文自动分块）。
凭证不入请求体；DeepSeek 未配置时报错。

### 3.3 `POST /api/adjudicate` — 规则裁定

请求：

```json
{
  "scenario": "对方发动破坏效果时我方怪兽可以代替破坏吗？",
  "db_path": "data/cards.cdb",
  "ygocdb_cache_path": "data/ygocdb",
  "rules_epub_path": "ocg-rule-readthedocs-io-zh-cn-latest.epub",
  "llm": false, "simulate": false, "agent": false, "trace": false
}
```

- `scenario`（或 `query` 别名）必填；`agent: true` 走 LangGraph 多步路径，`trace` 控制是否返回节点/工具轨迹。

响应顶层字段：`scenario, resolved_cards, evidence, reasoning, answer, confidence, warnings, simulation`（agent 路径另有 `agent_coverage`、可选 `trace`）。证据条目带 `kind`/`relationship` 标注，可区分直接 FAQ、`refer` 关联裁定、类推裁定、supplement、卡文本、规则段落、模拟观察。

### 3.4 `POST /api/agent` — 统一 Agent 入口

请求：`{text, db_path?, semantic?, rerank?, llm_rerank?, max_steps = 6 (1..20), ygocdb_cache_path?, rules_epub_path?, specialist_trace?, structured_max_block_chars?, trace?}`。

响应固定：

```json
{
  "answer": "...",
  "warnings": [],
  "exhausted": false,
  "structured": {
    "version": 1,
    "summary": {"tool_call_count": 1, "warning_count": 0, "exhausted": false},
    "blocks": [
      {"type": "tool_result", "index": 1, "tool": "search_cards",
       "ok": true, "summary": "...", "result": {}}
    ]
  }
}
```

仅当请求 `trace: true` 时附带 `trace`。`exhausted: true` 表示达到步数上限，答案基于已有工具结果（bot 按正常回复处理）。

### 3.5 `POST /api/rift/query` — Riftbound 规则问答【待实现】

> 来自 openspec change `add-riftbound-rules-gateway`，实现前先读其 `proposal.md` / `design.md` / `specs/`。

请求：`{query, top_k = 6 (1..20), mode?, trace?}`（`mode ∈ {agent, oneshot}`）。
网关按同契约转发到上游 `{RIFTBOUND_RAG_BASE_URL}/api/query`，`mode`/`trace` 仅在调用方提供时透传。

成功响应（透传上游，字段齐全性由网关校验）：

```json
{
  "answer": "…… [R-CR-716.1]",
  "warnings": [],
  "sources": [{"rule_id": "R-CR-716.1", "topic": "……"}],
  "resolved_cards": [],
  "coverage": {"unresolved_mentions": [], "ambiguous_mentions": []},
  "exhausted": false
}
```

- 必有：`answer`/`warnings`/`sources`；可选 v2 字段（`resolved_cards`、`coverage`、`exhausted`、`mode`、`trace`）上游返回才透传，网关不自行合成。
- 错误语义（取代通用 500）：参数非法 → `400`；上游不可达或非 200 → `502`；上游超时 → `504`；上游 200 但响应体畸形 → `502`。

## 4. CLI 一览（`python -m rag_agent <cmd>`）

| 命令 | 用途 |
|---|---|
| `download-db` / `inspect-db` | 下载 / 检查 cards.cdb |
| `build-index` | 构建 Chroma 向量索引（bge-m3） |
| `query` | 相似卡检索（`--semantic` `--rerank` `--llm` `--llm-rerank`） |
| `translate` | 翻译（`--source-lang` `--target-lang`） |
| `adjudicate` | 裁定（`--agent` `--trace` `--simulate`） |
| `agent` | 统一 Agent 入口（`--max-steps` `--trace`） |
| `download-rulings` / `build-ruling-index` / `inspect-ruling-index` / `query-rulings` | 裁定证据：抓取 ygocdb 缓存、构建/检查/直查持久索引 |
| `web` | 启动 Web 服务（`--host` `--port`，默认 7860） |
| `rift-query`【待实现】 | 查询 Riftbound 规则（`--top-k` `--mode` `--trace`） |

约定：命令异常统一打印 `error: ...` 到 stderr 并退出码 1。

## 5. 运行时配置（环境变量）

`Settings.from_env()` 读取，关键项：

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_DATA_DIR` | `data` | 数据根目录 |
| `RAG_CARDS_DB` | `data/cards.cdb` | 卡片库路径 |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Chroma 索引目录 |
| `RAG_DEVICE` / `RAG_EMBEDDING_DEVICE` / `RAG_RERANKER_DEVICE` | `auto` | 计算设备 |
| `RAG_EMBEDDING_MODEL` / `RAG_RERANKER_MODEL` | `BAAI/bge-m3` / `BAAI/bge-reranker-v2-m3` | 本地模型 |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` / `DEEPSEEK_TIMEOUT_SECONDS` | — / `https://api.deepseek.com` / `deepseek-chat` / `60` | LLM 凭证与超时 |
| `YGOCDB_CACHE_PATH` / `OCG_RULES_EPUB` / `OCG_RULES_INDEX_PATH` | `data/ygocdb` / （epub 文件） / `data/ocg_rules` | 裁定证据来源 |
| `RAG_RULING_*` | 见 `config.py` | 裁定索引与检索参数 |
| `RAG_ADJUDICATION_*` | 见 `config.py` | 裁定 agent 行为参数 |
| `HF_HUB_OFFLINE` | — | 置 `1` 避免每次启动联网检查模型 |
| `RIFTBOUND_RAG_BASE_URL`【待实现】 | `http://127.0.0.1:7862` | rift-rag 上游地址 |
| `RIFTBOUND_RAG_TIMEOUT_SECONDS`【待实现】 | `200` | 上游超时（< bot 240s；非法值回退默认） |

不进 git：`cards.cdb`、Chroma 索引、`data/` 下运行时数据、任何密钥。

## 6. 外部集成契约

### 6.1 kami-man bot 客户端

| bot 侧 | 调用 | 配置 env（默认值） |
|---|---|---|
| `services/ygo_rag_api.py` | `/api/query` | `YGO_RAG_API_URL`（`http://127.0.0.1:7860/api/query`） |
| `services/ygo_rag_agent_api.py` | `/api/agent` | `YGO_RAG_AGENT_API_URL`（`…/api/agent`），超时 240s |
| `services/ygo_rag_translation_api.py` | `/api/translate` | `YGO_RAG_TRANSLATE_API_URL`（`…/api/translate`） |
| `services/rift_rag_api.py` | `{base}/api/query` | `RIFT_RAG_BASE_URL`（现 `http://127.0.0.1:7862`），超时 240s；客户端已容忍解析 v2 可选字段 |

Rift 网关上线后，bot 仅需改环境变量 `RIFT_RAG_BASE_URL=http://127.0.0.1:7860/api/rift`（客户端拼 `{base}/api/query`），**无代码改动**。

### 6.2 riftsim rift-rag 上游

`POST {base}/api/query`：请求 `{query, top_k(1..20)}`，成功 `{answer, warnings, sources[{rule_id, topic}]}`，错误 `400/500 {"error"}`；`GET /healthz` 探活。
上游规划的 v2（agent 模式、`resolved_cards`/`coverage`/`exhausted`）未实现时网关按 legacy 契约透传，将来上游升级后网关与 bot 均无需改动。

超时链约定：bot 240s > ygo-rag 网关 200s > 上游生成 180s。

## 7. 开发约定

- **测试**：直接运行 `python -m pytest`（或 `.\.venv\Scripts\python.exe -m pytest`），不需要 `PYTEST_DISABLE_PLUGIN_AUTOLOAD` 等额外防护（历史说明见 README，已废弃）。
- **测试数据**：测试使用合成 SQLite fixture，不依赖真实 `cards.cdb` / 网络 / LLM。
- **编码**：全仓库 UTF-8，详见 [AGENTS.md](../AGENTS.md)。
- **变更流程**：行为变化走 OpenSpec（`openspec/`）：`openspec-propose` 建 change → 审阅 artifacts → apply → 完成后 archive。架构级决策写进对应 change 的 `design.md`。
- **模式**：
  - Web handler 可注入（`create_app(query_handler=..., ...)`），测试不启动真实服务。
  - 重模型（embedding/rerank）永远在子进程 worker 中运行，与网络进程隔离。
  - 可选重依赖延迟加载；无 DeepSeek 时检索功能仍可用。
