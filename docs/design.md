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

### 日文卡图翻译 `POST /api/card-translate`

请求：`{"images": ["https://.../card.jpg"]}`，支持一张公网 HTTP(S) 卡图 URL 或 JPEG/PNG/GIF/WebP Base64 data URL（最大 10 MiB）。
响应沿用 `/api/translate` 的 `translation` / `structured` 契约，返回中文卡名、效果、
独立的灵摆效果，并标注机器翻译。Bot 从 QQ 图片域名下载卡图后传入 Base64，避免模型无法访问临时链接；RAG 服务不下载图片或读取本地路径。
缺少图片、非法 URL 或客户端传入密钥返回 400；非卡片或卡名/效果不完整、无法辨认返回 422；
模型异常沿用 500/504。Bot 对失败请求退回每日次数，成功结果不写入聊天会话历史。

沿用后端 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`，图像模型使用独立环境变量
`DEEPSEEK_VISION_MODEL`（默认 `deepseek-flash`）。普通聊天的 `DEEPSEEK_MODEL` 不变。
自建代理须支持该图片模型、OpenAI 格式的 `image_url` 和 JSON 输出。
部署时先升级本服务，再升级 `kami-man`；Bot 从 Agent 地址推导此端点。


### 群聊风格接话 `POST /api/group-chat`

仅 group_id=819151070。请求为 group_id、messages（1..15 条匿名 speaker/text，
单条最多 500 字符）、style_samples（最多 8 条、单条最多 500 字符）。响应 {reply:string}，
空串表示不接话，最多 100 字符。客户端凭证被拒绝；不使用工具、不写 Agent 会话。
后端沿用服务器 DeepSeek 凭证，GROUP_CHAT_MODEL 默认 deepseek-flash，25 秒模型超时。
所有对话/风格样本放在 user 数据中；固定 system prompt 控制短回复、允许弃答和样本指令隔离。
Bot 实现唯一群白名单、样本留存、3%随机触发、300秒冷却、每日20次调用上限，
配置与计数持久化在独立 SQLite，不占普通 @神人 用户额度。Bot请求超时35秒，
发送前重新检查开关，45秒以上旧结果丢弃。无启用通知、失败不发群消息。


### 自动群聊风格 `POST /api/group-style`

仅 group_id=819151070。请求：previous_summary（最多600字符）、style_samples
（3..8条，每条最多500字符）、statistics（message_count、participant_count、
mean_length、short_message_ratio 数字统计）。首次统计至少200条、后续至少100条，
至少3位成员。返回 {summary:string}，最多600字符，只描述表达习惯、不复述私事。
不使用工具、不写聊天会话；凭证只在服务端。使用 GROUP_CHAT_MODEL，模型超时35秒。
Bot本地SQLite持久化每日尝试、租约、源消息水位和总结；最多每日成功更新一次，
失败最多每日3次，跨重启保留。停采/清空使在途租约失效；清空删除风格，30天过期。
`/api/group-chat` 增加可选 style_profile（最多600字符），放在user数据中作语气参考，
不能覆盖固定system规则。旧客户端可省略此字段。


### 场景表达学习与群聊预算

`POST /api/group-learning`：仅 group_id=819151070，samples 为 3..8 条
`{source_id:"s1".."s8",text:string(2..500)}`，ID 唯一，不携带成员标识。
返回 `{expressions:[{situation,expression,source_id}]}`，最多 5 条，场景 2..60 字符，
表达 2..80 字符且必须为引用样本的连续原文。模型不明确时返回空数组。
客户端再次核验来源及原文后入库；模型超时 35 秒、客户端 45 秒，不调用工具。
`/api/group-chat` 兼容增加可选 expressions（最多 5 条 situation/expression 对）；
expressions 与 style_samples 总计最多 8 项，近期 messages 仍最多 15 条。
接话判断、表达选择与回复生成合为一次 JSON 调用；不影响旧客户端。

`GET /api/group-chat/status` 只返回本月调用数、输入/输出/缓存/推理 token、
费用估算、今日估算、日/月限额、用量未知或在途调用数，不暴露提示词、样本或密钥。
仅监听已有 loopback 服务；预算拒绝返回 429，其他已有接口不使用此群聊预算。
三个群聊模型路由均明确 `extra_body={thinking:{type:"disabled"}}`。

预算在 `GROUP_CHAT_BUDGET_DB`（默认 data/group_chat_budget.db）独立持久化。
`GROUP_CHAT_DAILY_BUDGET_CNY=1`，`GROUP_CHAT_MONTHLY_BUDGET_CNY=10`，0 表示不允许调用。
`GROUP_CHAT_INPUT_CNY_PER_MILLION=2`，`GROUP_CHAT_OUTPUT_CNY_PER_MILLION=8`，
使用 Flash 高峰未缓存价，缓存优惠和空闲优惠不计入费用估算；模型或价格改变时须更新。
调用前按序列化消息 UTF-8 字节数加 1024 的输入上界与 max_tokens 输出上界原子预留，
返回用量后以 input_tokens/output_tokens 结算（输出包含推理 token，不重复收费）。
解析失败前已经结算；调用失败、崩溃或缺少用量保留预留额，防止重试/重启绕过限额。
账本只保留 366 天的数值审计记录，不存提示词或回复，不能当作供应商账单。

Bot 端 SQLite 增量增加表达、来源证据、学习租约和近期机器人回复表。
首次 100 条/3 人，后续 50 条新消息，最多 2 个成功批次与 3 次尝试/北京时间日；
每小时检查、失败至少间隔 1 小时、租约 120 秒；清空/停采使旧租约失效。
学习水位代表该批消息已抽样处理，不代表每条原文都发给模型；每次仍最多 8 条。
表达最多 300 条、30 天，按中文双字片段/英文词重合、证据次数与新鲜度检索最多 5 条。
没有匹配时不强加表达。近期机器人回复只用于 24 小时重复检查，不进入人类学习样本。
部署先后端再 Bot；回滚用部署前源码备份，保留新增数据表与费用账本。


### 原文人格与可选表情分类

`POST /api/group-chat` 兼容增加可选 meme_categories：最多 32 条
`{category,description}`，分类名符合 `[a-z_][a-z0-9_]{0,31}`、不可重复，描述 1..160 字符。
服务器只接收分类文字，不接收图片路径、文件或 URL。模型在原有一次调用中同时判断接话、
输出短文字和可选分类。请求提供分类时响应增加 meme_category（广告分类名或 null），
没有分类的旧客户端仍只返回 `{reply}`。未知/错误类型分类回退 null，违规回复同时禁用配图；
reply 为空且分类有效表示纯表情；两者均空表示不接话。模型输出上限 240 token，继续非思考模式。

人格文件由 GROUP_CHAT_PERSONA_FILE 指定，默认 personas/deepseek-group/persona.txt。
默认路径不存在时保持旧文本群聊；环境变量为空字符串可禁用人格，显式路径不存在则报错。
正文最多 20000 字符，UTF-8 原样载入；完整 source.docx、文本与 provenance.json 保存来源证据。
固定运行协议在原文外规定数据隔离、无工具、隐私和 JSON 约束，不编辑原文内容。
风格总结与历史表达仍置于 user 数据中，学习路由没有读取或改写人格文件。

Bot 只发送本地白名单分类中经过大小、目录边界、符号链接和文件头检查的 PNG/JPEG/GIF/WebP。
发送前重新检查管理员开关、45 秒结果年龄及源消息水位；纯表情同样受这些检查。
同一 SQLite 增量表 group_meme_history 记录分类、SHA256、预留时间与北京时间日期，不存成员信息。
默认 25% 配图放行、5 次尝试/日、30 分钟冷却、7 天图片去重，保留 30 天数值记录。
每次最多一张 4MiB 图片；以 Base64 跨 OneBot 容器发送，不依赖宿主路径共享。
不新增模型/向量库/识图调用，群聊模型费用仍计入已有预算；图片字节不传给 LLM。

### @ 闲聊与专业 Agent 隔离

新增本机端点，不改变 /api/agent 的专业提示词或工具：
- POST /api/mention-route：text(1..8000)、previous_route/quoted_route（空、casual或professional）。
  明确专业关键词直接 professional；问候可本地 casual；其余中立 JSON 分类，confidence>=0.85
  才选 casual。不加载人格、历史、群聊样本或引用原文；异常/预算拒绝/格式错误默认 professional。
- POST /api/mention-chat：session_id、group_id、text、可选 style_profile/expressions/meme_categories。
  session_id 为 qq:<bot>:group:<group>:user:<user>，群号必须一致。参考条数和长度复用现有边界；
  非819151070群不允许携带学习风格。响应 route=casual 时含 reply/meme_category；专业转交仅返回
  route=professional，不写入闲聊历史。专业关键词在读取人格前转交。
- POST /api/mention-clear：session_id。无需模型，串行清空闲聊、专业及旧混合空间，返回 {cleared:true}。

人格原文仍完整读取原文件、不改写；闲聊没有工具，模型可用 needs_professional=true 请求转交。
Bot转交仅发送原始问题及已验证专业引用，不发送人格草稿、闲聊历史或学习样本。
分类和人格使用既有 GROUP_CHAT_MODEL、非思考调用及共享预算，分别记为 mention_route/mention_chat；
专业服务不使用此预算，分类预算不足也不阻断专业处理。

专业历史继续由原 AGENT_SESSION_DB 管理，但新 mention client 使用 base+:professional。
闲聊使用 MENTION_PERSONA_SESSION_DB（默认 data/mention_persona_sessions.db），30分钟TTL、最多8轮。
两者数据库、键和读取路径独立；旧 base 会话不迁移到新专业空间。清空等待在途生成完成再删除。
Bot每个bot/group/user会话串行处理，data/mention_routes.db保存最近输出分支及文本SHA256，
每会话最多60条、总计10000条、7天过期；中立分类最多读取30分钟内的前一分支标签。
只有同一范围内唯一被标记为professional的引用内容才可转交；未知、旧引用或两分支相同内容
出现来源冲突时省略。没有任何人格输出的二次专业改写。

@闲聊可以使用本地表情，但图片不送给LLM；显式mention可在其他群使用固定人格/素材，
不放宽随机学习的原群白名单。原MemeLibrary默认白名单不变，仅mention实例显式允许正整数群号。
所有失败仍按既有个人额度退款；清空和用量命令免费。


### 固定指令分流更新（2026-10-20）

此规则替代上述语义分类和自动升级：@消息以 ck、pl、rb、rbpl、translate、查卡、裁定、翻译、规则、查价、卡价、符文规则/裁定、符文战场规则、卡图翻译等指令开头时走专业Agent，其余文字全部走原人格。英文指令支持 / 前缀及空格、冒号或中文参数。数字卡号、链接、发动等词和前一次分支不会单独触发专业。无需模型分类，人格不能自动升级；接口错误退回额度并提示失败。专业追问也要带指令。卡图翻译、清空对话及用量控制保留，专业与人格历史继续隔离。


### @ 人格长回复与分段发送（2026-10-20）

仅放宽明确@人格回复：按内容决定长短，最多2000字符、1800输出token；原人格文件不改。长于约240字符时优先按句子和段落拆成最多4条，首条生成完成后立即发送，其余间隔0.8至1.6秒，同群人格消息组不互相穿插，表情仅附最后一条。完整回复先校验，每段保存为人格引用来源；一次提问仍计一次额度。代码围栏内容整体发送。这是完整生成后的分段交付，不是流式生成。随机接话仍限100字符，专业链保持原发送方式。验证：335项机器人测试和499项后端测试通过；真实模型287字符经模拟发送拆为两条，没有发送QQ群测试消息。详见deploy/persona-segmented-replies-2026-10-20.md。


### 管理员私聊触发（2026-10-20）

机器人superusers管理员可直接私聊，无需@，普通聊天使用原人格及分段发送，专业指令和卡图翻译保持原链路。私聊会话使用独立private命名空间，不读取群聊风格或表达，不进入群聊采集；清空对话仅清除自己的私聊上下文。表情使用固定图库及独立私聊收件人计数库。群聊状态指令在私聊提示回到已启用群查询。新聊天入口不会放行普通私聊用户；原有高优先级命令保持原行为。验证847项通过。详见deploy/admin-private-chat-2026-10-20.md。


### 私聊爱弥斯人格（2026-10-20）

管理员私聊改用独立的爱弥斯/小爱原文：ygo-rag/personas/wuwa-aemeath/persona.txt；旁边保存source.docx和provenance.json。原文来自shirachino/astrbot-persona-prompts固定版本869d6793b93a756a410f00da44852f8b774e48f1的wuwa-aemeath/小爱人格提示词.docx，提取3481字符。可通过PRIVATE_CHAT_PERSONA_FILE指定私聊文件。群聊保持用户定制的大肥鱼人格，专业Agent不使用人格。小爱使用新私聊历史后缀:persona:aemeath，不读旧:persona历史；清空对话同时清除新旧私聊历史。512项后端测试通过，真实API确认小爱称呼。详见kami-man/deploy/private-aemeath-persona-2026-10-20.md。
