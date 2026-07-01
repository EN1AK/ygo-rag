# Yu-Gi-Oh! Card RAG Agent

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

PowerShell 中如果查询文本包含中文弯引号，建议用单引号包裹整个 query：

```powershell
.\.venv\Scripts\python.exe -m rag_agent query '有没有效果类似“我身作盾”的卡' --db data/cards.cdb --rerank
```

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

## Test

当前全局 Anaconda 环境可能自动加载不兼容 pytest 插件，建议禁用插件自动加载：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
```
