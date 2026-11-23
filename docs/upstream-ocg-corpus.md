# 上游 OCG 裁定语料

当前版本由 `build/reference-rag-corpora/current.json` 记录。数据来源是上游项目在 GitHub 发布的整组 OCG 静态文件。基准版本是来源修订 33215，提交 `41c6b8966263796b6548cc6b368c6bc06b04dc70`。

| 文件 | 基准数量 | 用途 |
|---|---:|---|
| cards.json | 14,505 张卡 | 卡名、别名、卡文与属性等 |
| rulings.json | 40,624 条 | 卡片裁定、FAQ 等记录 |
| qa-index.json | 37,476 条 | 上游结构化官方 Q&A 索引 |
| qa-discovery-index.json | 14,422 张卡、11,443 个 Q&A ID | 卡片与问答的关联；计数与正文索引口径不同 |
| ocg-rule-corpus.json | 41 篇文档 | OCG Rule 规则正文，检索时按段落拆分 |
| official-responses.json | 1 条 | 单列的事务局答复，保留来源与确认状态 |
| evidence-index.json.gz | 1 个聚合索引 | 供上游检索使用的预编译证据索引 |
| rag-data-revision-manifest.json、snapshot-meta.json | 2 份元数据 | 文件哈希、来源修订和同步状态 |

规则书及法规资料可能通过聚合索引进入证据，聚合索引和各源文件有重叠，不应把条目数加总成独立裁定数。KONAMI 官方数据库是裁定的一手来源，YGOResources 是上游快照的结构化来源。上游的托管云检索、独立向量文件、TCG 资源、本项目其他检索线和模拟资料不在这套 OCG 本地语料里。

在线 QA 指问答处理过程中，在已有卡片或问答线索后请求 YGOResources 的公开卡片或 Q&A 数据，运行时取得内容。固定快照更新则拉取版本已固定的 GitHub 发布文件，答题时直接读取本地版本。

在 home 的 `D:\workspace\rag`，用仓库 WSL Python 运行 `scripts/sync_upstream_ocg_corpus.py status` 可看当前版本，`refresh` 可检查并更新。脚本校验同一 Git 提交中的九个文件、语料清单的 SHA-256 和记录数量，完整通过后才更新版本指针，旧快照保留。历史评测依照其原始版本号复现。定期同步只访问公开 GitHub 上游发布文件。

从 scripts/reference_rag.mjs 启动的新评测默认读取当前版本。可设置 UPSTREAM_OCG_PINNED_CORPUS=1 回到初始快照。续跑按原计划记录的版本读取；参考 loop 按首轮证据的版本补查。

## Windows 定时更新

由 home 的 Windows 任务计划程序运行 `OCG-Upstream-Corpus-Refresh`，每周一 09:00（中国标准时间）。任务使用同一 Windows 账户的后台登录，执行 `scripts/run_upstream_ocg_corpus_refresh.ps1`，再调用 WSL 中的同步程序。错过运行时间后会尽快补跑，同时只运行一个实例。日志写入 `build/reference-rag-corpora/scheduled-refresh.log`。

计划任务配置可用 `scripts/register_upstream_ocg_corpus_task.ps1` 重新注册。可用 `Start-ScheduledTask -TaskName OCG-Upstream-Corpus-Refresh` 手动触发，用 `Get-ScheduledTaskInfo -TaskName OCG-Upstream-Corpus-Refresh` 查看最近及下次运行时间。

## 部署前置文件

这些入口依赖已部署的上游固定代码和初始语料。仓库只版本化同步和运行脚本。新的工作区需要先按 [上游固定提交](https://github.com/coldiceh/ocg-ruling-assistant/tree/41c6b8966263796b6548cc6b368c6bc06b04dc70) 恢复 `build/reference-rag-pins` 中的 `backend` 及其依赖、Git 树清单 `tree.json`、代码清单 `code-files.txt`、固定题单（仅评测需要），并将提交中的表列九个文件放进 `verified-data`。随后用 `scripts/sync_upstream_ocg_corpus.py init` 校验并建立版本指针，再注册 Windows 任务。语料快照、凭据和运行日志由部署主机保存。
