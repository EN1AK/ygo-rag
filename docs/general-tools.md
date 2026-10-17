# Everyday tools for the unified Agent

The existing specialist services remain in place. rag_agent/general_tools.py adds a small,
read-only tool registry consumed by the same planner and tool runner:

| Tool | Purpose |
| --- | --- |
| web_search | Public web search with actual source URLs and publication dates |
| read_webpage | Public page text, 12,000 characters per page with next_offset |
| current_time | Actual current time in an IANA timezone; defaults to Asia/Shanghai |
| calculate | Decimal arithmetic, parentheses and bounded powers; no code execution |

Search and page reading use the official DeepSeek Search/Fetch endpoints:
https://api-docs.deepseek.com/zh-cn/guides/search/
They reuse DEEPSEEK_API_KEY only when DEEPSEEK_BASE_URL identifies api.deepseek.com.
Keys configured for a different model gateway are never forwarded to DeepSeek.
No additional provider, subscription or MCP service is required for these four tools.
DeepSeek documents both web endpoints at CNY 5 per 1,000 successful requests (checked 2026-10-17);
model token charges remain separate.

The planner decides when to search/read without separate QQ commands. Ordinary greetings and
stable explanations can answer directly; fresh information and explicit verification use web tools.
User-provided links can be read and summarized. Arithmetic uses the calculator and current
date/time uses current_time. Existing card text and rulings continue to use specialist evidence.

Only the necessary query or public URL reaches the web provider. The service does not attach
conversation history, session identifiers, archive files or credentials to search payloads.
The URL validator rejects credentials, private/local IP addresses, internal hostnames and
hostnames resolving to private addresses. There are no file-system or shell tools.
Web content is explicitly treated as untrusted evidence rather than tool instructions.
Search/fetch failures are reported rather than replaced with claims that browsing succeeded.

Each Agent request has at most three web attempts within the existing overall deadline and
step budget. Search returns at most five results; fetch responses are size-bounded and long
text is paged. Public results are cached in process for three minutes, up to 64 entries.
Outbound requests are spaced to remain below the provider's documented 5 QPS account limit.
Automatic redirects are disabled for the authenticated API request; errors never echo bodies
that might contain credentials. The response includes available verified source URLs. General web answers with URLs absent from
the actual search/read results are rejected and returned to the planner for verification.
If the step budget expires, a bounded evidence excerpt is returned instead of an unchecked answer.

Kami Man shows basic-tool answers as ordinary chat with concise sources and warnings.
Mixed requests with specialist card tools retain the existing structured card display.
The per-user daily @ quota still counts a user request, not each internal tool call.

Adding later file readers, generic image understanding or MCP-backed adapters should extend
this registry and its argument/observation contract. Do not replace the entire application
with another chat UI merely to obtain tools. Login-only or heavily scripted pages, general
file uploads and arbitrary code execution are outside this first batch.
