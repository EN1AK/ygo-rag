"""Read-only everyday tools. Only explicit search words/URLs reach the official web API."""
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, localcontext, ROUND_FLOOR
import ipaddress
import json
import operator
import re
import socket
import threading
import time
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import requests

GENERAL_TOOLS = frozenset({"web_search", "read_webpage", "current_time", "calculate"})
WEB_TOOLS = frozenset({"web_search", "read_webpage"})
GENERAL_TOOL_SCHEMAS = {
    "web_search": {"description": "搜索公开互联网信息、近期动态和需要核实的通用事实，返回真实来源链接；游戏卡文与裁定仍用专业工具。",
                   "arguments": {"query": "必填，1-128字的必要搜索词，可用 site:/after: 限制来源与日期；不要附带无关聊天历史或凭据", "count": "可选1-5，默认5"}},
    "read_webpage": {"description": "读取用户给出的公开网页或搜索结果链接正文；只支持无需登录的公网 HTTP/HTTPS 页面。",
                     "arguments": {"url": "必填，完整公开URL", "offset": "可选字符起点，默认0；继续阅读使用next_offset"}},
    "current_time": {"description": "查询准确当前日期、时间和星期；默认北京时间，可指定 IANA 时区。",
                     "arguments": {"timezone": "可选，默认Asia/Shanghai，如America/New_York"}},
    "calculate": {"description": "准确计算算术表达式，支持括号、+ - * / // % **；百分数请改为小数，不执行代码。",
                  "arguments": {"expression": "必填，最多300字符的纯数值算术表达式"}},
}
_CACHE = {}
_CACHE_LOCK = threading.Lock()
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0


def public_url(value, *, resolve=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 8192:
        raise ValueError("网页地址无效")
    value = value.strip()
    try:
        parts = urlsplit(value)
        host = (parts.hostname or "").rstrip(".").lower()
        port = parts.port
    except ValueError as exc:
        raise ValueError("网页地址无效") from exc
    if parts.scheme not in {"http", "https"} or not host or parts.username is not None or parts.password is not None or port not in {None, 80, 443}:
        raise ValueError("只允许不含账号密码的公开HTTP/HTTPS网页")
    if host in {"localhost", "metadata.google.internal"} or host.endswith((".localhost", ".local", ".internal", ".home", ".lan")):
        raise ValueError("不允许读取本机或内网地址")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("不允许读取本机或内网地址")
    if resolve and address is None:
        try:
            addresses = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("网页域名暂时无法解析") from exc
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise ValueError("不允许读取解析到内网的地址")
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", parts.query, ""))


def validate_general_arguments(tool, arguments):
    allowed = {"web_search": {"query", "count"}, "read_webpage": {"url", "offset"},
               "current_time": {"timezone"}, "calculate": {"expression"}}[tool]
    if set(arguments) - allowed:
        raise ValueError("基础工具包含未知参数")
    if tool == "web_search":
        query, count = arguments.get("query"), arguments.get("count", 5)
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 128:
            raise ValueError("搜索词须为1-128字")
        if type(count) is not int or not 1 <= count <= 5:
            raise ValueError("搜索条数须为1-5")
        return {"query": query.strip(), "count": count}
    if tool == "read_webpage":
        offset = arguments.get("offset", 0)
        if type(offset) is not int or not 0 <= offset <= 200000:
            raise ValueError("网页阅读起点无效")
        return {"url": public_url(arguments.get("url")), "offset": offset}
    if tool == "current_time":
        zone = arguments.get("timezone", "Asia/Shanghai")
        if not isinstance(zone, str) or len(zone) > 80:
            raise ValueError("时区无效")
        try: ZoneInfo(zone)
        except (ZoneInfoNotFoundError, ValueError) as exc: raise ValueError("请提供有效IANA时区") from exc
        return {"timezone": zone}
    expression = arguments.get("expression")
    if not isinstance(expression, str) or not 1 <= len(expression.strip()) <= 300:
        raise ValueError("算式须为1-300字符")
    return {"expression": expression.strip()}


def calculate(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > 60:
            raise ValueError("算式过于复杂")
        operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                     ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
        def evaluate(node):
            if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
                number = Decimal(ast.get_source_segment(expression, node).replace("_", ""))
            elif isinstance(node, ast.UnaryOp) and type(node.op) in {ast.UAdd, ast.USub}:
                number = evaluate(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            elif isinstance(node, ast.BinOp) and type(node.op) in operators:
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Pow) and (right != right.to_integral_value() or abs(right) > 50):
                    raise ValueError("幂指数须为-50到50的整数")
                if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                    quotient = (left / right).to_integral_value(rounding=ROUND_FLOOR)
                    number = quotient if isinstance(node.op, ast.FloorDiv) else left - quotient * right
                else:
                    number = operators[type(node.op)](left, right)
            else:
                raise ValueError("只支持纯数值算术，不支持函数或代码")
            if not number.is_finite() or abs(number) > Decimal("1e100"):
                raise ValueError("数值超出计算范围")
            return number
        with localcontext() as context:
            context.prec = 40
            value = evaluate(tree.body)
        result = format(value, "f")
        if "." in result: result = result.rstrip("0").rstrip(".")
        return result or "0"
    except (ArithmeticError, SyntaxError, TypeError, RecursionError) as exc:
        raise ValueError("算式无效、除数为零或数值超出范围") from exc


def _api(settings, operation, payload):
    if urlsplit(settings.deepseek_base_url).hostname != "api.deepseek.com" or not settings.deepseek_api_key:
        raise ValueError("网页工具需要配置官方DeepSeek账号；当前模型接口不能自动复用到其他供应商")
    key = (operation, json.dumps(payload, sort_keys=True, ensure_ascii=False))
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry and entry[0] > time.monotonic():
            return deepcopy(entry[1])
    from rag_agent.deadline import remaining
    global _NEXT_REQUEST
    with _RATE_LOCK:
        delay = max(0.0, _NEXT_REQUEST - time.monotonic())
        if delay >= remaining(30): raise TimeoutError("联网请求预算不足")
        if delay: time.sleep(delay)
        _NEXT_REQUEST = time.monotonic() + 0.26
    try:
        with requests.post("https://api.deepseek.com/v1/tools/" + operation,
                           headers={"Authorization": "Bearer " + settings.deepseek_api_key},
                           json=payload, timeout=remaining(30), allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                messages = {401: "联网凭据无效", 402: "联网账号余额不足", 403: "网页读取被安全策略拒绝",
                            404: "网页无法读取或无可提取正文", 429: "联网服务繁忙"}
                raise ValueError(messages.get(response.status_code, "联网服务暂不可用"))
            body = bytearray()
            for chunk in response.iter_content(65536):
                remaining()
                body.extend(chunk)
                if len(body) > 2 * 1024 * 1024: raise ValueError("网页响应过大，请缩小查询范围")
            data = json.loads(body.decode("utf-8"))
    except requests.RequestException as exc:
        raise ValueError("联网请求超时或连接失败") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("联网服务返回格式错误") from exc
    if not isinstance(data, dict): raise ValueError("联网服务返回格式错误")
    with _CACHE_LOCK:
        if len(_CACHE) >= 64: _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = (time.monotonic() + 180, deepcopy(data))
    return data


def run_general_tool(tool, arguments, settings):
    args = validate_general_arguments(tool, arguments)
    if tool == "calculate":
        result = calculate(args["expression"])
        return {"expression": args["expression"], "result": result, "answer": args["expression"] + " = " + result}
    if tool == "current_time":
        now = datetime.now(ZoneInfo(args["timezone"]))
        return {"timezone": args["timezone"], "datetime": now.isoformat(timespec="seconds"),
                "answer": now.strftime("%Y-%m-%d %H:%M:%S") + " 星期" + "一二三四五六日"[now.weekday()] + "（" + args["timezone"] + "）"}
    retrieved = datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")
    if tool == "web_search":
        raw = _api(settings, "search", {"query": args["query"], "count": args["count"], "need_content": False})
        if not isinstance(raw.get("results"), list): raise ValueError("搜索返回格式错误")
        results = []
        for item in raw["results"][:args["count"]]:
            if not isinstance(item, dict): continue
            try: url = public_url(item.get("url"))
            except ValueError: continue
            results.append({"title": str(item.get("title") or url)[:200], "url": url,
                            "summary": str(item.get("summary") or item.get("snippet") or "")[:1200],
                            "published_at": str(item.get("date") or "")[:80]})
        return {"query": args["query"], "results": results, "sources": [{"title": r["title"], "url": r["url"]} for r in results],
                "retrieved_at": retrieved, "answer": "\n\n".join(r["title"] + "\n" + r["url"] + "\n" + r["summary"] for r in results) or "未找到可用的网页结果。"}
    url = public_url(args["url"], resolve=True)
    raw = _api(settings, "fetch", {"url": url})
    if not isinstance(raw.get("content"), str) or not raw["content"].strip():
        raise ValueError("网页没有可读取的正文")
    full = raw["content"][:200000]
    start, end = args["offset"], args["offset"] + 12000
    if start >= len(full): raise ValueError("阅读起点超出网页正文")
    content = full[start:end]
    title = str(raw.get("title") or url)[:200]
    return {"title": title, "url": url, "content": content, "answer": content,
            "sources": [{"title": title, "url": url}], "retrieved_at": retrieved, "offset": start,
            "next_offset": end if end < len(full) else None, "truncated": len(raw["content"]) > end}


def verified_sources(observations):
    sources = {}
    ordered = sorted(observations, key=lambda o: o.tool != "read_webpage")
    for observation in ordered:
        if observation.ok and observation.tool in WEB_TOOLS:
            for source in observation.data.get("sources", []):
                try: url = public_url(source.get("url"))
                except (ValueError, AttributeError): continue
                sources.setdefault(url, str(source.get("title") or url)[:120])
    return sources


def citation_feedback(answer, observations):
    successful = [item for item in observations if item.ok]
    if not any(item.tool in WEB_TOOLS for item in successful) or any(item.tool not in GENERAL_TOOLS for item in successful):
        return ""
    known = {url.rstrip("/") for url in verified_sources(observations)}
    unknown = []
    for match in re.findall(r"https?://[^\s<>\]\)\"']+", answer):
        url = match.rstrip(".,;:!。，；：！）")
        try: normalized = public_url(url).rstrip("/")
        except ValueError: normalized = ""
        if normalized not in known and url not in unknown:
            unknown.append(url)
    if unknown:
        return ("回答包含本次未核实的链接。请先 read_webpage 读取这些公开网页，或删除未经核实的引用后只依据已有结果作答；"
                "不得声称已读取未取得的页面：" + " ".join(unknown[:3]))
    return ""


def attach_sources(answer, observations):
    sources = verified_sources(observations)
    if sources and not any(url in answer for url in sources):
        fetched = {source.get("url") for item in observations if item.ok and item.tool == "read_webpage"
                   for source in item.data.get("sources", []) if isinstance(source, dict)}
        selected = [(url, title) for url, title in sources.items() if not fetched or url in fetched][:3]
        answer += "\n\n参考来源：\n" + "\n".join(title + "\n" + url for url, title in selected)
    return answer
