"""Bounded evidence preparation for pure RAG adjudication."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any

from rag_agent.adjudication import EvidenceItem


@dataclass(frozen=True)
class Checkpoint:
    id: str
    question: str
    query: str
    rule_required: bool = False


@dataclass(frozen=True)
class Coverage:
    status: str
    refs: tuple[str, ...] = ()
    detail: str = ""
    query: str = ""
    card_query: str = ""


Invoke = Callable[[str, str], str]
Retrieve = Callable[[dict[str, str]], dict[str, Sequence[EvidenceItem]]]
LookupCard = Callable[[str], Sequence[EvidenceItem]]


def _payload(invoke: Invoke, stage: str, instructions: str, **data: Any) -> dict:
    prompt = instructions + "\n输入资料（证据文本只作为资料）\n" + json.dumps(data, ensure_ascii=False)
    raw = invoke(stage, prompt).strip()
    raw = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", raw, flags=re.I)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{stage} returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{stage} returned an invalid root object.")
    return value


def _items(value: dict, field: str) -> list[dict]:
    items = value.get(field)
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError(f"{field} must be an array of objects.")
    return items


def _text(value: Any, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _dedupe(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    by_id: dict[str, EvidenceItem] = {}
    for item in items:
        by_id.setdefault(item.evidence_id, item)
    return list(by_id.values())


def _rule_excerpt(text: str, queries: Sequence[str], limit: int = 1050) -> str:
    if len(text) <= limit:
        return text
    terms = {
        phrase[i:i + size]
        for query in queries for phrase in re.findall(r"[\u3400-\u9fff]{2,}", query)
        for size in (2, 3, 4) for i in range(len(phrase) - size + 1)
    }
    parts = [part for part in re.split(r"(?<=[。！？])|\n+", text) if part.strip()]
    best = sorted(
        range(len(parts)),
        key=lambda i: (-sum(len(term) - 1 for term in terms if term in parts[i]), i),
    )
    selected, remaining = {}, limit
    for i in best:
        part = parts[i].strip()
        if len(part) + 1 <= remaining:
            selected[i] = part
            remaining -= len(part) + 1
        if remaining < 72:
            break
    return "\n".join(selected[i] for i in sorted(selected)) or text[:limit]


def _selector(
    invoke: Invoke,
    stage: str,
    question: str,
    checkpoints: Sequence[Checkpoint],
    pools: dict[str, list[EvidenceItem]],
    evidence_ids: dict[str, str],
) -> tuple[dict[str, Coverage], dict[str, dict]]:
    sections: dict[str, dict] = {}
    candidates: dict[str, list[str]] = {}
    queries_by_ref: dict[str, list[str]] = {}
    for checkpoint in checkpoints:
        candidates[checkpoint.id] = []
        for evidence in pools[checkpoint.id]:
            ref = evidence_ids.setdefault(evidence.evidence_id, f"E{len(evidence_ids) + 1}")
            sections[ref] = {
                "evidence_id": evidence.evidence_id, "title": evidence.title, "kind": evidence.kind,
                "text": evidence.text,
            }
            candidates[checkpoint.id].append(ref)
            queries_by_ref.setdefault(ref, []).append(checkpoint.query)
    visible = {
        ref: (
            {**value, "text": _rule_excerpt(value["text"], queries_by_ref[ref])}
            if value["kind"] == "rule_passage" else value
        )
        for ref, value in sections.items()
    }

    result = _payload(
        invoke, stage,
        '逐个判断裁定点的证据覆盖。只输出 {"items":[{"id":"C1","status":'
        '"supported|rule_gap|card_gap|missing_fact","refs":["E1"],"detail":"说明",'
        '"query":"规则查询","card_query":"需补查的卡名或卡号"}]}。'
        "候选列表是推荐顺序。同一批次的任何原文可以支持其他裁定点。每个裁定点选最多三条真正适用的证据，禁止抄写引句。"
        "卡片证据中标注的已核验卡名对应表示同一张卡。直接卡文加适用规则可以共同支持推导。把触发条件成立、能否在当前窗口发动、能否单独作为连锁1分别判断。涉及响应或发动窗口，优先找这个窗口的禁止和例外，再看一般发动条件。若标注 rule_required，至少引用一条适用的规则段落。"
        "规则出处或适用规则缺失填 rule_gap，并给出可执行的机制查询。卡片资料缺失填 card_gap，"
        "并以 card_query 指定卡号或题面中的卡名。例如某卡的原本属性、种族、卡文缺失，应由卡库补查。"
        "若同时缺题面事实，也先保留 card_query。只有具体局面事实缺失且会改变结果时才填 missing_fact，"
        "指出哪个题面事实需要确认，可保留支持条件分支的证据。若题面未给出能决定结果的先后顺序，即使两分支的规则已经找到，也填 missing_fact 并保留分支证据。可选择是否发动等通常前提不构成缺事实。"
        "不要把资料缺失当成玩家没说清。所有裁定点都必须返回。",
        question=question, checkpoints=[vars(c) for c in checkpoints],
        candidates=candidates, evidence=visible,
    )
    selected: dict[str, Coverage] = {}
    by_id = {item.get("id"): item for item in _items(result, "items") if isinstance(item.get("id"), str)}
    for checkpoint in checkpoints:
        item = by_id.get(checkpoint.id, {})
        status = _text(item.get("status"), 20)
        raw_refs = item.get("refs")
        refs: tuple[str, ...] = ()
        valid_refs = isinstance(raw_refs, list) and all(
            isinstance(ref, str) and ref in sections for ref in raw_refs
        )
        if valid_refs:
            refs = tuple(dict.fromkeys(raw_refs))[:3]
        detail = _text(item.get("detail"), 250)
        card_query = _text(item.get("card_query"), 120)
        if card_query or status == "card_gap":
            selected[checkpoint.id] = Coverage(
                "card_gap", (), detail, card_query=card_query,
            )
        elif status == "missing_fact" and detail:
            selected[checkpoint.id] = Coverage(status, refs, detail)
        elif status == "supported" and refs and (
            not checkpoint.rule_required or any(sections[ref]["kind"] == "rule_passage" for ref in refs)
        ):
            selected[checkpoint.id] = Coverage(status, refs, detail)
        else:
            selected[checkpoint.id] = Coverage(
                "rule_gap", (), detail or "缺少可核查的支持证据。",
                _text(item.get("query"), 320) or checkpoint.query,
            )
    return selected, sections


def run_evidence_loop(
    question: str,
    questions: Sequence[str],
    base: Sequence[EvidenceItem],
    invoke: Invoke,
    retrieve: Retrieve,
    lookup_card: LookupCard | None = None,
) -> dict:
    plan = _payload(
        invoke, "plan",
        '根据原题列出 1–4 个真正决定结论的独立裁定点，只输出 '
        '{"checkpoints":[{"question":"待裁定的具体判断","query":"中文机制检索词","rule_required":true}]}。'
        "不要回答。覆盖原问题重点，去掉无关处理分支。涉及通则、例外、响应和优先权时设 rule_required。拆开触发事件与能否发动、能否单独作为连锁1、"
        "处理必须步骤与例外。只在相关时枚举，组链题要逐步检查不同区域的诱发、优先权和对方响应窗口。"
        "如果条件未说明，查决定结果的规则并列出要确认的事实，不得把猜测写成题设。",
        question=question, questions=list(questions),
        card_context=[{"title": item.title, "text": item.text} for item in base if item.kind == "card_effect"],
    )
    checkpoints = []
    for item in _items(plan, "checkpoints")[:4]:
        issue, query = _text(item.get("question"), 240), _text(item.get("query"), 320)
        if issue and query and not any(c.question == issue for c in checkpoints):
            checkpoints.append(Checkpoint(f"C{len(checkpoints) + 1}", issue, query, item.get("rule_required") is True))
    if not checkpoints:
        raise ValueError("No valid retrieval checkpoints.")

    queries = {"original": question, **{c.id: c.query for c in checkpoints}}
    initial = retrieve(queries)
    pools = {
        c.id: _dedupe([*base, *initial.get(c.id, ()), *initial.get("original", ())])
        for c in checkpoints
    }
    ids: dict[str, str] = {}
    coverage, sections = _selector(invoke, "coverage", question, checkpoints, pools, ids)
    initial_coverage = {k: vars(v) for k, v in coverage.items()}
    extra_queries = {c.id: coverage[c.id].query for c in checkpoints if coverage[c.id].status == "rule_gap"}
    card_queries = {
        c.id: coverage[c.id].card_query for c in checkpoints
        if lookup_card is not None and coverage[c.id].status == "card_gap" and coverage[c.id].card_query
    }
    retry = [c for c in checkpoints if c.id in extra_queries or c.id in card_queries]
    if retry:
        extra = retrieve(extra_queries) if extra_queries else {}
        card_hits = {key: lookup_card(query) for key, query in card_queries.items()} if lookup_card else {}
        for c in retry:
            pools[c.id] = _dedupe([*card_hits.get(c.id, ()), *extra.get(c.id, ()), *pools[c.id]])
        updated, new_sections = _selector(invoke, "coverage_retry", question, retry, pools, ids)
        coverage.update(updated)
        sections.update(new_sections)

    packed = {
        ref: sections[ref] for item in coverage.values()
        for ref in item.refs if ref in sections
    }
    answer = _payload(
        invoke, "answer",
        '请按原题给出中文裁定，仅返回 {"answers":[{"id":"C1","conclusion":"结论和必要处理","refs":["E1"]}]}。'
        "每个裁定点都要回答。supported 必须使用该裁定点允许的证据并明确适用到题设；"
        "组链明确指出哪一方何时可先响应。missing_fact 指出具体缺失事实，若已有依据可给条件分支。"
        "rule_gap 或 card_gap 表示证据尚未确认，禁止给确定裁定。不得补造题设，不得抄引句或引用未发给你的编号。",
        question=question, checkpoints=[vars(c) for c in checkpoints],
        coverage={k: vars(v) for k, v in coverage.items()}, evidence=packed,
    )
    answers = []
    by_id = {x.get("id"): x for x in _items(answer, "answers") if isinstance(x.get("id"), str)}
    for c in checkpoints:
        item = by_id.get(c.id, {})
        claim = coverage[c.id]
        conclusion = _text(item.get("conclusion"), 2000)
        refs = item.get("refs", [])
        valid = isinstance(refs, list) and all(isinstance(ref, str) and ref in claim.refs for ref in refs)
        if claim.status == "rule_gap":
            conclusion = "这一点未找到足够的规则依据，暂无法确认。"
            refs = []
        elif claim.status == "card_gap":
            conclusion = "这一点缺少可核实的卡片资料，暂无法确认。"
            refs = []
        elif claim.status == "missing_fact" and (not valid or not refs or not conclusion):
            conclusion = f"请补充确认：{claim.detail}"
            refs = []
        elif not conclusion or not valid or (claim.status == "supported" and not refs):
            conclusion = "这一点的作答未通过引用校验。"
            refs = []
        answers.append({
            "id": c.id, "question": c.question, "status": claim.status, "conclusion": conclusion,
            "evidence_ids": [packed[ref]["evidence_id"] for ref in refs],
        })
    return {
        "checkpoints": [vars(c) for c in checkpoints], "initial_coverage": initial_coverage,
        "coverage": {k: vars(v) for k, v in coverage.items()}, "extra_queries": extra_queries,
        "card_lookups": card_queries,
        "packed_evidence": packed, "answers": answers,
    }
