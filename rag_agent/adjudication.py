from __future__ import annotations

import json
import math
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import unquote, urldefrag
from xml.etree import ElementTree

from rag_agent.card_metadata import decode_effect_category_labels
from rag_agent.cards import Card, normalize_effect_text
from rag_agent.retrieval.sparse import tokenize_text


class Llm(Protocol):
    def invoke(self, prompt: str):
        ...


@dataclass(frozen=True)
class AdjudicationProgressEvent:
    stage: str
    message: str
    tool: str = ""
    ok: bool | None = None
    evidence_count: int | None = None
    sufficient: bool | None = None
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "message": self.message,
            "tool": self.tool,
            "ok": self.ok,
            "evidence_count": self.evidence_count,
            "sufficient": self.sufficient,
            "fallback": self.fallback,
        }


AdjudicationProgressCallback = Callable[[AdjudicationProgressEvent], None]


@dataclass(frozen=True)
class ParsedScenario:
    original_text: str
    card_mentions: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    effect_actions: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    replacement_candidates: tuple[str, ...] = ()
    questions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "card_mentions": list(self.card_mentions),
            "roles": list(self.roles),
            "effect_actions": list(self.effect_actions),
            "objects": list(self.objects),
            "replacement_candidates": list(self.replacement_candidates),
            "questions": list(self.questions),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ResolvedCard:
    mention: str
    matches: tuple[Card, ...]
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "ambiguous": self.ambiguous,
            "matches": [
                {
                    "card_id": card.card_id,
                    "name": card.name,
                    "effect_text": card.description,
                }
                for card in self.matches
            ],
        }


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    source: str
    text: str
    relationship: str
    title: str = ""
    card_ids: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "relationship": self.relationship,
            "card_ids": list(self.card_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReasoningStep:
    text: str
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}


@dataclass(frozen=True)
class SimulationStatus:
    status: str
    observations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "observations": list(self.observations),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class AdjudicationResponse:
    scenario: ParsedScenario
    resolved_cards: tuple[ResolvedCard, ...]
    evidence: tuple[EvidenceItem, ...]
    reasoning: tuple[ReasoningStep, ...]
    answer: str
    confidence: str
    warnings: tuple[str, ...]
    simulation: SimulationStatus
    agent_coverage: Mapping[str, Any] | None = None
    agent_trace: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "scenario": self.scenario.to_dict(),
            "resolved_cards": [card.to_dict() for card in self.resolved_cards],
            "evidence": [item.to_dict() for item in self.evidence],
            "reasoning": [step.to_dict() for step in self.reasoning],
            "answer": self.answer,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "simulation": self.simulation.to_dict(),
        }
        if self.agent_coverage is not None:
            payload["agent_coverage"] = dict(self.agent_coverage)
        if self.agent_trace:
            payload["agent_trace"] = [dict(entry) for entry in self.agent_trace]
        return payload


def derive_adjudication_source_status(
    evidence: Sequence[EvidenceItem],
    *,
    faq_searched: bool,
    rules_searched: bool,
    sufficient: bool,
) -> dict[str, str]:
    faq_found = any(item.kind == "faq" for item in evidence)
    rules_found = any(item.kind == "rule_passage" for item in evidence)
    faq_status = "found" if faq_found else "not_found" if faq_searched else "not_searched"
    rules_status = (
        "found" if rules_found else "not_found" if rules_searched else "not_searched"
    )
    failure_reason = ""
    if rules_status == "not_found" and faq_found:
        failure_reason = (
            "已依据 FAQ 证据完成裁定，但未找到对应的规则文档机制。"
            if sufficient
            else "现有 FAQ 证据不足，且未找到对应的规则文档机制。"
        )
    elif not sufficient and faq_status == "not_found" and rules_status == "not_found":
        failure_reason = "未找到 FAQ 和规则文档证据，当前证据不足。"
    elif not sufficient and rules_status == "not_found":
        failure_reason = "未找到对应的规则文档机制，当前证据不足。"
    elif not sufficient and faq_status == "not_found":
        failure_reason = "未找到 FAQ 证据，当前证据覆盖不足。"
    elif not sufficient:
        failure_reason = "当前证据覆盖不足，无法给出高置信度裁定。"
    return {
        "faq": faq_status,
        "rules": rules_status,
        "failure_reason": failure_reason,
    }


def normalize_adjudication_answer_source_status(
    answer: str,
    evidence: Sequence[EvidenceItem],
    *,
    faq_searched: bool,
    rules_searched: bool,
    sufficient: bool,
) -> str:
    try:
        raw_payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM adjudication answer must be valid JSON.") from exc
    if not isinstance(raw_payload, Mapping):
        raise ValueError("LLM adjudication answer must be a JSON object.")
    payload = dict(raw_payload)
    payload["source_status"] = derive_adjudication_source_status(
        evidence,
        faq_searched=faq_searched,
        rules_searched=rules_searched,
        sufficient=sufficient,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class YgocdbFaqDocument:
    fid: str
    primary_card_ids: tuple[int, ...]
    refer_card_ids: tuple[int, ...]
    html_refer_card_ids: tuple[int, ...]
    title_html: str
    question_html: str
    answer_html: str
    title_plain: str
    question_plain: str
    answer_plain: str
    date: str = ""
    source_url: str = ""
    snapshot_at: str = ""

    @property
    def ruling_id(self) -> str:
        return f"ygocdb:faq:{self.fid}"

    @property
    def searchable_text(self) -> str:
        return "\n".join(
            part
            for part in [self.title_plain, self.question_plain, self.answer_plain]
            if part
        )

    def to_evidence(self, relationship: str) -> EvidenceItem:
        card_ids = tuple(
            sorted(set(self.primary_card_ids).union(self.refer_card_ids))
        )
        return EvidenceItem(
            evidence_id=self.ruling_id,
            kind="faq",
            source="ygocdb",
            title=self.title_plain,
            text=self.searchable_text,
            relationship=relationship,
            card_ids=card_ids,
            metadata={
                "fid": self.fid,
                "date": self.date,
                "primary_card_ids": list(self.primary_card_ids),
                "refer_card_ids": list(self.refer_card_ids),
                "html_refer_card_ids": list(self.html_refer_card_ids),
                "source_url": self.source_url,
                "snapshot_at": self.snapshot_at,
            },
        )


@dataclass(frozen=True)
class YgocdbSupplementDocument:
    primary_card_id: int
    refer_card_ids: tuple[int, ...]
    html_refer_card_ids: tuple[int, ...]
    text_html: str
    text_plain: str
    date: str = ""
    source_url: str = ""
    snapshot_at: str = ""

    @property
    def supplement_id(self) -> str:
        return f"ygocdb:supplement:{self.primary_card_id}"

    def to_evidence(self, relationship: str = "card_supplement") -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.supplement_id,
            kind="card_supplement",
            source="ygocdb",
            title=f"ygocdb supplement: {self.primary_card_id}",
            text=self.text_plain,
            relationship=relationship,
            card_ids=tuple(
                sorted({self.primary_card_id}.union(self.refer_card_ids))
            ),
            metadata={
                "date": self.date,
                "primary_card_id": self.primary_card_id,
                "refer_card_ids": list(self.refer_card_ids),
                "html_refer_card_ids": list(self.html_refer_card_ids),
                "source_url": self.source_url,
                "snapshot_at": self.snapshot_at,
            },
        )


@dataclass(frozen=True)
class RuleSection:
    rule_id: str
    title: str
    chapter_path: tuple[str, ...]
    xhtml_path: str
    anchor: str
    text: str

    @property
    def chapter_id(self) -> str:
        return self.xhtml_path

    def to_evidence(self, relationship: str = "rule_document") -> EvidenceItem:
        return EvidenceItem(
            evidence_id=self.rule_id,
            kind="rule_passage",
            source="ocgrules",
            title=self.title,
            text=self.text,
            relationship=relationship,
            metadata={
                "chapter_path": list(self.chapter_path),
                "chapter_id": self.chapter_id,
                "xhtml_path": self.xhtml_path,
                "anchor": self.anchor,
            },
        )


@dataclass(frozen=True)
class RuleChapterCandidate:
    chapter_id: str
    title: str
    chapter_path: tuple[str, ...]
    section_titles: tuple[str, ...]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "chapter_path": list(self.chapter_path),
            "section_titles": list(self.section_titles),
            "score": self.score,
        }


@dataclass(frozen=True)
class RuleChapterCatalogEntry:
    chapter_id: str
    title: str
    chapter_path: tuple[str, ...]
    section_titles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chapter_id": self.chapter_id,
            "title": self.title,
            "chapter_path": list(self.chapter_path),
            "section_titles": list(self.section_titles),
        }


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return normalize_effect_text(" ".join(self.parts))


def html_to_plain_text(value: Any) -> str:
    parser = _TextExtractor()
    parser.feed(str(value or ""))
    parser.close()
    return parser.text()


def extract_html_refer_ids(*html_values: str) -> tuple[int, ...]:
    ids: set[int] = set()
    for html in html_values:
        for match in re.findall(r'data-refer=["\']?(\d+)', str(html or "")):
            ids.add(int(match))
    return tuple(sorted(ids))


def _refer_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        return ()
    ids = []
    for key in value.keys():
        try:
            ids.append(int(key))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(ids)))


def _snapshot_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class YgocdbCache:
    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path

    def load_card_payload(self, card_id: int) -> dict[str, Any] | None:
        for path in self._candidate_paths(card_id):
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        if self.cache_path.is_file():
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, Mapping):
                nested = payload.get(str(card_id)) or payload.get(card_id)
                if isinstance(nested, Mapping):
                    return dict(nested)
                if int(payload.get("id", -1)) == card_id:
                    return dict(payload)
        return None

    def fetch_card_payload(
        self,
        card_id: int,
        *,
        timeout: float = 30.0,
        use_proxy: bool = True,
    ) -> dict[str, Any]:
        url = ygocdb_card_url(card_id)
        opener = urllib.request.build_opener() if use_proxy else urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )
        with opener.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def save_card_payload(self, card_id: int, payload: Mapping[str, Any]) -> Path:
        if self.cache_path.suffix:
            raise ValueError("cannot save individual card payloads to a single JSON file cache")
        self.cache_path.mkdir(parents=True, exist_ok=True)
        path = self._candidate_paths(card_id)[0]
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def get_card_payload(
        self,
        card_id: int,
        *,
        allow_fetch: bool = False,
        timeout: float = 30.0,
        use_proxy: bool = True,
    ) -> dict[str, Any] | None:
        payload = self.load_card_payload(card_id)
        if payload is not None or not allow_fetch:
            return payload
        payload = self.fetch_card_payload(card_id, timeout=timeout, use_proxy=use_proxy)
        if not self.cache_path.suffix:
            self.save_card_payload(card_id, payload)
        return payload

    def _candidate_paths(self, card_id: int) -> list[Path]:
        if self.cache_path.suffix:
            return [self.cache_path]
        return [
            self.cache_path / f"{card_id}.json",
            self.cache_path / f"{card_id}_show_all.json",
            self.cache_path / f"card_{card_id}.json",
        ]


def ygocdb_card_url(card_id: int) -> str:
    return f"https://ygocdb.com/api/v0/card/{card_id}?show=all"


class YgocdbRulingIndex:
    def __init__(
        self,
        faqs: Sequence[YgocdbFaqDocument] = (),
        supplements: Sequence[YgocdbSupplementDocument] = (),
    ) -> None:
        self.faq_by_fid: dict[str, YgocdbFaqDocument] = {}
        primary: dict[int, set[str]] = defaultdict(set)
        refer: dict[int, set[str]] = defaultdict(set)
        for faq in faqs:
            self.faq_by_fid[faq.fid] = faq
            for card_id in faq.primary_card_ids:
                primary[card_id].add(faq.fid)
            for card_id in faq.refer_card_ids:
                refer[card_id].add(faq.fid)
        self.faq_ids_by_primary_card_id = {
            card_id: tuple(sorted(ids)) for card_id, ids in primary.items()
        }
        self.faq_ids_by_refer_card_id = {
            card_id: tuple(sorted(ids)) for card_id, ids in refer.items()
        }
        self.supplement_by_card_id = {
            supplement.primary_card_id: supplement for supplement in supplements
        }

    @classmethod
    def from_payloads(
        cls,
        payloads: Sequence[Mapping[str, Any]],
        *,
        snapshot_at: str | None = None,
    ) -> "YgocdbRulingIndex":
        snapshot = snapshot_at or _snapshot_now()
        merged: dict[str, YgocdbFaqDocument] = {}
        supplements: list[YgocdbSupplementDocument] = []
        for payload in payloads:
            card_id = int(payload.get("id") or payload.get("card_id") or 0)
            if card_id <= 0:
                continue
            source_url = ygocdb_card_url(card_id)
            for faq_payload in payload.get("faqs") or ():
                faq = normalize_ygocdb_faq(
                    faq_payload,
                    primary_card_id=card_id,
                    source_url=source_url,
                    snapshot_at=snapshot,
                )
                existing = merged.get(faq.fid)
                merged[faq.fid] = merge_faq_documents(existing, faq) if existing else faq
            supplement_payload = payload.get("supplement")
            if isinstance(supplement_payload, Mapping):
                supplement = normalize_ygocdb_supplement(
                    supplement_payload,
                    primary_card_id=card_id,
                    source_url=source_url,
                    snapshot_at=snapshot,
                )
                if supplement.text_plain:
                    supplements.append(supplement)
        return cls(list(merged.values()), supplements)

    def direct_faqs(self, card_id: int) -> list[YgocdbFaqDocument]:
        return [self.faq_by_fid[fid] for fid in self.faq_ids_by_primary_card_id.get(card_id, ())]

    def related_faqs(self, card_id: int) -> list[YgocdbFaqDocument]:
        return [self.faq_by_fid[fid] for fid in self.faq_ids_by_refer_card_id.get(card_id, ())]

    def supplement(self, card_id: int) -> YgocdbSupplementDocument | None:
        return self.supplement_by_card_id.get(card_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        effect_tags: Sequence[str] = (),
        card_effect_tags_by_id: Mapping[int, Sequence[str]] | None = None,
    ) -> list[YgocdbFaqDocument]:
        query_text = "\n".join([query, " ".join(effect_tags)])
        scored = _score_faq_documents(
            query_text,
            self.faq_by_fid.values(),
            top_k=top_k,
            effect_tags=effect_tags,
            card_effect_tags_by_id=card_effect_tags_by_id or {},
        )
        return [self.faq_by_fid[fid] for fid, _score in scored]


def normalize_ygocdb_faq(
    payload: Mapping[str, Any],
    *,
    primary_card_id: int,
    source_url: str = "",
    snapshot_at: str = "",
) -> YgocdbFaqDocument:
    title_html = str(payload.get("title") or "")
    question_html = str(payload.get("question") or "")
    answer_html = str(payload.get("answer") or "")
    return YgocdbFaqDocument(
        fid=str(payload.get("fid") or ""),
        primary_card_ids=(primary_card_id,),
        refer_card_ids=_refer_ids(payload.get("refer")),
        html_refer_card_ids=extract_html_refer_ids(
            title_html,
            question_html,
            answer_html,
        ),
        title_html=title_html,
        question_html=question_html,
        answer_html=answer_html,
        title_plain=html_to_plain_text(title_html),
        question_plain=html_to_plain_text(question_html),
        answer_plain=html_to_plain_text(answer_html),
        date=str(payload.get("date") or ""),
        source_url=source_url,
        snapshot_at=snapshot_at,
    )


def merge_faq_documents(
    left: YgocdbFaqDocument | None,
    right: YgocdbFaqDocument,
) -> YgocdbFaqDocument:
    if left is None:
        return right
    return YgocdbFaqDocument(
        fid=left.fid,
        primary_card_ids=tuple(sorted(set(left.primary_card_ids).union(right.primary_card_ids))),
        refer_card_ids=tuple(sorted(set(left.refer_card_ids).union(right.refer_card_ids))),
        html_refer_card_ids=tuple(
            sorted(set(left.html_refer_card_ids).union(right.html_refer_card_ids))
        ),
        title_html=left.title_html or right.title_html,
        question_html=left.question_html or right.question_html,
        answer_html=left.answer_html or right.answer_html,
        title_plain=left.title_plain or right.title_plain,
        question_plain=left.question_plain or right.question_plain,
        answer_plain=left.answer_plain or right.answer_plain,
        date=left.date or right.date,
        source_url=left.source_url or right.source_url,
        snapshot_at=left.snapshot_at or right.snapshot_at,
    )


def normalize_ygocdb_supplement(
    payload: Mapping[str, Any],
    *,
    primary_card_id: int,
    source_url: str = "",
    snapshot_at: str = "",
) -> YgocdbSupplementDocument:
    text_html = str(payload.get("text") or "")
    return YgocdbSupplementDocument(
        primary_card_id=primary_card_id,
        refer_card_ids=_refer_ids(payload.get("refer")),
        html_refer_card_ids=extract_html_refer_ids(text_html),
        text_html=text_html,
        text_plain=html_to_plain_text(text_html),
        date=str(payload.get("date") or ""),
        source_url=source_url,
        snapshot_at=snapshot_at,
    )


class RuleIndex:
    def __init__(self, sections: Sequence[RuleSection]) -> None:
        self.sections = list(sections)
        self.section_by_id = {section.rule_id: section for section in self.sections}
        sections_by_chapter: dict[str, list[RuleSection]] = defaultdict(list)
        for section in self.sections:
            sections_by_chapter[section.chapter_id].append(section)
        self.sections_by_chapter = {
            chapter_id: tuple(chapter_sections)
            for chapter_id, chapter_sections in sections_by_chapter.items()
        }

    @classmethod
    def from_epub(cls, epub_path: Path, *, max_section_chars: int = 1600) -> "RuleIndex":
        return cls(parse_ocg_rules_epub(epub_path, max_section_chars=max_section_chars))

    def chapter_catalog(self) -> tuple[RuleChapterCatalogEntry, ...]:
        """Return the complete chapter catalog without section bodies."""
        entries: list[RuleChapterCatalogEntry] = []
        for chapter_id, sections in self.sections_by_chapter.items():
            representative = sections[0]
            chapter_path = _rule_chapter_path(representative)
            entries.append(
                RuleChapterCatalogEntry(
                    chapter_id=chapter_id,
                    title=chapter_path[-1] if chapter_path else representative.title,
                    chapter_path=chapter_path,
                    section_titles=tuple(
                        dict.fromkeys(
                            section.chapter_path[-1]
                            if section.chapter_path
                            else section.title
                            for section in sections
                            if section.title
                        )
                    ),
                )
            )
        return tuple(entries)

    def classify_chapters(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RuleChapterCandidate]:
        query_terms = set(tokenize_text(query))
        if not query_terms:
            return []
        chapter_records: list[
            tuple[str, str, tuple[str, ...], tuple[str, ...], Sequence[RuleSection]]
        ] = []
        chapter_metadata: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
        title_term_frequency: Counter[str] = Counter()
        catalog = {entry.chapter_id: entry for entry in self.chapter_catalog()}
        for chapter_id, sections in self.sections_by_chapter.items():
            entry = catalog[chapter_id]
            chapter_path = entry.chapter_path
            title = entry.title
            section_titles = entry.section_titles
            chapter_metadata[chapter_id] = (title, chapter_path, section_titles)
            title_term_frequency.update(
                set(tokenize_text("\n".join([*chapter_path, *section_titles])))
            )
            chapter_records.append(
                (chapter_id, title, chapter_path, section_titles, sections)
            )
        chapter_count = max(len(chapter_records), 1)
        term_idf = {
            term: math.log((1 + chapter_count) / (1 + frequency)) + 1.0
            for term, frequency in title_term_frequency.items()
        }
        scored_chapters: list[tuple[str, float]] = []
        for chapter_id, _title, chapter_path, section_titles, sections in chapter_records:
            score = _score_rule_chapter(
                query_terms,
                chapter_path,
                section_titles,
                sections,
                term_idf,
            )
            if score > 0:
                scored_chapters.append((chapter_id, score))
        scored_chapters.sort(key=lambda item: (-item[1], item[0]))
        return [
            RuleChapterCandidate(
                chapter_id=chapter_id,
                title=chapter_metadata[chapter_id][0],
                chapter_path=chapter_metadata[chapter_id][1],
                section_titles=chapter_metadata[chapter_id][2],
                score=score,
            )
            for chapter_id, score in scored_chapters[:top_k]
        ]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        chapter_ids: Sequence[str] = (),
    ) -> list[RuleSection]:
        allowed_chapters = set(chapter_ids)
        sections = [
            section
            for section in self.sections
            if not allowed_chapters or section.chapter_id in allowed_chapters
        ]
        scored = _score_documents(
            query,
            [
                (
                    section.rule_id,
                    _weighted_rule_search_text(section),
                )
                for section in sections
            ],
            top_k=top_k,
        )
        return [self.section_by_id[rule_id] for rule_id, _score in scored]


def parse_ocg_rules_epub(
    epub_path: Path,
    *,
    max_section_chars: int = 1600,
) -> list[RuleSection]:
    with zipfile.ZipFile(epub_path) as archive:
        toc_entries = _read_epub_nav_entries(archive) or _read_epub_ncx_entries(archive)
        paths_with_anchors = {
            path
            for path, anchor in (_split_href(href) for _title, href, _chapter in toc_entries)
            if anchor
        }
        sections: list[RuleSection] = []
        for title, href, chapter_path in toc_entries:
            path, anchor = _split_href(href)
            if not path or path.startswith("_") or path not in archive.namelist():
                continue
            if not anchor and path in paths_with_anchors:
                continue
            html = archive.read(path).decode("utf-8", errors="replace")
            text = extract_rule_section_text(html, anchor)
            if not text:
                continue
            base = RuleSection(
                rule_id=_rule_id(path, anchor),
                title=title,
                chapter_path=tuple(chapter_path),
                xhtml_path=path,
                anchor=anchor,
                text=text,
            )
            sections.extend(_split_rule_section(base, max_section_chars=max_section_chars))
        return _dedupe_rule_sections(sections)


def extract_rule_section_text(html: str, anchor: str = "") -> str:
    try:
        root = ElementTree.fromstring(html)
    except ElementTree.ParseError:
        return html_to_plain_text(html)
    target = _find_element_by_anchor(root, anchor) if anchor else None
    if target is None:
        target = _first_descendant(root, "body") or root
    parent_map = _parent_map(root)
    heading = _heading_ancestor(target, parent_map) or (
        target if _heading_level(target) is not None else None
    )
    if heading is not None:
        elements = _heading_section_elements(heading, parent_map)
    elif _local_name(target.tag) in {"section", "article", "div", "body"}:
        elements = [target]
    else:
        elements = [target]
    return _plain_text_from_elements(elements)


def _read_epub_nav_entries(archive: zipfile.ZipFile) -> list[tuple[str, str, tuple[str, ...]]]:
    if "nav.xhtml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("nav.xhtml"))
    entries: list[tuple[str, str, tuple[str, ...]]] = []

    def walk(element, parents: tuple[str, ...] = ()) -> None:
        for child in list(element):
            tag = _local_name(child.tag)
            if tag == "li":
                anchor = _first_child(child, "a")
                title = "".join(anchor.itertext()).strip() if anchor is not None else ""
                href = anchor.attrib.get("href", "") if anchor is not None else ""
                next_parents = parents + ((title,) if title else ())
                if title and href:
                    entries.append((title, href, next_parents))
                for nested in child:
                    if _local_name(nested.tag) == "ol":
                        walk(nested, next_parents)
            elif tag == "ol":
                walk(child, parents)
            else:
                walk(child, parents)

    walk(root)
    return entries


def _read_epub_ncx_entries(archive: zipfile.ZipFile) -> list[tuple[str, str, tuple[str, ...]]]:
    if "toc.ncx" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("toc.ncx"))
    entries: list[tuple[str, str, tuple[str, ...]]] = []

    def walk(nav_point, parents: tuple[str, ...] = ()) -> None:
        label = nav_point.find(".//{*}navLabel/{*}text")
        content = nav_point.find("{*}content")
        title = "".join(label.itertext()).strip() if label is not None else ""
        href = content.attrib.get("src", "") if content is not None else ""
        next_parents = parents + ((title,) if title else ())
        if title and href:
            entries.append((title, href, next_parents))
        for child in nav_point.findall("{*}navPoint"):
            walk(child, next_parents)

    for nav_point in root.findall(".//{*}navMap/{*}navPoint"):
        walk(nav_point)
    return entries


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_element_by_anchor(root, anchor: str):
    if not anchor:
        return None
    for element in root.iter():
        if element.attrib.get("id") == anchor or element.attrib.get("name") == anchor:
            return element
    return None


def _first_descendant(element, local_name: str):
    for descendant in element.iter():
        if _local_name(descendant.tag) == local_name:
            return descendant
    return None


def _first_child(element, local_name: str):
    for child in list(element):
        if _local_name(child.tag) == local_name:
            return child
    return None


def _parent_map(root) -> dict[Any, Any]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _heading_level(element) -> int | None:
    name = _local_name(element.tag).lower()
    if len(name) == 2 and name.startswith("h") and name[1].isdigit():
        level = int(name[1])
        if 1 <= level <= 6:
            return level
    return None


def _heading_ancestor(element, parent_map: Mapping[Any, Any]):
    current = element
    while current is not None:
        if _heading_level(current) is not None:
            return current
        current = parent_map.get(current)
    return None


def _heading_section_elements(heading, parent_map: Mapping[Any, Any]) -> list[Any]:
    parent = parent_map.get(heading)
    if parent is None:
        return [heading]
    siblings = list(parent)
    try:
        start = siblings.index(heading)
    except ValueError:
        return [heading]
    level = _heading_level(heading) or 6
    selected = []
    for sibling in siblings[start:]:
        if sibling is not heading:
            sibling_level = _heading_level(sibling)
            if sibling_level is not None and sibling_level <= level:
                break
        selected.append(sibling)
    return selected or [heading]


def _plain_text_from_elements(elements: Sequence[Any]) -> str:
    blocks: list[str] = []
    for element in elements:
        blocks.extend(_element_text_blocks(element))
    return "\n".join(dict.fromkeys(block for block in blocks if block))


def _element_text_blocks(element) -> list[str]:
    ignored = {"script", "style", "nav"}
    block_tags = {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    blocks: list[str] = []

    def walk(node) -> str:
        tag = _local_name(node.tag).lower()
        if tag in ignored:
            return ""
        parts: list[str] = []
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        for child in list(node):
            child_text = walk(child)
            if child_text:
                parts.append(child_text)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
        text = normalize_effect_text(" ".join(parts))
        if tag in block_tags and text:
            blocks.append(text)
            return ""
        return text

    trailing = walk(element)
    if trailing:
        blocks.append(trailing)
    return blocks


def _split_href(href: str) -> tuple[str, str]:
    path, anchor = urldefrag(href)
    return unquote(path), unquote(anchor)


def _rule_id(path: str, anchor: str) -> str:
    return f"ocgrules:{path}#{anchor or 'top'}"


def _split_rule_section(
    section: RuleSection,
    *,
    max_section_chars: int,
) -> list[RuleSection]:
    if max_section_chars <= 0 or len(section.text) <= max_section_chars:
        return [section]
    chunks: list[RuleSection] = []
    chunk_texts = _chunk_rule_text(section.text, max_chars=max_section_chars)
    for index, chunk_text in enumerate(chunk_texts, start=1):
        if not chunk_text:
            continue
        chunks.append(
            RuleSection(
                rule_id=f"{section.rule_id}:chunk:{index}",
                title=section.title,
                chapter_path=section.chapter_path,
                xhtml_path=section.xhtml_path,
                anchor=section.anchor,
                text=chunk_text,
            )
        )
    return chunks


def _chunk_rule_text(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [
        normalize_effect_text(part)
        for part in re.split(r"\n+", text)
        if normalize_effect_text(part)
    ]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_parts = (
            _split_long_rule_paragraph(paragraph, max_chars=max_chars)
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for part in paragraph_parts:
            projected = current_len + len(part) + (1 if current else 0)
            if current and projected > max_chars:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += len(part) + (1 if current_len else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _split_long_rule_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？?!])\s*", paragraph)
        if part.strip()
    ]
    if not sentences:
        return [paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars)]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                sentence[index : index + max_chars]
                for index in range(0, len(sentence), max_chars)
            )
            continue
        projected = len(current) + len(sentence) + (1 if current else 0)
        if current and projected > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def _dedupe_rule_sections(sections: Sequence[RuleSection]) -> list[RuleSection]:
    deduped: dict[str, RuleSection] = {}
    for section in sections:
        deduped.setdefault(section.rule_id, section)
    return list(deduped.values())


def _weighted_rule_search_text(section: RuleSection) -> str:
    chapter = " > ".join(section.chapter_path)
    return "\n".join(
        [
            section.title,
            section.title,
            section.title,
            chapter,
            chapter,
            section.text,
        ]
    )


def _score_documents(
    query: str,
    documents: Sequence[tuple[str, str]],
    *,
    top_k: int,
) -> list[tuple[str, float]]:
    query_terms = tokenize_text(query)
    if not query_terms:
        return []
    query_set = set(query_terms)
    scored = []
    for doc_id, text in documents:
        terms = tokenize_text(text)
        if not terms:
            continue
        score = sum(1 for term in terms if term in query_set)
        if score:
            scored.append((doc_id, float(score)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def _score_faq_documents(
    query: str,
    faqs: Sequence[YgocdbFaqDocument],
    *,
    top_k: int,
    effect_tags: Sequence[str] = (),
    card_effect_tags_by_id: Mapping[int, Sequence[str]] | None = None,
) -> list[tuple[str, float]]:
    query_terms = tokenize_text(query)
    query_set = set(query_terms)
    tag_set = set(effect_tags)
    if not query_terms and not tag_set:
        return []

    scored = []
    tag_map = card_effect_tags_by_id or {}
    for faq in faqs:
        terms = tokenize_text(faq.searchable_text)
        score = float(sum(1 for term in terms if term in query_set))
        if tag_set and tag_map:
            card_ids = set(faq.primary_card_ids).union(faq.refer_card_ids)
            faq_tags = {
                tag
                for card_id in card_ids
                for tag in tag_map.get(card_id, ())
            }
            score += 2.0 * len(tag_set.intersection(faq_tags))
        if score:
            scored.append((faq.fid, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


class SimulationVerifier(Protocol):
    def verify(self, scenario: ParsedScenario) -> SimulationStatus:
        ...


class UnavailableSimulationVerifier:
    def verify(self, scenario: ParsedScenario) -> SimulationStatus:
        return SimulationStatus(
            status="not_available",
            warnings=("Simulator adapter is not available.",),
        )


def parse_scenario(
    text: str,
    *,
    known_cards: Sequence[Card] = (),
) -> ParsedScenario:
    original = text.strip()
    mentions = _extract_card_mentions(original, known_cards)
    questions = _extract_question_candidates(original)
    roles = tuple(role for role in ("我方", "对方", "双方") if role in original)
    actions = tuple(
        keyword
        for keyword in (
            "发动",
            "破坏",
            "特殊召唤",
            "特招",
            "解放",
            "无效",
            "攻击",
            "伤害",
            "降低",
            "代替",
            "适用",
            "选择",
        )
        if keyword in original
    )
    replacements = tuple(
        mention
        for mention in mentions
        if mention in original and ("代替" in original or "降低" in original)
    )
    warnings = () if mentions else ("No card names were identified from the scenario.",)
    return ParsedScenario(
        original_text=original,
        card_mentions=mentions,
        roles=roles,
        effect_actions=actions,
        objects=mentions,
        replacement_candidates=replacements,
        questions=questions or ((original,) if original else ()),
        warnings=warnings,
    )


def _rule_chapter_path(section: RuleSection) -> tuple[str, ...]:
    if len(section.chapter_path) >= 3:
        return section.chapter_path[:-1]
    return section.chapter_path


def _score_rule_chapter(
    query_terms: set[str],
    chapter_path: Sequence[str],
    section_titles: Sequence[str],
    sections: Sequence[RuleSection],
    term_idf: Mapping[str, float],
) -> float:
    path_text = " > ".join(chapter_path)
    path_score = _weighted_term_overlap(query_terms, path_text, term_idf)
    section_score = max(
        (
            _weighted_term_overlap(query_terms, title, term_idf)
            for title in section_titles
        ),
        default=0.0,
    )
    body_terms = {
        term
        for section in sections
        for term in tokenize_text(section.text)
    }
    body_score = sum(
        _rule_topic_term_weight(term) * term_idf.get(term, 1.0)
        for term in query_terms & body_terms
    )
    return path_score * 8.0 + section_score * 12.0 + body_score


def _weighted_term_overlap(
    query_terms: set[str],
    text: str,
    term_idf: Mapping[str, float],
) -> float:
    terms = set(tokenize_text(text))
    return sum(
        _rule_topic_term_weight(term) * term_idf.get(term, 1.0)
        for term in query_terms & terms
    )


def _rule_topic_term_weight(term: str) -> float:
    return 2.0 if len(term) > 1 else 0.25


QUESTION_MARKERS = (
    "吗",
    "能",
    "可以",
    "是否",
    "能否",
    "可否",
    "会不会",
    "会否",
    "是否可以",
    "是否能",
    "是否成功",
    "能成功",
    "可以成功",
    "成功处理",
    "处理成功",
    "还会",
    "还能",
    "是不是",
    "是否适用",
    "是否发动",
    "是否处理",
    "是否无效",
    "是否破坏",
    "是否特殊召唤",
)

SCENARIO_PARSER_FIELDS = (
    "card_mentions",
    "roles",
    "effect_actions",
    "objects",
    "replacement_candidates",
    "questions",
    "warnings",
)

NEUTRAL_EFFECT_ACTIONS = frozenset(
    {
        "发动",
        "适用",
        "处理",
        "结算",
        "破坏",
        "代替",
        "无效",
        "特殊召唤",
        "特招",
        "通常召唤",
        "反转召唤",
        "召唤",
        "送去墓地",
        "除外",
        "返回手卡",
        "返回卡组",
        "加入手卡",
        "抽卡",
        "丢弃",
        "解放",
        "选择",
        "取对象",
        "支付",
        "降低",
        "上升",
        "增加",
        "减少",
        "攻击",
        "战斗",
        "伤害",
        "贯穿伤害",
    }
)

PREFIX_QUESTION_MARKERS = (
    "是否",
    "能否",
    "可否",
    "会不会",
    "会否",
    "为什么",
    "为何",
    "怎么",
    "如何",
    "是不是",
    "哪一个",
    "哪张",
    "何时",
)

QUESTION_REPORTING_MARKERS = (
    "文档说明",
    "文档已经说明",
    "规则说明",
    "规则规定",
    "FAQ写着",
    "FAQ 写着",
    "记载了",
    "记述了",
    "指出了",
    "确认了",
)

QUESTION_META_SUFFIXES = (
    "的判断标准",
    "的规则",
    "的说明",
    "的记述",
    "的问题",
    "的条件",
    "的方法",
    "的结论",
)

SCENARIO_ROLE_LABELS = frozenset(
    {"我方", "对方", "对手", "双方", "自己", "对方玩家", "我方玩家"}
)

CARD_MENTION_STOPWORDS = frozenset(
    {"效果", "发动", "无效", "破坏", "召唤", "特招", "怪兽", "魔法", "陷阱", "卡片", "对象", "处理", "适用"}
)

QUESTION_DELIMITER_PAIRS = {
    "「": "」",
    "『": "』",
    "“": "”",
    "‘": "’",
    "《": "》",
    "〈": "〉",
    "〔": "〕",
    "〖": "〗",
    '"': '"',
    "（": "）",
    "(": ")",
    "【": "】",
    "[": "]",
}


def _extract_question_candidates(text: str) -> tuple[str, ...]:
    questions: list[str] = []
    for segment in _split_top_level_question_segments(text):
        for part in _split_top_level_clauses(segment, "。；;"):
            candidate = part.strip(" \t\r\n，,")
            if not candidate:
                continue
            candidate = _minimal_question_clause(candidate)
            if _is_question_fragment(candidate):
                questions.append(candidate)
    return tuple(dict.fromkeys(questions))


def _minimal_question_clause(value: str) -> str:
    if not _top_level_positions(value, "?？"):
        return value
    clauses = _split_top_level_clauses(value, "，,：:")
    for clause in reversed(clauses):
        if _top_level_positions(clause, "?？") and _question_clause_has_content(clause):
            return clause
    return value


def _top_level_positions(value: str, markers: str) -> list[int]:
    stack: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value):
        if stack and character == stack[-1]:
            stack.pop()
        elif character in QUESTION_DELIMITER_PAIRS:
            stack.append(QUESTION_DELIMITER_PAIRS[character])
        elif not stack and character in markers:
            positions.append(index)
    return positions


def _split_top_level_clauses(value: str, separators: str) -> list[str]:
    positions = _top_level_positions(value, separators)
    clauses: list[str] = []
    start = 0
    for position in positions:
        clause = value[start:position].strip()
        if clause:
            clauses.append(clause)
        start = position + 1
    trailing = value[start:].strip()
    if trailing:
        clauses.append(trailing)
    return clauses


def _split_top_level_question_segments(value: str) -> list[str]:
    stripped = value.strip()
    positions = _top_level_positions(stripped, "?？")
    segments: list[str] = []
    start = 0
    for position in positions:
        segment = stripped[start : position + 1].strip()
        if segment:
            segments.append(segment)
        start = position + 1
    trailing = stripped[start:].strip()
    if trailing:
        segments.append(trailing)
    return segments


def _question_clause_has_content(value: str) -> bool:
    positions = _top_level_positions(value, "?？")
    if not positions:
        return False
    prefix = value[: positions[-1]].strip()
    return bool(prefix.strip(" \t\r\n，,：:）)】]」』”\"'"))


def complete_scenario_questions(scenario: ParsedScenario) -> ParsedScenario:
    deterministic_questions = _extract_question_candidates(scenario.original_text)
    questions = _dedupe_questions(scenario.questions)
    added: list[str] = []
    for question in deterministic_questions:
        if not any(_same_question(question, existing) for existing in questions):
            questions.append(question)
            added.append(question)
    if not questions and scenario.original_text.strip():
        questions.append(scenario.original_text.strip())
    warnings = list(scenario.warnings)
    if added:
        warnings.append(
            "Question completeness checker added omitted questions: "
            + " | ".join(added)
        )
    return ParsedScenario(
        original_text=scenario.original_text,
        card_mentions=scenario.card_mentions,
        roles=scenario.roles,
        effect_actions=scenario.effect_actions,
        objects=scenario.objects,
        replacement_candidates=scenario.replacement_candidates,
        questions=tuple(_dedupe_questions(questions)),
        warnings=tuple(warnings),
    )


def _dedupe_questions(questions: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for question in questions:
        question = question.strip()
        if not question:
            continue
        if any(_same_question(question, existing) for existing in deduped):
            continue
        deduped.append(question)
    return deduped


def _same_question(left: str, right: str) -> bool:
    left_norm = re.sub(r"\s+", "", left.strip("。；;，,？?"))
    right_norm = re.sub(r"\s+", "", right.strip("。；;，,？?"))
    return bool(left_norm and right_norm) and (
        left_norm == right_norm
        or left_norm in right_norm
        or right_norm in left_norm
    )


def build_scenario_parser_prompt(text: str) -> str:
    return (
        "你是游戏王 OCG 场景事实抽取器（Parser），不是裁定器。只返回严格 JSON，不要输出 "
        "Markdown 或额外说明。\n"
        "你的唯一任务是逐字抽取用户原文中已经出现的事实片段；绝不裁定、绝不回答问题、绝不补充规则知识，"
        "也不得纠正、改写或推断卡名和效果。模糊卡名必须保留原文，交给后续 Card Resolution。\n"
        "顶层必须是 JSON 对象，且必须恰好包含以下键：card_mentions, roles, effect_actions, objects, "
        "replacement_candidates, questions, warnings。每个值必须是 JSON 数组；数组元素必须是非空字符串，"
        "不得使用对象或其他类型，不得增加 answer 等字段。warnings 必须是空数组。\n"
        "除 warnings 外，每个字符串都必须能在用户原文中定位；只允许忽略空格和常见卡名分隔符差异，"
        "不得概括、改写或新增推理。card_mentions 只能填写能和本地卡名对应的原文片段。"
        "objects 和 replacement_candidates 只能引用已填写的 card_mentions。"
        "questions 只抽取原文中带问号的疑问片段；每一项必须保留原文问号且只能包含一个问题，"
        "问号后不得附带答案、解释或下一问。原文没有问号时不要补写问号，questions 返回空数组，"
        "由后续确定性检查器处理。要覆盖每一个带问号的具体问题，不能只保留最后一个。\n"
        "effect_actions 只能填写原文已有的中性机制标签，例如：发动、适用、处理、结算、破坏、代替、无效、"
        "特殊召唤、降低、选择。不得填写“会造成贯穿伤害”“以实际卡名为准”等结论、建议或规则推断。\n\n"
        f"场景：\n{text}"
    )


def _normalize_extractive_text(value: str) -> str:
    return re.sub(r"[\s「」『』【】《》〈〉·・･\-_—－]", "", value).casefold()


def _is_source_extract(value: str, source: str) -> bool:
    normalized_value = _normalize_extractive_text(value)
    normalized_source = _normalize_extractive_text(source)
    return bool(normalized_value) and normalized_value in normalized_source


def _is_question_fragment(value: str) -> bool:
    stripped = value.strip()
    question_positions = _top_level_positions(stripped, "?？")
    if ("?" in stripped or "？" in stripped) and not question_positions:
        return False
    if question_positions:
        trailing = stripped[question_positions[-1] + 1 :].strip()
        return not trailing.strip("）)】]」』”\"'。！？?!；;，,")
    if stripped.endswith(("。", "；", ";")):
        return False
    if stripped.endswith("吗"):
        return True
    if any(marker in stripped for marker in QUESTION_REPORTING_MARKERS):
        return False
    if any(stripped.endswith(suffix) for suffix in QUESTION_META_SUFFIXES):
        return False
    if any(marker in stripped for marker in PREFIX_QUESTION_MARKERS):
        return True
    return bool(
        re.match(
            r"\s*(?:can|could|would|will|does|do|did|is|are|was|were|what|how|why|"
            r"when|where|who|which|whether)\b",
            value,
            flags=re.IGNORECASE,
        )
    )


def _is_llm_question_fragment(value: str) -> bool:
    stripped = value.strip()
    question_positions = _top_level_positions(stripped, "?？")
    if len(question_positions) != 1:
        return False
    trailing = stripped[question_positions[0] + 1 :].strip()
    return not trailing.strip("）)】]」』”\"'。！？?!；;，,")


def split_questions_deterministically(source: str) -> tuple[str, ...]:
    """Split source questions, explicitly preserving unsplittable source as one issue."""
    questions = _extract_question_candidates(source)
    if questions:
        return questions
    stripped = source.strip()
    return (stripped,) if stripped else ()


def build_question_split_prompt(source: str, *, focus: str = "") -> str:
    focus_instruction = (
        f"\n本次关注点：{focus.strip()}。关注点只用于检查可能遗漏的位置；"
        "即使提供关注点，也必须返回原文中的完整问题集，不得只返回关注部分。"
        if focus.strip()
        else ""
    )
    return (
        "你是游戏王 OCG 问题边界抽取器，不是裁定器。只返回严格 JSON，不要 Markdown "
        "或额外说明。\n"
        '顶层必须恰好是 {"questions":[字符串]}。每个字符串必须逐字来自用户原文，'
        "只能包含一个顶层终止问号边界，且问号后不得包含答案、解释或下一问。"
        "不得回答问题、不得添加规则、警告、建议或裁定。必须覆盖关注范围内的具体问题。"
        "如果原文没有可拆分的问句边界，必须把完整原文原样作为唯一 issue 返回；"
        "不得添加问号、改写、概括或补充答案。"
        f"{focus_instruction}\n\n用户原文：\n{source}"
    )


def validate_question_split_payload(payload: Any, source: str) -> tuple[str, ...]:
    if not isinstance(payload, dict) or set(payload) != {"questions"}:
        raise ValueError(
            'Question splitter output must be an object containing only "questions".'
        )
    raw_questions = payload["questions"]
    if not isinstance(raw_questions, list):
        raise ValueError("Question splitter questions must be an array.")
    if any(not isinstance(question, str) for question in raw_questions):
        raise ValueError("Question splitter questions must contain only strings.")
    questions = tuple(question.strip() for question in raw_questions)
    if any(not question for question in questions):
        raise ValueError("Question splitter questions must not contain empty strings.")
    if source.strip() and not questions:
        raise ValueError("Question splitter returned no questions for non-empty source text.")

    deterministic_candidates = _extract_question_candidates(source)
    has_terminal_question_boundary = bool(_top_level_positions(source, "?？"))
    if source.strip() and not has_terminal_question_boundary:
        if len(questions) != 1 or questions[0] != source.strip():
            raise ValueError(
                "Question splitter must return the exact full source as the only issue "
                "when no deterministic question boundary exists."
            )
        return questions

    for question in questions:
        if not _is_source_extract(question, source):
            raise ValueError("Question splitter returned text not found in the source.")
        if not _is_llm_question_fragment(question):
            raise ValueError(
                "Question splitter questions must have exactly one top-level terminal "
                "question boundary and no trailing answer text."
            )
        normalized = _normalize_extractive_text(question)
        if not any(
            normalized in _normalize_extractive_text(candidate)
            for candidate in deterministic_candidates
        ):
            raise ValueError(
                "Question splitter question crosses a deterministic source boundary."
            )
    for candidate in deterministic_candidates:
        if not any(_same_question(candidate, question) for question in questions):
            raise ValueError(
                "Question splitter omitted a deterministic source question boundary."
            )
    return tuple(_dedupe_questions(questions))


def build_card_mention_extraction_prompt(source: str, *, focus: str = "") -> str:
    focus_instruction = (
        f"\n本次关注点：{focus.strip()}。关注点只用于缩小抽取范围，不得生成原文外卡名。"
        if focus.strip()
        else ""
    )
    return (
        "你是游戏王 OCG 卡名片段抽取器，不是裁定器。只返回严格 JSON，不要 Markdown "
        "或额外说明。\n"
        '顶层必须恰好是 {"mentions":[字符串]}。每个字符串必须是用户原文中逐字出现的'
        "可能卡名片段。不得返回对象、卡片 id、答案、规则、警告、建议、效果结论或整句问题。"
        "只抽取具体卡片专名；不要把“大猫”“小猫”“那只怪兽”等上下文昵称、普通指代，"
        "或未作为具体卡名使用的系列名当成独立卡片。"
        f"{focus_instruction}\n\n用户原文：\n{source}"
    )


def validate_card_mention_extraction_payload(
    payload: Any,
    source: str,
    *,
    known_cards: Sequence[Card] = (),
) -> tuple[str, ...]:
    if not isinstance(payload, dict) or set(payload) != {"mentions"}:
        raise ValueError(
            'Card mention extractor output must be an object containing only "mentions".'
        )
    raw_mentions = payload["mentions"]
    if not isinstance(raw_mentions, list):
        raise ValueError("Card mention extractor mentions must be an array.")
    if any(not isinstance(mention, str) for mention in raw_mentions):
        raise ValueError("Card mention extractor mentions must contain only strings.")
    mentions = tuple(mention.strip() for mention in raw_mentions)
    if any(not mention for mention in mentions):
        raise ValueError("Card mention extractor mentions must not contain empty strings.")
    known_exact_names = {
        _normalize_extractive_text(card.name)
        for card in known_cards
        if card.name.strip()
    }
    filtered_mentions: list[str] = []
    for mention in mentions:
        if not _is_source_extract(mention, source):
            raise ValueError("Card mention extractor returned text not found in the source.")
        normalized = _normalize_extractive_text(mention)
        if (
            not normalized
            or len(normalized) > 80
            or mention in SCENARIO_ROLE_LABELS
            or mention in NEUTRAL_EFFECT_ACTIONS
            or normalized in CARD_MENTION_STOPWORDS
            or any(
                marker in mention
                for marker in (
                    "可以发动",
                    "不能发动",
                    "可以适用",
                    "不能适用",
                    "会造成",
                    "不会造成",
                    "结论",
                    "建议",
                    "应当",
                    "应该",
                    "以实际卡名为准",
                )
            )
            or re.search(r"[，。；！？?,;:]", mention)
            or _is_question_fragment(mention)
        ):
            raise ValueError(
                "Card mention extractor returned answer, mechanism, or question pollution."
            )
        is_short_non_exact = (
            len(normalized) < 4
            and normalized not in known_exact_names
            and not _source_quotes_exact_mention(source, mention)
        )
        if not is_short_non_exact:
            filtered_mentions.append(mention)
    return tuple(dict.fromkeys(filtered_mentions))


def _source_quotes_exact_mention(source: str, mention: str) -> bool:
    escaped = re.escape(mention.strip())
    if not escaped:
        return False
    return bool(
        re.search(
            rf"(?:「{escaped}」|『{escaped}』|“{escaped}”|\"{escaped}\")",
            source,
        )
    )


SCENARIO_FACT_FIELDS = frozenset(
    {"roles", "effect_actions", "objects", "replacement_candidates"}
)


def build_scenario_fact_extraction_prompt(
    source: str,
    *,
    field_name: str,
    card_mentions: Sequence[str] = (),
    focus: str = "",
) -> str:
    if field_name not in SCENARIO_FACT_FIELDS:
        raise ValueError(f"Unsupported scenario fact field: {field_name}.")
    field_rules = {
        "roles": "values 只能是原文中的中性玩家角色，例如我方、对方、对手、双方、自己。",
        "effect_actions": (
            "values 只能是原文中的中性机制动作标签，例如发动、适用、处理、结算、破坏、"
            "代替、无效、特殊召唤、降低、选择。"
        ),
        "objects": "values 只能逐字引用当前已抽取卡名。",
        "replacement_candidates": "values 只能逐字引用当前已抽取卡名中的代替候选。",
    }
    focus_instruction = (
        f"\n本次关注点：{focus.strip()}。关注点不得成为输出值。"
        if focus.strip()
        else ""
    )
    return (
        "你是游戏王 OCG 单字段事实抽取器，不是裁定器。只返回严格 JSON，不要 Markdown "
        "或额外说明。\n"
        f'本次只抽取字段 {field_name}，顶层必须恰好是 {{"values":[字符串]}}。'
        "所有值必须逐字来自用户原文。不得输出裁定、答案、警告、建议、规则知识或其他字段。\n"
        f"{field_rules[field_name]}{focus_instruction}\n"
        f"当前已抽取卡名：{json.dumps(list(card_mentions), ensure_ascii=False)}\n\n"
        f"用户原文：\n{source}"
    )


def validate_scenario_fact_payload(
    payload: Any,
    source: str,
    *,
    field_name: str,
    card_mentions: Sequence[str] = (),
) -> tuple[str, ...]:
    if field_name not in SCENARIO_FACT_FIELDS:
        raise ValueError(f"Unsupported scenario fact field: {field_name}.")
    if not isinstance(payload, dict) or set(payload) != {"values"}:
        raise ValueError(
            'Scenario fact extractor output must be an object containing only "values".'
        )
    raw_values = payload["values"]
    if not isinstance(raw_values, list):
        raise ValueError("Scenario fact extractor values must be an array.")
    if any(not isinstance(value, str) for value in raw_values):
        raise ValueError("Scenario fact extractor values must contain only strings.")
    values = tuple(value.strip() for value in raw_values)
    if any(not value for value in values):
        raise ValueError("Scenario fact extractor values must not contain empty strings.")
    for value in values:
        if not _is_source_extract(value, source):
            raise ValueError("Scenario fact extractor returned text not found in the source.")

    if field_name == "roles":
        if any(value not in SCENARIO_ROLE_LABELS for value in values):
            raise ValueError("Scenario roles must contain only neutral player-role labels.")
    elif field_name == "effect_actions":
        if any(value not in NEUTRAL_EFFECT_ACTIONS for value in values):
            raise ValueError(
                "Scenario effect_actions must contain only neutral mechanism labels."
            )
    else:
        allowed = {
            _normalize_extractive_text(mention)
            for mention in card_mentions
            if mention.strip()
        }
        if any(_normalize_extractive_text(value) not in allowed for value in values):
            raise ValueError(
                f"Scenario {field_name} values must reference current card mentions."
            )
    return tuple(dict.fromkeys(values))


def extract_scenario_facts_deterministically(
    source: str,
    *,
    field_name: str,
    card_mentions: Sequence[str] = (),
) -> tuple[str, ...]:
    if field_name not in SCENARIO_FACT_FIELDS:
        raise ValueError(f"Unsupported scenario fact field: {field_name}.")
    if field_name == "roles":
        return tuple(
            role
            for role in sorted(
                SCENARIO_ROLE_LABELS,
                key=lambda value: (source.find(value), value),
            )
            if role in source
        )
    if field_name == "effect_actions":
        return tuple(
            action
            for action in sorted(
                NEUTRAL_EFFECT_ACTIONS,
                key=lambda value: (source.find(value), value),
            )
            if action in source
        )
    values = tuple(mention for mention in card_mentions if mention in source)
    if field_name == "objects":
        return values
    if not any(marker in source for marker in ("代替", "替代", "降低")):
        return ()
    return values


def _matches_known_card_name(value: str, known_cards: Sequence[Card]) -> bool:
    normalized = _normalize_extractive_text(value)
    if len(normalized) < 2 or normalized in CARD_MENTION_STOPWORDS:
        return False
    for card in known_cards:
        card_name = _normalize_extractive_text(card.name)
        if not card_name:
            continue
        if normalized == card_name or normalized in card_name:
            return True
        if (
            min(len(normalized), len(card_name)) >= 4
            and abs(len(normalized) - len(card_name)) <= 2
            and SequenceMatcher(None, normalized, card_name).ratio() >= 0.8
        ):
            return True
    return False


def _is_plausible_card_mention(
    value: str,
    known_cards: Sequence[Card],
) -> bool:
    normalized = _normalize_extractive_text(value)
    if not normalized or len(normalized) > 80:
        return False
    if value in SCENARIO_ROLE_LABELS or value in NEUTRAL_EFFECT_ACTIONS:
        return False
    if re.search(r"[，。；！？?,;:]", value) or _is_question_fragment(value):
        return False
    return _matches_known_card_name(value, known_cards)


def _validate_scenario_parser_payload(
    payload: Any,
    source: str,
    known_cards: Sequence[Card],
) -> dict[str, tuple[str, ...]]:
    if not isinstance(payload, dict):
        raise ValueError("LLM scenario parser JSON must be an object.")

    expected = set(SCENARIO_PARSER_FIELDS)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise ValueError("LLM scenario parser has invalid fields (" + "; ".join(details) + ").")

    validated: dict[str, tuple[str, ...]] = {}
    for field_name in SCENARIO_PARSER_FIELDS:
        items = payload[field_name]
        if not isinstance(items, list):
            raise ValueError(f"LLM scenario parser field {field_name} must be an array.")
        if any(not isinstance(item, str) for item in items):
            raise ValueError(
                f"LLM scenario parser field {field_name} must contain only strings."
            )
        stripped = tuple(item.strip() for item in items)
        if any(not item for item in stripped):
            raise ValueError(
                f"LLM scenario parser field {field_name} must not contain empty strings."
            )
        validated[field_name] = stripped

    if validated["warnings"]:
        raise ValueError("LLM scenario parser warnings must be an empty array.")

    for field_name in SCENARIO_PARSER_FIELDS:
        if field_name == "warnings":
            continue
        for item in validated[field_name]:
            if not _is_source_extract(item, source):
                raise ValueError(
                    f"LLM scenario parser field {field_name} contains text not found in the source."
                )

    for question in validated["questions"]:
        if not _is_llm_question_fragment(question):
            raise ValueError(
                "LLM scenario parser questions must be source question fragments with exactly "
                "one terminal question boundary."
            )
        question_normalized = _normalize_extractive_text(question)
        deterministic_candidates = _extract_question_candidates(source)
        if not any(
            question_normalized in _normalize_extractive_text(candidate)
            for candidate in deterministic_candidates
        ):
            raise ValueError(
                "LLM scenario parser questions must stay within one deterministic source "
                "question boundary."
            )

    for mention in validated["card_mentions"]:
        if not _is_plausible_card_mention(mention, known_cards):
            raise ValueError(
                "LLM scenario parser card_mentions must contain only plausible card-name strings."
            )

    for role in validated["roles"]:
        if role not in SCENARIO_ROLE_LABELS:
            raise ValueError(
                "LLM scenario parser roles must contain only neutral player-role labels."
            )

    allowed_entity_values = {
        _normalize_extractive_text(item)
        for item in validated["card_mentions"]
    }
    for field_name in ("objects", "replacement_candidates"):
        for item in validated[field_name]:
            if _normalize_extractive_text(item) not in allowed_entity_values:
                raise ValueError(
                    f"LLM scenario parser field {field_name} must reference an extracted source entity."
                )

    for action in validated["effect_actions"]:
        if action not in NEUTRAL_EFFECT_ACTIONS:
            raise ValueError(
                "LLM scenario parser effect_actions must contain only neutral mechanism labels."
            )

    return validated


def parse_scenario_with_llm(
    text: str,
    llm: Llm,
    *,
    known_cards: Sequence[Card] = (),
) -> ParsedScenario:
    prompt = build_scenario_parser_prompt(text)
    response = llm.invoke(prompt)
    content = str(getattr(response, "content", response))
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM scenario parser returned invalid JSON.") from exc
    validated = _validate_scenario_parser_payload(payload, text, known_cards)
    return complete_scenario_questions(ParsedScenario(
        original_text=text,
        card_mentions=validated["card_mentions"],
        roles=validated["roles"],
        effect_actions=validated["effect_actions"],
        objects=validated["objects"],
        replacement_candidates=validated["replacement_candidates"],
        questions=validated["questions"],
        warnings=(),
    ))


def resolve_card_mentions(
    mentions: Sequence[str],
    cards: Sequence[Card],
) -> tuple[ResolvedCard, ...]:
    resolved = []
    for mention in mentions:
        normalized = _normalize_name(mention)
        exact = [card for card in cards if _normalize_name(card.name) == normalized]
        matches = exact or [
            card
            for card in cards
            if normalized and normalized in _normalize_name(card.name)
        ]
        resolved.append(
            ResolvedCard(
                mention=mention,
                matches=tuple(matches),
                ambiguous=len(matches) > 1,
            )
        )
    return tuple(resolved)


def _extract_card_mentions(text: str, cards: Sequence[Card]) -> tuple[str, ...]:
    positioned = [
        (text.index(card.name), -len(card.name), card.name)
        for card in cards
        if (
            card.name
            and len(card.name) >= 2
            and card.name in text
            and _normalize_extractive_text(card.name) not in CARD_MENTION_STOPWORDS
            and card.name not in SCENARIO_ROLE_LABELS
        )
    ]
    positioned.sort()
    mentions = [name for _position, _length, name in positioned]
    seen: set[str] = set()
    unique = []
    for name in mentions:
        if name not in seen:
            unique.append(name)
            seen.add(name)
    quoted = re.findall(r"[「『“\"]([^」』”\"]{2,})[」』”\"]", text)
    for value in quoted:
        if value not in seen and _is_likely_quoted_card_mention(value, cards):
            unique.append(value)
            seen.add(value)
    return tuple(unique)


def _is_likely_quoted_card_mention(value: str, cards: Sequence[Card]) -> bool:
    return _matches_known_card_name(value, cards)


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def build_card_effect_evidence(resolved_cards: Sequence[ResolvedCard]) -> list[EvidenceItem]:
    evidence = []
    for resolved in resolved_cards:
        if len(resolved.matches) != 1:
            continue
        card = resolved.matches[0]
        effect_category_names = tuple(decode_effect_category_labels(card.category))
        text_parts = [f"卡名：{card.name}"]
        stats: list[str] = []
        if card.atk is not None:
            stats.append(f"ATK {card.atk}")
        if card.defense is not None:
            stats.append(f"DEF {card.defense}")
        if card.level is not None:
            stats.append(f"等级/阶级/连接 {card.level}")
        if stats:
            text_parts.append("卡片数值：" + " / ".join(stats))
        if effect_category_names:
            text_parts.append("效果标签：" + "、".join(effect_category_names))
        text_parts.append("效果文本：")
        text_parts.append(card.description)
        evidence.append(
            EvidenceItem(
                evidence_id=f"card:{card.card_id}:effect",
                kind="card_effect",
                source="cards.cdb",
                title=card.name,
                text="\n".join(text_parts),
                relationship="resolved_card",
                card_ids=(card.card_id,),
                metadata={
                    "mention": resolved.mention,
                    "atk": card.atk,
                    "def": card.defense,
                    "level": card.level,
                    "type": card.type,
                    "race": card.race,
                    "attribute": card.attribute,
                    "effect_category_names": list(effect_category_names),
                },
            )
        )
    return evidence


class RulingAdjudicator:
    def __init__(
        self,
        cards: Sequence[Card],
        *,
        ruling_index: YgocdbRulingIndex | None = None,
        rule_index: RuleIndex | None = None,
        llm: Llm | None = None,
        simulator: SimulationVerifier | None = None,
    ) -> None:
        self.cards = list(cards)
        self.card_effect_tags_by_id = {
            card.card_id: decode_effect_category_labels(card.category)
            for card in self.cards
        }
        self.ruling_index = ruling_index
        self.rule_index = rule_index
        self.llm = llm
        self.simulator = simulator

    def adjudicate(self, scenario_text: str, *, simulate: bool = False) -> AdjudicationResponse:
        warnings: list[str] = []
        try:
            scenario = (
                parse_scenario_with_llm(scenario_text, self.llm, known_cards=self.cards)
                if self.llm is not None
                else parse_scenario(scenario_text, known_cards=self.cards)
            )
        except Exception as exc:
            warnings.append(f"LLM scenario parser failed; using deterministic parser: {exc}")
            scenario = parse_scenario(scenario_text, known_cards=self.cards)
        warnings.extend(scenario.warnings)

        resolved = resolve_card_mentions(scenario.card_mentions, self.cards)
        for item in resolved:
            if not item.matches:
                warnings.append(f"Card mention could not be resolved: {item.mention}")
            elif item.ambiguous:
                warnings.append(f"Card mention is ambiguous: {item.mention}")

        evidence = self._gather_evidence(scenario, resolved, warnings)
        simulation = self._simulate(scenario, simulate)
        warnings.extend(simulation.warnings)
        reasoning, answer, confidence = self._reason(scenario, evidence, warnings)
        return AdjudicationResponse(
            scenario=scenario,
            resolved_cards=resolved,
            evidence=tuple(evidence),
            reasoning=tuple(reasoning),
            answer=answer,
            confidence=confidence,
            warnings=tuple(warnings),
            simulation=simulation,
        )

    def _gather_evidence(
        self,
        scenario: ParsedScenario,
        resolved: Sequence[ResolvedCard],
        warnings: list[str],
    ) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = build_card_effect_evidence(resolved)
        card_ids = [
            item.matches[0].card_id
            for item in resolved
            if len(item.matches) == 1
        ]
        effect_tags = _resolved_effect_tags(resolved)
        mechanism_query = _mechanism_query(scenario, effect_tags=effect_tags)
        if self.ruling_index is None:
            warnings.append("ygocdb ruling evidence is unavailable.")
        else:
            for card_id in card_ids:
                evidence.extend(
                    faq.to_evidence("direct_card")
                    for faq in self.ruling_index.direct_faqs(card_id)
                )
                evidence.extend(
                    faq.to_evidence("related_card")
                    for faq in self.ruling_index.related_faqs(card_id)
                )
                supplement = self.ruling_index.supplement(card_id)
                if supplement is not None:
                    evidence.append(supplement.to_evidence())
            for faq in self.ruling_index.search(
                mechanism_query,
                top_k=3,
                effect_tags=effect_tags,
                card_effect_tags_by_id=self.card_effect_tags_by_id,
            ):
                evidence.append(faq.to_evidence("analogous"))

        if self.rule_index is None:
            warnings.append("OCG rules evidence is unavailable.")
        else:
            evidence.extend(
                section.to_evidence()
                for section in self.rule_index.search(mechanism_query, top_k=5)
            )
        return dedupe_evidence(evidence)

    def _simulate(self, scenario: ParsedScenario, simulate: bool) -> SimulationStatus:
        if not simulate:
            return SimulationStatus(status="not_requested")
        verifier = self.simulator or UnavailableSimulationVerifier()
        return verifier.verify(scenario)

    def _reason(
        self,
        scenario: ParsedScenario,
        evidence: Sequence[EvidenceItem],
        warnings: Sequence[str],
    ) -> tuple[list[ReasoningStep], str, str]:
        if self.llm is not None and evidence:
            try:
                raw_answer = self._llm_answer(scenario, evidence)
                source_evidence = [
                    item for item in evidence if item.kind in {"faq", "rule_passage"}
                ]
                answer = normalize_adjudication_answer_source_status(
                    raw_answer,
                    evidence,
                    faq_searched=self.ruling_index is not None,
                    rules_searched=self.rule_index is not None,
                    sufficient=bool(source_evidence),
                )
                allowed_ids = {item.evidence_id for item in evidence}
                validate_evidence_citations(answer, allowed_ids)
                validate_adjudication_answer_json(answer, allowed_ids)
                return (
                    [ReasoningStep("LLM generated an evidence-cited answer.", tuple())],
                    answer,
                    "medium",
                )
            except Exception:
                pass
        cited = tuple(item.evidence_id for item in evidence[:5])
        steps = [
            ReasoningStep(
                "Collected card text, ygocdb ruling evidence, and OCG rules evidence where available.",
                cited,
            )
        ]
        if not evidence:
            answer = "未找到足够证据，无法给出可靠裁定。"
            confidence = "low"
        else:
            answer = "\n".join(
                [
                    "基于已检索到的证据，当前只能给出证据摘要式裁定：",
                    *[
                        f"- [{item.evidence_id}] {item.title or item.kind}: {item.text[:180]}"
                        for item in evidence[:5]
                    ],
                    "请根据以上证据区分直接裁定、关联裁定、类似裁定和规则文档依据。",
                ]
            )
            confidence = "medium" if not warnings else "low"
        return steps, answer, confidence

    def _llm_answer(self, scenario: ParsedScenario, evidence: Sequence[EvidenceItem]) -> str:
        prompt = build_adjudication_answer_prompt(scenario, evidence)
        response = self.llm.invoke(prompt)
        return str(getattr(response, "content", response))


def build_adjudication_answer_prompt(
    scenario: ParsedScenario,
    evidence: Sequence[EvidenceItem],
) -> str:
    questions = list(scenario.questions) or [scenario.original_text]
    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "kind": item.kind,
            "relationship": item.relationship,
            "retrieval_stage": item.metadata.get("retrieval_stage", ""),
            "retrieval_stages": list(item.metadata.get("retrieval_stages", [])),
            "title": item.title,
            "card_ids": list(item.card_ids),
            "text": item.text,
        }
        for item in evidence
    ]
    output_schema = {
        "conclusion": "简短给出整体裁定结论。",
        "source_status": {
            "faq": "found | not_found | not_searched",
            "rules": "found | not_found | not_searched",
            "failure_reason": "证据不足或来源未命中的原因；没有则为空字符串。",
        },
        "answers": [
            {
                "question": "复述正在回答的问题。",
                "case_conclusion": "基于本案实际事实的结论。",
                "hypothetical_conclusion": (
                    "如果问题包含“如果/假设/即使”等假设分支，回答该假设分支；"
                    "否则使用空字符串。"
                ),
                "reasoning": [
                    {
                        "judgment": "只基于证据作出的一个关键判断。",
                        "evidence_ids": ["引用的 evidence_id"],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": "证据 id",
                "role": (
                    "direct_card / related_card / analogous / rule_document / "
                    "card_effect / card_supplement"
                ),
                "use": "说明这条证据如何用于裁定。",
            }
        ],
        "uncertainties": [
            "证据缺口或场景事实缺失；如果没有则使用空数组。"
        ],
    }
    return "\n".join(
        [
            "你是游戏王 OCG 裁定助手。",
            "只能依据提供的证据回答，不要补充证据之外的事实。",
            "只返回严格 JSON，不要使用 Markdown、代码块或额外说明文字。",
            "",
            "裁定回答规则：",
            "1. 在 conclusion 字段最先给出整体裁定结论。",
            "2. questions 中的每个问题都要分别回答，每个问题对应 answers 中的一个对象。",
            "3. 如果问题包含“如果/假设/即使”等假设分支，必须同时给出 case_conclusion 和 hypothetical_conclusion；不能只说前提不成立就停止。",
            "4. 证据优先级：direct_card FAQ 优先，其次 related_card FAQ，其次 card_supplement 和 card_effect，其次 analogous FAQ，最后 rule_document。",
            "5. 使用 analogous FAQ 时，必须在 reasoning.judgment 或 evidence_use.use 中说明类比关系。",
            "6. 每个关键判断都必须引用 evidence_ids，且所有 id 必须来自提供的 evidence。",
            "7. 如果证据不足，仍需回答可以确定的部分，并在 uncertainties 中列出缺口。",
            "8. 判断“后续处理是否适用”时，必须区分只是破坏了任意卡，还是破坏了该效果要求破坏的对象/那些卡。",
            "9. source_status 必须按 schema 输出；程序会依据实际检索和证据覆盖结果重新校正，不能用模型推测替代工具结果。",
            "",
            "输出 JSON schema：",
            json.dumps(output_schema, ensure_ascii=False, indent=2),
            "",
            "scenario:",
            scenario.original_text,
            "",
            "questions:",
            json.dumps(questions, ensure_ascii=False, indent=2),
            "",
            "evidence:",
            json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        ]
    )

def _resolved_effect_tags(resolved: Sequence[ResolvedCard]) -> tuple[str, ...]:
    tags: list[str] = []
    for item in resolved:
        if len(item.matches) != 1:
            continue
        tags.extend(decode_effect_category_labels(item.matches[0].category))
    return tuple(dict.fromkeys(tags))


def _mechanism_query(
    scenario: ParsedScenario,
    *,
    effect_tags: Sequence[str] = (),
) -> str:
    return "\n".join(
        [
            scenario.original_text,
            " ".join(scenario.effect_actions),
            " ".join(scenario.replacement_candidates),
            " ".join(effect_tags),
            "破坏 代替破坏 特殊召唤 效果处理 连锁",
        ]
    )


def dedupe_evidence(items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    relationship_priority = {
        "direct_card": 0,
        "related_card": 1,
        "card_supplement": 2,
        "card_effect": 3,
        "analogous": 4,
        "rule_document": 5,
    }
    deduped: dict[str, EvidenceItem] = {}
    for item in items:
        existing = deduped.get(item.evidence_id)
        if existing is None:
            deduped[item.evidence_id] = item
            continue
        relationships = tuple(
            sorted(
                dict.fromkeys(
                    [
                        *existing.relationship.split(","),
                        *item.relationship.split(","),
                    ]
                ),
                key=lambda value: relationship_priority.get(value, 99),
            )
        )
        retrieval_stages = list(
            dict.fromkeys(
                [
                    *existing.metadata.get("retrieval_stages", []),
                    *([existing.metadata.get("retrieval_stage")] if existing.metadata.get("retrieval_stage") else []),
                    *item.metadata.get("retrieval_stages", []),
                    *([item.metadata.get("retrieval_stage")] if item.metadata.get("retrieval_stage") else []),
                ]
            )
        )
        deduped[item.evidence_id] = EvidenceItem(
            evidence_id=existing.evidence_id,
            kind=existing.kind,
            source=existing.source,
            text=existing.text,
            relationship=",".join(relationships),
            title=existing.title,
            card_ids=tuple(sorted(set(existing.card_ids).union(item.card_ids))),
            metadata=dict(existing.metadata)
            | dict(item.metadata)
            | {
                "retrieval_stage": retrieval_stages[0] if retrieval_stages else "",
                "retrieval_stages": retrieval_stages,
            },
        )
    return list(deduped.values())


_CITATION_RE = re.compile(r"\[([a-zA-Z0-9_:/#.\-]+)\]")


def validate_evidence_citations(answer: str, allowed_ids: set[str]) -> None:
    unknown = [
        citation
        for citation in _CITATION_RE.findall(answer)
        if ":" in citation and citation not in allowed_ids
    ]
    if unknown:
        raise ValueError(f"Answer cited unknown evidence ids: {', '.join(sorted(set(unknown)))}")


def validate_adjudication_answer_json(answer: str, allowed_ids: set[str]) -> None:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM adjudication answer must be valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("LLM adjudication answer must be a JSON object.")

    def require_field(mapping: Mapping[str, Any], key: str, path: str) -> Any:
        if key not in mapping:
            raise ValueError(f"LLM adjudication answer field {path} is required.")
        return mapping[key]

    for key in (
        "conclusion",
        "source_status",
        "answers",
        "evidence_use",
        "uncertainties",
    ):
        if key not in payload:
            raise ValueError(f"LLM adjudication answer field {key} is required.")
    conclusion = payload["conclusion"]
    if not isinstance(conclusion, str) or not conclusion.strip():
        raise ValueError(
            "LLM adjudication answer field conclusion must be a non-empty string."
        )
    source_status = payload.get("source_status")
    if not isinstance(source_status, Mapping):
        raise ValueError("LLM adjudication answer source_status must be an object.")
    if set(source_status) != {"faq", "rules", "failure_reason"}:
        raise ValueError(
            "LLM adjudication answer source_status must contain faq, rules, and failure_reason."
        )
    allowed_source_values = {"found", "not_found", "not_searched"}
    for source in ("faq", "rules"):
        if source_status.get(source) not in allowed_source_values:
            raise ValueError(
                f"LLM adjudication answer source_status.{source} has an invalid value."
            )
    if not isinstance(source_status.get("failure_reason"), str):
        raise ValueError(
            "LLM adjudication answer source_status.failure_reason must be a string."
        )
    answers = payload["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(
            "LLM adjudication answer field answers must be a non-empty array."
        )
    for answer_index, answer_item in enumerate(answers):
        answer_path = f"answers[{answer_index}]"
        if not isinstance(answer_item, Mapping):
            raise ValueError(
                f"LLM adjudication answer field {answer_path} must be an object."
            )
        question = require_field(answer_item, "question", f"{answer_path}.question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"LLM adjudication answer field {answer_path}.question must be a non-empty string."
            )
        case_conclusion = require_field(
            answer_item,
            "case_conclusion",
            f"{answer_path}.case_conclusion",
        )
        if not isinstance(case_conclusion, str) or not case_conclusion.strip():
            raise ValueError(
                f"LLM adjudication answer field {answer_path}.case_conclusion must be a non-empty string."
            )
        hypothetical_conclusion = require_field(
            answer_item,
            "hypothetical_conclusion",
            f"{answer_path}.hypothetical_conclusion",
        )
        if not isinstance(hypothetical_conclusion, str):
            raise ValueError(
                f"LLM adjudication answer field {answer_path}.hypothetical_conclusion must be a string."
            )
        reasoning = require_field(answer_item, "reasoning", f"{answer_path}.reasoning")
        if not isinstance(reasoning, list):
            raise ValueError(
                f"LLM adjudication answer field {answer_path}.reasoning must be an array."
            )
        for reasoning_index, reasoning_item in enumerate(reasoning):
            reasoning_path = f"{answer_path}.reasoning[{reasoning_index}]"
            if not isinstance(reasoning_item, Mapping):
                raise ValueError(
                    f"LLM adjudication answer field {reasoning_path} must be an object."
                )
            judgment = require_field(
                reasoning_item,
                "judgment",
                f"{reasoning_path}.judgment",
            )
            if not isinstance(judgment, str) or not judgment.strip():
                raise ValueError(
                    f"LLM adjudication answer field {reasoning_path}.judgment must be a non-empty string."
                )
            evidence_ids = require_field(
                reasoning_item,
                "evidence_ids",
                f"{reasoning_path}.evidence_ids",
            )
            if not isinstance(evidence_ids, list) or not all(
                isinstance(evidence_id, str) for evidence_id in evidence_ids
            ):
                raise ValueError(
                    f"LLM adjudication answer field {reasoning_path}.evidence_ids must be an array of strings."
                )
    evidence_use = payload["evidence_use"]
    if not isinstance(evidence_use, list):
        raise ValueError(
            "LLM adjudication answer field evidence_use must be an array."
        )
    for evidence_index, evidence_item in enumerate(evidence_use):
        evidence_path = f"evidence_use[{evidence_index}]"
        if not isinstance(evidence_item, Mapping):
            raise ValueError(
                f"LLM adjudication answer field {evidence_path} must be an object."
            )
        for key in ("evidence_id", "role", "use"):
            field_path = f"{evidence_path}.{key}"
            value = require_field(evidence_item, key, field_path)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"LLM adjudication answer field {field_path} must be a non-empty string."
                )
    uncertainties = payload["uncertainties"]
    if not isinstance(uncertainties, list) or not all(
        isinstance(uncertainty, str) for uncertainty in uncertainties
    ):
        raise ValueError(
            "LLM adjudication answer field uncertainties must be an array of strings."
        )
    cited_ids = _collect_json_evidence_ids(payload)
    unknown = sorted(cited_id for cited_id in cited_ids if cited_id not in allowed_ids)
    if unknown:
        raise ValueError(f"Answer cited unknown evidence ids: {', '.join(unknown)}")


def _collect_json_evidence_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "evidence_id" and isinstance(nested, str):
                ids.add(nested)
            elif key == "evidence_ids" and isinstance(nested, list):
                ids.update(item for item in nested if isinstance(item, str))
            else:
                ids.update(_collect_json_evidence_ids(nested))
    elif isinstance(value, list):
        for item in value:
            ids.update(_collect_json_evidence_ids(item))
    return ids
