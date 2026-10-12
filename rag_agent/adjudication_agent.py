from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence, TypedDict

from rag_agent.adjudication import (
    AdjudicationProgressCallback,
    AdjudicationProgressEvent,
    AdjudicationResponse,
    EvidenceItem,
    Llm,
    ParsedScenario,
    ReasoningStep,
    ResolvedCard,
    RuleIndex,
    SimulationStatus,
    YgocdbRulingIndex,
    build_adjudication_answer_prompt,
    build_card_mention_extraction_prompt,
    build_card_effect_evidence,
    build_question_split_prompt,
    build_scenario_fact_extraction_prompt,
    complete_scenario_questions,
    dedupe_evidence,
    derive_adjudication_source_status,
    extract_scenario_facts_deterministically,
    normalize_adjudication_answer_source_status,
    parse_scenario,
    parse_scenario_with_llm,
    resolve_card_mentions,
    split_questions_deterministically,
    validate_card_mention_extraction_payload,
    validate_adjudication_answer_json,
    validate_evidence_citations,
    validate_question_split_payload,
    validate_scenario_fact_payload,
)
from rag_agent.card_metadata import decode_effect_category_labels
from rag_agent.cards import Card
from rag_agent.config import Settings
from rag_agent.retrieval.sparse import SparseCardRetriever
from rag_agent.ruling_index import PersistentRulingRetriever


CandidateStatus = Literal["resolved", "ambiguous", "unresolved"]
CandidateValidationStatus = Literal["accepted", "rejected", "needs_validation"]
ClaimStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "unknown",
]
ClaimType = Literal[
    "card_text",
    "ruling_fact",
    "rule_principle",
    "case_application",
    "hypothetical_branch",
]
EvidenceRelevance = Literal["high", "medium", "low", "none"]
EvidenceEntailment = Literal[
    "supports",
    "partially_supports",
    "contradicts",
    "does_not_entail",
    "unknown",
]
IssuePolarity = Literal["affirmative", "negative", "conditional", "unknown"]
AgentToolName = Literal[
    "extract_card_mentions",
    "split_questions",
    "extract_scenario_facts",
    "resolve_cards",
    "search_card",
    "search_direct_faq",
    "search_related_faq",
    "search_card_supplement",
    "search_analogous_faq",
    "classify_rule_chapters",
    "search_rules",
    "critique_coverage",
    "submit_ruling",
]


SHORT_MENTION_NORMALIZED_LENGTH = 3
SUBSTRING_MIN_NORMALIZED_LENGTH = 3
SUBSTRING_MIN_COVERAGE = 0.35
DEFAULT_VERIFIED_CARD_ALIASES: dict[str, tuple[int, str]] = {
    "卡通青眼究极龙": (71808988, "common translated word order"),
    "霸王眷龙 凶恶毒": (43387895, "common typo for 凶饿毒"),
    "幻禄的天杯龙": (23657016, "common 的/之 variant"),
    "奶油蛋糕喵★黯蜜味": (31603289, "known translated name variant"),
}
AUTO_MENTION_STOPWORDS = {
    normalize
    for normalize in (
        "发动",
        "效果",
        "破坏",
        "代替",
        "适用",
        "选择",
        "无效",
        "特殊召唤",
        "直接攻击",
        "战斗伤害",
        "卡名",
        "对象",
        "场上",
        "墓地",
        "额外卡组",
        "怪兽",
    )
}

CLAIM_STATUSES: set[str] = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "unknown",
}
CLAIM_TYPES: set[str] = {
    "card_text",
    "ruling_fact",
    "rule_principle",
    "case_application",
    "hypothetical_branch",
}
EVIDENCE_RELEVANCE_VALUES: set[str] = {"high", "medium", "low", "none"}
EVIDENCE_ENTAILMENT_VALUES: set[str] = {
    "supports",
    "partially_supports",
    "contradicts",
    "does_not_entail",
    "unknown",
}
ISSUE_POLARITY_VALUES: set[str] = {
    "affirmative",
    "negative",
    "conditional",
    "unknown",
}
MECHANISM_EVIDENCE_KINDS: set[str] = {"faq", "card_supplement", "rule_passage"}


@dataclass(frozen=True)
class CardCandidate:
    mention: str
    card_id: int
    name: str
    score: float
    source: str
    match_reason: str
    effect_text: str = ""
    normalized_mention: str = ""
    normalized_name: str = ""
    validation_status: CandidateValidationStatus = "needs_validation"
    validation_reason: str = "not validated"
    validation_source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "card_id": self.card_id,
            "name": self.name,
            "score": self.score,
            "source": self.source,
            "match_reason": self.match_reason,
            "effect_text": self.effect_text,
            "normalized_mention": self.normalized_mention,
            "normalized_name": self.normalized_name,
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "validation_source": self.validation_source,
        }


@dataclass(frozen=True)
class CardResolution:
    mention: str
    status: CandidateStatus
    candidates: tuple[CardCandidate, ...] = ()
    selected_card_id: int | None = None
    selection_source: str = "rules"
    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "status": self.status,
            "selected_card_id": self.selected_card_id,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selection_source": self.selection_source,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class ExtractedCardMention:
    text: str
    context: str = ""
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "context": self.context,
            "source": self.source,
        }


@dataclass(frozen=True)
class CardResolutionLoopTrace:
    extracted_mentions: tuple[ExtractedCardMention, ...] = ()
    retry_count: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "extracted_mentions": [mention.to_dict() for mention in self.extracted_mentions],
            "retry_count": self.retry_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CoverageReport:
    sufficient: bool
    missing_cards: tuple[str, ...] = ()
    ambiguous_cards: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    missing_hypothetical_evidence: tuple[str, ...] = ()
    next_action: str = "final"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "missing_cards": list(self.missing_cards),
            "ambiguous_cards": list(self.ambiguous_cards),
            "missing_evidence": list(self.missing_evidence),
            "missing_hypothetical_evidence": list(self.missing_hypothetical_evidence),
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class ClaimEvidenceCheck:
    evidence_id: str
    relevance: EvidenceRelevance
    entailment: EvidenceEntailment
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "relevance": self.relevance,
            "entailment": self.entailment,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ClaimCoverageClaim:
    claim_id: str
    claim: str
    claim_type: ClaimType
    status: ClaimStatus
    stable_id: str = ""
    issue_id: str = ""
    semantic_role: str = ""
    polarity: IssuePolarity = "unknown"
    supported_answer: str = ""
    question: str = ""
    branch: str = "main"
    required: bool = True
    decisive: bool = True
    depends_on: tuple[str, ...] = ()
    evidence_checks: tuple[ClaimEvidenceCheck, ...] = ()
    inference_reason: str = ""
    gap: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "stable_id": self.stable_id or self.claim_id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "status": self.status,
            "issue_id": self.issue_id,
            "semantic_role": self.semantic_role,
            "polarity": self.polarity,
            "supported_answer": self.supported_answer,
            "question": self.question,
            "branch": self.branch,
            "required": self.required,
            "decisive": self.decisive,
            "depends_on": list(self.depends_on),
            "evidence_checks": [check.to_dict() for check in self.evidence_checks],
            "inference_reason": self.inference_reason,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ClaimCoverageIssue:
    issue_id: str
    question: str
    status: ClaimStatus
    polarity: IssuePolarity = "unknown"
    supported_answer: str = ""
    claim_ids: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    unresolved_claim_ids: tuple[str, ...] = ()
    inference_reason: str = ""
    gap: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "question": self.question,
            "status": self.status,
            "polarity": self.polarity,
            "supported_answer": self.supported_answer,
            "claim_ids": list(self.claim_ids),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "unresolved_claim_ids": list(self.unresolved_claim_ids),
            "inference_reason": self.inference_reason,
            "gap": self.gap,
        }


@dataclass(frozen=True)
class ClaimCoverageGap:
    claim_id: str
    reason: str
    suggested_queries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "reason": self.reason,
            "suggested_queries": list(self.suggested_queries),
        }


@dataclass(frozen=True)
class ClaimCoverageReport:
    required_claims: tuple[ClaimCoverageClaim, ...] = ()
    issues: tuple[ClaimCoverageIssue, ...] = ()
    gaps: tuple[ClaimCoverageGap, ...] = ()
    suggested_queries: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    schema_version: int = 1
    source: str = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "required_claims": [claim.to_dict() for claim in self.required_claims],
            "issues": [issue.to_dict() for issue in self.issues],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "suggested_queries": list(self.suggested_queries),
            "contradictions": list(self.contradictions),
        }


@dataclass(frozen=True)
class ClaimCoverageAggregation:
    sufficient: bool
    blocking_claims: tuple[str, ...] = ()
    supported_issues: tuple[str, ...] = ()
    negative_supported_issues: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = ()
    partial_claims: tuple[str, ...] = ()
    unknown_claims: tuple[str, ...] = ()
    contradicted_claims: tuple[str, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    suggested_queries: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "blocking_claims": list(self.blocking_claims),
            "supported_issues": list(self.supported_issues),
            "negative_supported_issues": list(self.negative_supported_issues),
            "unresolved_issues": list(self.unresolved_issues),
            "partial_claims": list(self.partial_claims),
            "unknown_claims": list(self.unknown_claims),
            "contradicted_claims": list(self.contradicted_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "suggested_queries": list(self.suggested_queries),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class EvidenceContextItem:
    section: str
    evidence_id: str
    kind: str
    relationship: str
    title: str = ""
    card_ids: tuple[int, ...] = ()
    text: str = ""
    cited_by_claim_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "relationship": self.relationship,
            "title": self.title,
            "card_ids": list(self.card_ids),
            "text": self.text,
            "cited_by_claim_ids": list(self.cited_by_claim_ids),
        }


@dataclass(frozen=True)
class AdjudicationEvidenceContext:
    items: tuple[EvidenceContextItem, ...]
    evidence_revision: str

    def to_dict(self) -> dict[str, Any]:
        sections: dict[str, list[dict[str, Any]]] = {}
        for item in self.items:
            sections.setdefault(item.section, []).append(item.to_dict())
        return {
            "evidence_revision": self.evidence_revision,
            "sections": sections,
        }


@dataclass(frozen=True)
class TraceEntry:
    node: str
    summary: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PlannerAction:
    tool: AgentToolName
    arguments: Mapping[str, Any] = field(default_factory=dict)
    rationale: str = ""
    notes: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "rationale": self.rationale,
            "notes": list(self.notes),
            "hypotheses": list(self.hypotheses),
        }


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    ok: bool
    summary: str
    data: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    validation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "summary": self.summary,
            "data": dict(self.data),
            "evidence_ids": list(self.evidence_ids),
            "warnings": list(self.warnings),
            "validation_error": self.validation_error,
        }


@dataclass(frozen=True)
class SubmitValidationIssue:
    code: str
    path: str
    message: str
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "retryable": self.retryable,
        }


class SubmitValidationError(ValueError):
    def __init__(self, issues: Sequence[SubmitValidationIssue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))

    @property
    def consumes_submit_budget(self) -> bool:
        return any(not issue.retryable for issue in self.issues)


@dataclass(frozen=True)
class ToolHistoryEntry:
    step: int
    action: PlannerAction
    observation: ToolObservation
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict(),
            "duplicate": self.duplicate,
        }


@dataclass(frozen=True)
class SubmitAttempt:
    accepted: bool
    answer: str
    validation_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "answer": self.answer,
            "validation_error": self.validation_error,
        }


SCENARIO_PREPARATION_TOOLS = frozenset(
    {"extract_card_mentions", "split_questions", "extract_scenario_facts"}
)
SCENARIO_DEPENDENT_ACTIVE_TOOLS = frozenset(
    {
        "resolve_cards",
        "search_card",
        "search_direct_faq",
        "search_related_faq",
        "search_card_supplement",
        "search_analogous_faq",
        "classify_rule_chapters",
        "search_rules",
        "critique_coverage",
        "submit_ruling",
    }
)
MANDATORY_SCENARIO_FIELDS = ("card_mentions", "questions")
SCENARIO_FIELD_TO_TOOL = {
    "card_mentions": "extract_card_mentions",
    "questions": "split_questions",
    "roles": "extract_scenario_facts",
    "effect_actions": "extract_scenario_facts",
    "objects": "extract_scenario_facts",
    "replacement_candidates": "extract_scenario_facts",
}


def _initial_scenario_field_sources() -> dict[str, Mapping[str, Any]]:
    return {
        field_name: {
            "status": "pending",
            "tool": tool,
            "strategy": "",
            "focus": "",
            "revision": 0,
        }
        for field_name, tool in SCENARIO_FIELD_TO_TOOL.items()
    }


@dataclass(frozen=True)
class AdjudicationAgentState:
    scenario_text: str
    scenario: ParsedScenario | None = None
    agent_loop_preparation_required: bool = False
    scenario_field_sources: Mapping[str, Mapping[str, Any]] = field(
        default_factory=_initial_scenario_field_sources
    )
    scenario_revision: int = 0
    mention_queries: tuple[str, ...] = ()
    extracted_card_mentions: tuple[ExtractedCardMention, ...] = ()
    card_resolution_loop: CardResolutionLoopTrace = field(default_factory=CardResolutionLoopTrace)
    card_resolutions: tuple[CardResolution, ...] = ()
    resolved_cards: tuple[ResolvedCard, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    evidence_retrieval_stages: tuple[Mapping[str, Any], ...] = ()
    rule_chapter_catalog: tuple[Mapping[str, Any], ...] = ()
    rule_chapter_candidates: tuple[Mapping[str, Any], ...] = ()
    planner_notes: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    tool_history: tuple[ToolHistoryEntry, ...] = ()
    tool_signatures: tuple[str, ...] = ()
    submit_attempts: tuple[SubmitAttempt, ...] = ()
    agent_loop_complete: bool = False
    coverage: CoverageReport = field(
        default_factory=lambda: CoverageReport(sufficient=False, next_action="extract")
    )
    claim_coverage_report: ClaimCoverageReport | None = None
    claim_coverage_aggregation: ClaimCoverageAggregation | None = None
    trace: tuple[TraceEntry, ...] = ()
    warnings: tuple[str, ...] = ()
    iterations: int = 0
    final_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_text": self.scenario_text,
            "scenario": self.scenario.to_dict() if self.scenario else None,
            "agent_loop_preparation_required": self.agent_loop_preparation_required,
            "scenario_field_sources": {
                field_name: dict(provenance)
                for field_name, provenance in self.scenario_field_sources.items()
            },
            "scenario_revision": self.scenario_revision,
            "mention_queries": list(self.mention_queries),
            "extracted_card_mentions": [
                mention.to_dict() for mention in self.extracted_card_mentions
            ],
            "card_resolution_loop": self.card_resolution_loop.to_dict(),
            "card_resolutions": [
                resolution.to_dict() for resolution in self.card_resolutions
            ],
            "resolved_cards": [card.to_dict() for card in self.resolved_cards],
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_retrieval_stages": [
                dict(stage) for stage in self.evidence_retrieval_stages
            ],
            "rule_chapter_catalog": [
                dict(chapter) for chapter in self.rule_chapter_catalog
            ],
            "rule_chapter_candidates": [
                dict(candidate) for candidate in self.rule_chapter_candidates
            ],
            "planner_notes": list(self.planner_notes),
            "hypotheses": list(self.hypotheses),
            "tool_history": [entry.to_dict() for entry in self.tool_history],
            "tool_signatures": list(self.tool_signatures),
            "submit_attempts": [attempt.to_dict() for attempt in self.submit_attempts],
            "agent_loop_complete": self.agent_loop_complete,
            "coverage": self.coverage.to_dict(),
            "claim_coverage_report": (
                self.claim_coverage_report.to_dict()
                if self.claim_coverage_report is not None
                else None
            ),
            "claim_coverage_aggregation": (
                self.claim_coverage_aggregation.to_dict()
                if self.claim_coverage_aggregation is not None
                else None
            ),
            "trace": [entry.to_dict() for entry in self.trace],
            "warnings": list(self.warnings),
            "iterations": self.iterations,
            "final_answer": self.final_answer,
        }


class AgentGraphState(TypedDict, total=False):
    agent_state: AdjudicationAgentState


def create_initial_agent_state(scenario_text: str) -> AdjudicationAgentState:
    return AdjudicationAgentState(scenario_text=scenario_text)


def _scenario_replacing(
    scenario: ParsedScenario,
    **changes: Sequence[str],
) -> ParsedScenario:
    data: dict[str, Any] = {
        "original_text": scenario.original_text,
        "card_mentions": scenario.card_mentions,
        "roles": scenario.roles,
        "effect_actions": scenario.effect_actions,
        "objects": scenario.objects,
        "replacement_candidates": scenario.replacement_candidates,
        "questions": scenario.questions,
        "warnings": scenario.warnings,
    }
    data.update(changes)
    return ParsedScenario(**data)


def _updated_scenario_field_sources(
    state: AdjudicationAgentState,
    field_name: str,
    *,
    tool: str,
    strategy: str,
    focus: str,
    revision: int,
) -> dict[str, Mapping[str, Any]]:
    sources = {
        name: dict(provenance)
        for name, provenance in state.scenario_field_sources.items()
    }
    sources[field_name] = {
        "status": "ready",
        "tool": tool,
        "strategy": strategy,
        "focus": focus,
        "revision": revision,
    }
    return sources


def _derived_state_reset() -> dict[str, Any]:
    return {
        "coverage": CoverageReport(sufficient=False, next_action="retry"),
        "claim_coverage_report": None,
        "claim_coverage_aggregation": None,
        "submit_attempts": (),
        "agent_loop_complete": False,
        "final_answer": None,
    }


def _scenario_revision_reset(state: AdjudicationAgentState) -> dict[str, Any]:
    retained_history = tuple(
        entry
        for entry in state.tool_history
        if entry.action.tool not in SCENARIO_DEPENDENT_ACTIVE_TOOLS
    )
    retained_signatures: list[str] = []
    for signature in state.tool_signatures:
        try:
            payload = json.loads(signature)
        except (TypeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("tool") in SCENARIO_PREPARATION_TOOLS
        ):
            retained_signatures.append(signature)
    return {
        "evidence": (),
        "evidence_retrieval_stages": (),
        "rule_chapter_candidates": (),
        "tool_history": retained_history,
        "tool_signatures": tuple(retained_signatures),
        **_derived_state_reset(),
    }


def _mandatory_preparation_gaps(state: AdjudicationAgentState) -> tuple[str, ...]:
    if not state.agent_loop_preparation_required:
        return ()
    missing: list[str] = []
    for field_name in MANDATORY_SCENARIO_FIELDS:
        provenance = state.scenario_field_sources.get(field_name, {})
        if provenance.get("status") != "ready":
            missing.append(SCENARIO_FIELD_TO_TOOL[field_name])
    return tuple(missing)


def _merge_scenarios(
    primary: ParsedScenario,
    fallback: ParsedScenario,
) -> ParsedScenario:
    return complete_scenario_questions(ParsedScenario(
        original_text=primary.original_text or fallback.original_text,
        card_mentions=_merge_string_tuples(primary.card_mentions, fallback.card_mentions),
        roles=fallback.roles,
        effect_actions=fallback.effect_actions,
        objects=fallback.objects,
        replacement_candidates=fallback.replacement_candidates,
        questions=_merge_string_tuples(primary.questions, fallback.questions),
        warnings=fallback.warnings,
    ))


def _merge_string_tuples(
    primary: Sequence[str],
    fallback: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            item.strip()
            for item in (*primary, *fallback)
            if item and item.strip()
        )
    )


def normalize_card_name(value: str) -> str:
    separators = set(
        " \t\r\n\u3000\u00b7\u30fb\uff65-\u2014\u2013_/\uff0f,\uff0c\u3001.\u3002:"
        "\uff1a;\uff1b!\uff01?\uff1f\"'\u201c\u201d\u2018\u2019\uff08\uff09()"
        "\uff3b\uff3d[]\u3010\u3011\u300a\u300b\u3008\u3009\u300c\u300d\u300e\u300f"
    )
    return "".join(ch for ch in value.lower() if ch not in separators)


def candidate_mention_queries(
    text: str,
    scenario: ParsedScenario | None = None,
    *,
    alias_mentions: Sequence[str] = (),
) -> tuple[str, ...]:
    queries: list[str] = []
    if scenario is not None:
        queries.extend(scenario.card_mentions)
    queries.extend(alias for alias in alias_mentions if alias and alias in text)
    for match in re.findall(
        r"[\u4e00-\u9fffA-Za-z0-9]+(?:[\u00b7\u30fb\uff65\s\u3000][\u4e00-\u9fffA-Za-z0-9]+)+",
        text,
    ):
        cleaned = match.strip(" \t\r\n,.;:!?\uff0c\u3002\uff01\uff1f\uff1b")
        if len(normalize_card_name(cleaned)) >= 2:
            queries.append(cleaned)
    queries.extend(_middle_dot_windows(text))
    return _clean_candidate_mention_queries(queries)


def _clean_candidate_mention_queries(queries: Sequence[str]) -> tuple[str, ...]:
    kept = tuple(
        dict.fromkeys(
            query.strip()
            for query in queries
            if query.strip() and not _is_auto_mention_noise(query)
        )
    )
    return _drop_covered_short_mentions(kept)


def _is_auto_mention_noise(query: str) -> bool:
    normalized = normalize_card_name(query)
    if not normalized or len(normalized) < 2:
        return True
    if normalized in AUTO_MENTION_STOPWORDS:
        return True
    if re.search(
        r"\b(?:can|could|would|will|does|do|did|is|are|was|were|should|may|"
        r"might|what|how|why|when|where|whether)\b",
        query,
        flags=re.IGNORECASE,
    ):
        return True
    if any(fragment in query for fragment in ("场上·墓", "卡名·效")):
        return True
    if any(marker in query for marker in ("如果", "可以用", "还能", "是否", "吗", "？", "，", ",", "。")) and (
        len(normalized) > 12 or any(separator in query for separator in ("\u00b7", "\u30fb", "\uff65"))
    ):
        return True
    if len(normalized) > 30 and any(
        marker in query
        for marker in (
            "这张卡",
            "发动",
            "效果",
            "怪兽",
            "回合",
            "伤害",
            "对象",
            "场合",
        )
    ):
        return True
    return False


def _drop_covered_short_mentions(queries: Sequence[str]) -> tuple[str, ...]:
    normalized = {query: normalize_card_name(query) for query in queries}
    covered: set[str] = set()
    for query, query_normalized in normalized.items():
        if len(query_normalized) < 2:
            continue
        for other, other_normalized in normalized.items():
            if query == other:
                continue
            if len(other_normalized) <= len(query_normalized):
                continue
            if len(other_normalized) - len(query_normalized) < 2:
                continue
            if query_normalized and query_normalized in other_normalized:
                covered.add(query)
                break
    return tuple(query for query in queries if query not in covered)


def _middle_dot_windows(text: str) -> tuple[str, ...]:
    queries: list[str] = []
    token = r"[\u4e00-\u9fffA-Za-z0-9]"
    for match in re.finditer(r"[\u00b7\u30fb\uff65]", text):
        left_match = re.search(f"{token}+$", text[: match.start()])
        right_match = re.match(f"{token}+", text[match.end() :])
        if left_match is None or right_match is None:
            continue
        left = left_match.group(0)
        right = right_match.group(0)
        for left_len in range(2, min(len(left), 12) + 1):
            for right_len in range(1, min(len(right), 2) + 1):
                queries.append(f"{left[-left_len:]}{match.group(0)}{right[:right_len]}")
    return tuple(queries)


def parse_card_name_extraction_response(text: str) -> tuple[ExtractedCardMention, ...]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM card-name extraction output must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM card-name extraction output must be a JSON object.")
    if set(payload) != {"mentions"}:
        raise ValueError("LLM card-name extraction output contains unexpected fields.")
    mentions = payload.get("mentions")
    if not isinstance(mentions, list):
        raise ValueError("LLM card-name extraction mentions must be a list.")
    parsed: list[ExtractedCardMention] = []
    for item in mentions:
        if not isinstance(item, dict):
            raise ValueError("LLM card-name extraction mention must be an object.")
        if not set(item).issubset({"text", "context"}):
            raise ValueError("LLM card-name extraction mention contains unexpected fields.")
        value = str(item.get("text") or "").strip()
        if not value:
            continue
        parsed.append(
            ExtractedCardMention(
                text=value,
                context=str(item.get("context") or "").strip(),
                source="llm",
            )
        )
    return tuple(dict.fromkeys(parsed))


def extract_card_mentions_with_llm(
    scenario_text: str,
    llm: Llm,
) -> tuple[ExtractedCardMention, ...]:
    response = llm.invoke(build_card_name_extraction_prompt(scenario_text))
    return parse_card_name_extraction_response(str(getattr(response, "content", response)))


AGENT_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "extract_card_mentions": {
        "description": (
            "Prepare or correct source-grounded card mentions. Use before card resolution "
            "or evidence retrieval."
        ),
        "arguments": {
            "focus": "optional source-grounded extraction focus",
            "strategy": "optional llm | deterministic; defaults to llm",
        },
    },
    "split_questions": {
        "description": (
            "Prepare or correct source-grounded top-level questions. Replaces the current "
            "question list atomically."
        ),
        "arguments": {
            "focus": "optional source-grounded extraction focus",
            "strategy": "optional llm | deterministic; defaults to llm",
        },
    },
    "extract_scenario_facts": {
        "description": (
            "Optionally prepare one mechanism-fact field. This is not required before retrieval."
        ),
        "arguments": {
            "field": "required: roles | effect_actions | objects | replacement_candidates",
            "focus": "optional source-grounded extraction focus",
            "strategy": "optional llm | deterministic; defaults to llm",
        },
    },
    "resolve_cards": {
        "description": "Resolve possible card-name mentions through local card lookup/RAG and validity gates.",
        "arguments": {"mentions": "array of mention strings; omit or empty to use known scenario mentions"},
    },
    "search_card": {
        "description": "Search local card candidates only. This does not resolve cards.",
        "arguments": {"query": "card-name search string", "top_k": "optional integer"},
    },
    "search_direct_faq": {
        "description": "Search direct FAQ evidence for resolved or provided card ids.",
        "arguments": {"card_ids": "optional array of local resolved card ids", "query": "optional query text"},
    },
    "search_related_faq": {
        "description": "Search refer-related FAQ evidence for resolved or provided card ids.",
        "arguments": {"card_ids": "optional array of local resolved card ids", "query": "optional query text"},
    },
    "search_card_supplement": {
        "description": "Search ygocdb supplement evidence for resolved or provided card ids.",
        "arguments": {"card_ids": "optional array of local resolved card ids", "query": "optional query text"},
    },
    "search_analogous_faq": {
        "description": "Search analogous FAQ evidence with a mechanism-specific query.",
        "arguments": {"query": "required mechanism-specific query", "card_ids": "optional array of card ids"},
    },
    "classify_rule_chapters": {
        "description": "可选兼容工具：用词法匹配给 EPUB 规则章节排序；不是 search_rules 的前置条件。",
        "arguments": {
            "query": "必填，只表达一个核心机制的分类查询",
            "top_k": "可选，返回的候选章节数量",
        },
    },
    "search_rules": {
        "description": "检索 OCG 规则段落；chapter_ids 可直接选择完整章节目录中的任意有效 id。",
        "arguments": {
            "query": "必填，机制化的规则段落查询",
            "chapter_ids": "可选，EPUB 章节 id 数组",
        },
    },
    "critique_coverage": {
        "description": "Inspect unresolved cards and evidence gaps before deciding next action.",
        "arguments": {},
    },
    "submit_ruling": {
        "description": "Submit final strict JSON ruling answer for validation.",
        "arguments": {"answer": "required answer object or JSON string"},
    },
}


SUBMIT_RULING_ANSWER_SCHEMA: dict[str, Any] = {
    "conclusion": "必填。总体结论，先直接回答最终裁定。",
    "source_status": {
        "faq": "必填。found | not_found | not_searched；程序按实际工具结果校正。",
        "rules": "必填。found | not_found | not_searched；程序按实际工具结果校正。",
        "failure_reason": "必填字符串。证据缺口或来源未命中原因；没有则为空字符串。",
    },
    "answers": [
        {
            "question": "必填。原问题中的一个具体问题。",
            "case_conclusion": "必填。本案结论；如果不能确定，明确说明不确定点。",
            "hypothetical_conclusion": "必填。若问题包含假设分支，在这里回答假设成立时的结论；没有假设则为空字符串。",
            "reasoning": [
                {
                    "judgment": "必填。一个关键判断；判断后必须列出依据。",
                    "claim_id": "已有 Claim Coverage Report 时建议填写。必须来自 claim_coverage_report.required_claims；若引用 issue_id，可为空。",
                    "issue_id": "已有 Claim Coverage Report 时可填写。必须来自 claim_coverage_report.issues。",
                    "claim_ids": ["可选。需要引用多个 claim 时使用。"],
                    "issue_ids": ["可选。需要引用多个 issue 时使用。"],
                    "evidence_ids": [
                        "必填。只能使用 evidence_pool 中已经存在的 evidence_id。"
                    ],
                }
            ],
        }
    ],
    "evidence_use": [
        {
            "evidence_id": "必填。只能使用 evidence_pool 中已经存在的 evidence_id。",
            "role": "必填。说明 direct_card / related_card / analogous / rule_document / card_supplement 等证据关系。",
            "use": "必填。简述该证据支持哪个判断。",
        }
    ],
    "uncertainties": [
        "必填数组。列出未解析卡名、证据不足或类比限制；没有则为空数组。"
    ],
}


def parse_agent_planner_response(text: str) -> PlannerAction:
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Agent planner output must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Agent planner output must be a JSON object.")
    allowed_top_keys = {"tool", "action", "arguments", "rationale", "notes", "hypotheses"}
    if not set(payload).issubset(allowed_top_keys):
        raise ValueError("Agent planner output contains unexpected fields.")
    raw_tool = payload.get("tool", payload.get("action"))
    if not isinstance(raw_tool, str) or not raw_tool:
        raise ValueError("Agent planner output must include a tool name.")
    if raw_tool not in AGENT_TOOL_SCHEMAS:
        raise ValueError(f"Unknown adjudication tool: {raw_tool}.")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("Agent planner arguments must be a JSON object.")
    _validate_tool_arguments(raw_tool, arguments)
    return PlannerAction(
        tool=raw_tool,  # type: ignore[arg-type]
        arguments=arguments,
        rationale=str(payload.get("rationale") or "").strip(),
        notes=_string_tuple(payload.get("notes")),
        hypotheses=_string_tuple(payload.get("hypotheses")),
    )


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list):
        raise ValueError("Agent planner notes and hypotheses must be strings or arrays.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def _validate_tool_arguments(tool: str, arguments: Mapping[str, Any]) -> None:
    if tool in {"extract_card_mentions", "split_questions"}:
        _optional_string(arguments, "focus")
        _optional_parser_strategy(arguments)
        _reject_unknown_arguments(arguments, {"focus", "strategy"})
        return
    if tool == "extract_scenario_facts":
        field_name = arguments.get("field")
        if field_name not in {
            "roles",
            "effect_actions",
            "objects",
            "replacement_candidates",
        }:
            raise ValueError(
                "field must be one of roles, effect_actions, objects, "
                "replacement_candidates."
            )
        _optional_string(arguments, "focus")
        _optional_parser_strategy(arguments)
        _reject_unknown_arguments(arguments, {"field", "focus", "strategy"})
        return
    if tool == "resolve_cards":
        _optional_string_list(arguments, "mentions")
        _reject_unknown_arguments(arguments, {"mentions"})
        return
    if tool == "search_card":
        _required_string(arguments, "query")
        _optional_positive_int(arguments, "top_k")
        _reject_unknown_arguments(arguments, {"query", "top_k"})
        return
    if tool in {"search_direct_faq", "search_related_faq", "search_card_supplement"}:
        _optional_int_list(arguments, "card_ids")
        _optional_string(arguments, "query")
        _reject_unknown_arguments(arguments, {"card_ids", "query"})
        return
    if tool == "search_analogous_faq":
        _required_string(arguments, "query")
        _optional_int_list(arguments, "card_ids")
        _reject_unknown_arguments(arguments, {"query", "card_ids"})
        return
    if tool == "classify_rule_chapters":
        _required_string(arguments, "query")
        _optional_positive_int(arguments, "top_k")
        _reject_unknown_arguments(arguments, {"query", "top_k"})
        return
    if tool == "search_rules":
        _required_string(arguments, "query")
        _optional_string_list(arguments, "chapter_ids")
        _reject_unknown_arguments(arguments, {"query", "chapter_ids"})
        return
    if tool == "critique_coverage":
        _reject_unknown_arguments(arguments, set())
        return
    if tool == "submit_ruling":
        if "answer" not in arguments:
            raise ValueError("submit_ruling requires answer.")
        _reject_unknown_arguments(arguments, {"answer"})
        return
    raise ValueError(f"Unknown adjudication tool: {tool}.")


def _reject_unknown_arguments(arguments: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(arguments).difference(allowed)
    if unknown:
        raise ValueError(f"Unexpected tool arguments: {', '.join(sorted(unknown))}.")


def _required_string(arguments: Mapping[str, Any], key: str) -> None:
    if not isinstance(arguments.get(key), str) or not str(arguments.get(key)).strip():
        raise ValueError(f"{key} must be a non-empty string.")


def _optional_string(arguments: Mapping[str, Any], key: str) -> None:
    if key in arguments and not isinstance(arguments.get(key), str):
        raise ValueError(f"{key} must be a string.")


def _optional_parser_strategy(arguments: Mapping[str, Any]) -> None:
    if "strategy" not in arguments:
        return
    if arguments.get("strategy") not in {"llm", "deterministic"}:
        raise ValueError("strategy must be llm or deterministic.")


def _optional_string_list(arguments: Mapping[str, Any], key: str) -> None:
    if key not in arguments or arguments.get(key) is None:
        return
    value = arguments.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings.")


def _optional_int_list(arguments: Mapping[str, Any], key: str) -> None:
    if key not in arguments or arguments.get(key) is None:
        return
    value = arguments.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array of integers.")
    for item in value:
        if isinstance(item, bool):
            raise ValueError(f"{key} must be an array of integers.")
        try:
            int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an array of integers.") from exc


def _optional_positive_int(arguments: Mapping[str, Any], key: str) -> None:
    if key not in arguments or arguments.get(key) is None:
        return
    if isinstance(arguments.get(key), bool):
        raise ValueError(f"{key} must be a positive integer.")
    try:
        value = int(arguments[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer.")


def parse_claim_coverage_report(
    text: str,
    *,
    allowed_evidence_ids: set[str],
) -> ClaimCoverageReport:
    try:
        payload = json.loads(_strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Claim Coverage Report must be valid JSON.") from exc
    return claim_coverage_report_from_mapping(
        payload,
        allowed_evidence_ids=allowed_evidence_ids,
        source="llm",
    )


def claim_coverage_report_from_mapping(
    payload: Any,
    *,
    allowed_evidence_ids: set[str],
    source: str = "llm",
) -> ClaimCoverageReport:
    if not isinstance(payload, Mapping):
        raise ValueError("Claim Coverage Report must be a JSON object.")
    allowed_keys = {
        "schema_version",
        "required_claims",
        "claims",
        "issues",
        "gaps",
        "suggested_queries",
        "contradictions",
        "source",
    }
    unknown_keys = set(payload).difference(allowed_keys)
    if unknown_keys:
        raise ValueError(
            "Claim Coverage Report contains unexpected fields: "
            + ", ".join(sorted(unknown_keys))
        )
    raw_claims = payload.get("required_claims", payload.get("claims"))
    if not isinstance(raw_claims, list):
        raise ValueError("Claim Coverage Report requires required_claims array.")
    claims = tuple(
        _claim_from_mapping(item, allowed_evidence_ids=allowed_evidence_ids)
        for item in raw_claims
    )
    if not claims:
        raise ValueError("Claim Coverage Report requires at least one claim.")
    claim_ids = {claim.claim_id for claim in claims}
    if len(claim_ids) != len(claims):
        raise ValueError("Claim Coverage Report contains duplicate claim ids.")
    for claim in claims:
        missing = [claim_id for claim_id in claim.depends_on if claim_id not in claim_ids]
        if missing:
            raise ValueError(
                f"Claim {claim.claim_id} depends on unknown claim ids: "
                + ", ".join(missing)
            )
    issues = tuple(
        _issue_from_mapping(item, claim_ids=claim_ids, allowed_evidence_ids=allowed_evidence_ids)
        for item in _list_value(payload, "issues")
    )
    if not issues:
        issues = _issues_from_claims(claims)
    else:
        issues = _normalize_issues_against_claims(issues, claims)
    issue_ids = {issue.issue_id for issue in issues}
    if len(issue_ids) != len(issues):
        raise ValueError("Claim Coverage Report contains duplicate issue ids.")
    gaps = tuple(_gap_from_mapping(item, claim_ids=claim_ids) for item in _list_value(payload, "gaps"))
    suggested_queries = tuple(str(item).strip() for item in _list_value(payload, "suggested_queries") if str(item).strip())
    contradictions = tuple(str(item).strip() for item in _list_value(payload, "contradictions") if str(item).strip())
    schema_version = int(payload.get("schema_version") or 1)
    return ClaimCoverageReport(
        required_claims=claims,
        issues=issues,
        gaps=gaps,
        suggested_queries=suggested_queries,
        contradictions=contradictions,
        schema_version=schema_version,
        source=str(payload.get("source") or source),
    )


def _claim_from_mapping(
    payload: Any,
    *,
    allowed_evidence_ids: set[str],
) -> ClaimCoverageClaim:
    if not isinstance(payload, Mapping):
        raise ValueError("Claim entry must be an object.")
    claim_id = _non_empty_string(payload, "claim_id")
    claim = _non_empty_string(payload, "claim")
    if "claim_type" not in payload and "type" not in payload:
        raise ValueError(f"claim_type is required for {claim_id}.")
    raw_type = str(payload.get("claim_type", payload.get("type")))
    if raw_type not in CLAIM_TYPES:
        raise ValueError(f"Invalid claim type for {claim_id}: {raw_type}.")
    raw_status = str(payload.get("status") or "")
    if raw_status not in CLAIM_STATUSES:
        raise ValueError(f"Invalid claim status for {claim_id}: {raw_status}.")
    raw_polarity = str(payload.get("polarity") or payload.get("answer_polarity") or "unknown")
    if raw_polarity not in ISSUE_POLARITY_VALUES:
        raise ValueError(f"Invalid claim polarity for {claim_id}: {raw_polarity}.")
    depends_on = tuple(str(item).strip() for item in _list_value(payload, "depends_on") if str(item).strip())
    evidence_checks = tuple(
        _evidence_check_from_mapping(item, allowed_evidence_ids=allowed_evidence_ids)
        for item in _list_value(payload, "evidence_checks")
    )
    return ClaimCoverageClaim(
        claim_id=claim_id,
        stable_id=str(payload.get("stable_id") or ""),
        claim=claim,
        claim_type=raw_type,  # type: ignore[arg-type]
        status=raw_status,  # type: ignore[arg-type]
        issue_id=str(payload.get("issue_id") or ""),
        semantic_role=str(payload.get("semantic_role") or payload.get("role") or ""),
        polarity=raw_polarity,  # type: ignore[arg-type]
        supported_answer=str(payload.get("supported_answer") or payload.get("answer") or ""),
        question=str(payload.get("question") or ""),
        branch=str(payload.get("branch") or "main"),
        required=bool(payload.get("required", True)),
        decisive=bool(payload.get("decisive", True)),
        depends_on=depends_on,
        evidence_checks=evidence_checks,
        inference_reason=str(payload.get("inference_reason") or ""),
        gap=str(payload.get("gap") or ""),
    )


def _issue_from_mapping(
    payload: Any,
    *,
    claim_ids: set[str],
    allowed_evidence_ids: set[str],
) -> ClaimCoverageIssue:
    if not isinstance(payload, Mapping):
        raise ValueError("Issue entry must be an object.")
    issue_id = _non_empty_string(payload, "issue_id")
    question = _non_empty_string(payload, "question")
    raw_status = str(payload.get("status") or "")
    if raw_status not in CLAIM_STATUSES:
        raise ValueError(f"Invalid issue status for {issue_id}: {raw_status}.")
    raw_polarity = str(payload.get("polarity") or "unknown")
    if raw_polarity not in ISSUE_POLARITY_VALUES:
        raise ValueError(f"Invalid issue polarity for {issue_id}: {raw_polarity}.")
    raw_claim_ids = tuple(
        str(item).strip()
        for item in _list_value(payload, "claim_ids")
        if str(item).strip()
    )
    unknown_claim_ids = [claim_id for claim_id in raw_claim_ids if claim_id not in claim_ids]
    if unknown_claim_ids:
        raise ValueError(
            f"Issue {issue_id} cites unknown claim ids: "
            + ", ".join(unknown_claim_ids)
        )
    supporting_evidence_ids = tuple(
        str(item).strip()
        for item in _list_value(payload, "supporting_evidence_ids")
        if str(item).strip()
    )
    unknown_evidence = [
        evidence_id
        for evidence_id in supporting_evidence_ids
        if evidence_id not in allowed_evidence_ids
    ]
    if unknown_evidence:
        raise ValueError(
            f"Issue {issue_id} cites unknown evidence ids: "
            + ", ".join(unknown_evidence)
        )
    unresolved_claim_ids = tuple(
        str(item).strip()
        for item in _list_value(payload, "unresolved_claim_ids")
        if str(item).strip()
    )
    unknown_unresolved = [
        claim_id for claim_id in unresolved_claim_ids if claim_id not in claim_ids
    ]
    if unknown_unresolved:
        raise ValueError(
            f"Issue {issue_id} references unknown unresolved claim ids: "
            + ", ".join(unknown_unresolved)
        )
    return ClaimCoverageIssue(
        issue_id=issue_id,
        question=question,
        status=raw_status,  # type: ignore[arg-type]
        polarity=raw_polarity,  # type: ignore[arg-type]
        supported_answer=str(payload.get("supported_answer") or payload.get("answer") or ""),
        claim_ids=raw_claim_ids,
        supporting_evidence_ids=supporting_evidence_ids,
        unresolved_claim_ids=unresolved_claim_ids,
        inference_reason=str(payload.get("inference_reason") or ""),
        gap=str(payload.get("gap") or ""),
    )


def _issues_from_claims(
    claims: Sequence[ClaimCoverageClaim],
) -> tuple[ClaimCoverageIssue, ...]:
    issues: list[ClaimCoverageIssue] = []
    grouped: dict[str, list[ClaimCoverageClaim]] = {}
    for claim in claims:
        issue_id = claim.issue_id or stable_issue_id(
            claim.question or claim.claim,
            branch=claim.branch,
            semantic_role=claim.semantic_role or claim.claim_type,
        )
        grouped.setdefault(issue_id, []).append(claim)
    for issue_id, issue_claims in grouped.items():
        decisive_claims = [claim for claim in issue_claims if claim.required and claim.decisive]
        if any(claim.status == "supported" for claim in decisive_claims):
            status: ClaimStatus = "supported"
        elif any(claim.status == "contradicted" for claim in decisive_claims):
            status = "contradicted"
        elif any(claim.status == "partially_supported" for claim in decisive_claims):
            status = "partially_supported"
        elif any(claim.status == "unsupported" for claim in decisive_claims):
            status = "unsupported"
        else:
            status = "unknown"
        polarity = next(
            (claim.polarity for claim in issue_claims if claim.polarity != "unknown"),
            "unknown",
        )
        supported_answer = next(
            (claim.supported_answer for claim in issue_claims if claim.supported_answer),
            "",
        )
        supporting_evidence_ids = tuple(
            dict.fromkeys(
                check.evidence_id
                for claim in issue_claims
                for check in claim.evidence_checks
                if check.entailment in {"supports", "partially_supports"}
            )
        )
        unresolved_claim_ids = tuple(
            claim.claim_id for claim in issue_claims if claim.status != "supported"
        )
        issues.append(
            ClaimCoverageIssue(
                issue_id=issue_id,
                question=issue_claims[0].question or issue_claims[0].claim,
                status=status,
                polarity=polarity,
                supported_answer=supported_answer,
                claim_ids=tuple(claim.claim_id for claim in issue_claims),
                supporting_evidence_ids=supporting_evidence_ids,
                unresolved_claim_ids=unresolved_claim_ids,
                inference_reason=next(
                    (claim.inference_reason for claim in issue_claims if claim.inference_reason),
                    "",
                ),
                gap=next((claim.gap for claim in issue_claims if claim.gap), ""),
            )
        )
    return tuple(issues)


def stabilize_claim_coverage_report(
    report: ClaimCoverageReport,
    state: AdjudicationAgentState,
) -> ClaimCoverageReport:
    previous = state.claim_coverage_report
    previous_claims = {
        claim.claim_id: claim for claim in previous.required_claims
    } if previous is not None else {}
    previous_signatures = {
        _claim_signature(claim): claim.claim_id
        for claim in previous.required_claims
    } if previous is not None else {}
    claims: list[ClaimCoverageClaim] = []
    used_claim_ids: set[str] = set()
    for claim in report.required_claims:
        signature = _claim_signature(claim)
        existing = previous_claims.get(claim.claim_id)
        claim_id = claim.claim_id
        if existing is not None and _claim_signature(existing) != signature:
            claim_id = stable_claim_id(
                claim.question or claim.claim,
                branch=claim.branch,
                claim_type=claim.claim_type,
                semantic_role=claim.semantic_role or claim.claim,
            )
        elif signature in previous_signatures:
            claim_id = previous_signatures[signature]
        claim_id = _unique_stable_id(claim_id, used_claim_ids)
        used_claim_ids.add(claim_id)
        generated_stable_id = stable_claim_id(
            claim.question or claim.claim,
            branch=claim.branch,
            claim_type=claim.claim_type,
            semantic_role=claim.semantic_role or claim.claim,
        )
        issue_id = claim.issue_id or stable_issue_id(
            claim.question or claim.claim,
            branch=claim.branch,
            semantic_role=claim.semantic_role or claim.claim_type,
        )
        claims.append(
            ClaimCoverageClaim(
                claim_id=claim_id,
                stable_id=claim.stable_id or generated_stable_id,
                claim=claim.claim,
                claim_type=claim.claim_type,
                status=claim.status,
                issue_id=issue_id,
                semantic_role=claim.semantic_role,
                polarity=claim.polarity,
                supported_answer=claim.supported_answer,
                question=claim.question,
                branch=claim.branch,
                required=claim.required,
                decisive=claim.decisive,
                depends_on=tuple(
                    previous_signatures.get(_claim_signature(previous_claims[dependency_id]), dependency_id)
                    if dependency_id in previous_claims
                    else dependency_id
                    for dependency_id in claim.depends_on
                ),
                evidence_checks=claim.evidence_checks,
                inference_reason=claim.inference_reason,
                gap=claim.gap,
            )
        )
    id_map = {
        old.claim_id: new.claim_id
        for old, new in zip(report.required_claims, claims, strict=False)
    }
    claims = [
        ClaimCoverageClaim(
            claim_id=claim.claim_id,
            stable_id=claim.stable_id,
            claim=claim.claim,
            claim_type=claim.claim_type,
            status=claim.status,
            issue_id=claim.issue_id,
            semantic_role=claim.semantic_role,
            polarity=claim.polarity,
            supported_answer=claim.supported_answer,
            question=claim.question,
            branch=claim.branch,
            required=claim.required,
            decisive=claim.decisive,
            depends_on=tuple(id_map.get(dependency_id, dependency_id) for dependency_id in claim.depends_on),
            evidence_checks=claim.evidence_checks,
            inference_reason=claim.inference_reason,
            gap=claim.gap,
        )
        for claim in claims
    ]
    issues = tuple(
        ClaimCoverageIssue(
            issue_id=issue.issue_id,
            question=issue.question,
            status=issue.status,
            polarity=issue.polarity,
            supported_answer=issue.supported_answer,
            claim_ids=tuple(id_map.get(claim_id, claim_id) for claim_id in issue.claim_ids),
            supporting_evidence_ids=issue.supporting_evidence_ids,
            unresolved_claim_ids=tuple(
                id_map.get(claim_id, claim_id) for claim_id in issue.unresolved_claim_ids
            ),
            inference_reason=issue.inference_reason,
            gap=issue.gap,
        )
        for issue in (report.issues or _issues_from_claims(claims))
    )
    issues = _normalize_issues_against_claims(
        issues if report.issues else _issues_from_claims(claims),
        claims,
    )
    gaps = tuple(
        ClaimCoverageGap(
            claim_id=id_map.get(gap.claim_id, gap.claim_id),
            reason=gap.reason,
            suggested_queries=gap.suggested_queries,
        )
        for gap in report.gaps
    )
    return ClaimCoverageReport(
        required_claims=tuple(claims),
        issues=issues,
        gaps=gaps,
        suggested_queries=report.suggested_queries,
        contradictions=report.contradictions,
        schema_version=max(2, report.schema_version),
        source=report.source,
    )


def claim_coverage_semantic_defects(
    report: ClaimCoverageReport,
    state: AdjudicationAgentState,
) -> tuple[str, ...]:
    """Return structural semantic defects that make a claim graph unsafe to aggregate."""
    defects: list[str] = []
    claims_by_id = {claim.claim_id: claim for claim in report.required_claims}
    scenario_questions = tuple(state.scenario.questions) if state.scenario else ()
    issue_questions = tuple(issue.question for issue in report.issues)
    for question in scenario_questions:
        normalized_question = normalize_card_name(question)
        if not any(
            normalized_question == normalize_card_name(issue_question)
            or normalized_question in normalize_card_name(issue_question)
            or normalize_card_name(issue_question) in normalized_question
            for issue_question in issue_questions
            if normalize_card_name(issue_question)
        ):
            defects.append(f"用户问题没有对应 issue：{question}")

    for issue in report.issues:
        issue_claims = [
            claims_by_id[claim_id]
            for claim_id in issue.claim_ids
            if claim_id in claims_by_id
        ]
        decisive_applications = [
            claim
            for claim in issue_claims
            if claim.required
            and claim.decisive
            and claim.claim_type == "case_application"
        ]
        if not decisive_applications:
            defects.append(
                f"issue {issue.issue_id} 缺少 required=true、decisive=true 的 case_application"
            )

    for claim in report.required_claims:
        branch_is_hypothetical = claim.branch.lower().startswith("hypothetical")
        question_is_hypothetical = _has_hypothetical(claim.question)
        if claim.claim_type == "hypothetical_branch" and not (
            branch_is_hypothetical and question_is_hypothetical
        ):
            defects.append(
                f"claim {claim.claim_id} 被标为 hypothetical_branch，"
                "但不属于用户明确提出的假设分支"
            )
    return tuple(dict.fromkeys(defects))


def ensure_actionable_claim_gaps(
    report: ClaimCoverageReport,
    *,
    evidence_by_id: Mapping[str, EvidenceItem],
) -> ClaimCoverageReport:
    aggregation = aggregate_claim_coverage(
        report,
        evidence_by_id=evidence_by_id,
    )
    blocking_ids = set(aggregation.blocking_claims)
    blocking_ids.update(aggregation.partial_claims)
    blocking_ids.update(aggregation.unknown_claims)
    blocking_ids.update(aggregation.unsupported_claims)
    blocking_ids.update(aggregation.contradicted_claims)
    if not blocking_ids:
        return report

    gaps_by_id = {gap.claim_id: gap for gap in report.gaps}
    claims: list[ClaimCoverageClaim] = []
    global_queries = list(report.suggested_queries)
    for claim in report.required_claims:
        if claim.claim_id not in blocking_ids:
            claims.append(claim)
            continue
        reason = claim.gap.strip() or (
            "现有证据尚不能直接推出该场景应用命题，需要检索能明确支持或反驳它的裁定或规则。"
        )
        query = f"{claim.question} {claim.claim}".strip()
        existing_gap = gaps_by_id.get(claim.claim_id)
        suggested_queries = tuple(
            dict.fromkeys(
                [
                    *(existing_gap.suggested_queries if existing_gap else ()),
                    query,
                ]
            )
        )
        gaps_by_id[claim.claim_id] = ClaimCoverageGap(
            claim_id=claim.claim_id,
            reason=(existing_gap.reason.strip() if existing_gap and existing_gap.reason.strip() else reason),
            suggested_queries=suggested_queries,
        )
        global_queries.extend(suggested_queries)
        claims.append(replace(claim, gap=reason))
    return replace(
        report,
        required_claims=tuple(claims),
        gaps=tuple(gaps_by_id.values()),
        suggested_queries=tuple(dict.fromkeys(query for query in global_queries if query)),
    )


def _normalize_issues_against_claims(
    issues: Sequence[ClaimCoverageIssue],
    claims: Sequence[ClaimCoverageClaim],
) -> tuple[ClaimCoverageIssue, ...]:
    claims_by_id = {claim.claim_id: claim for claim in claims}
    normalized: list[ClaimCoverageIssue] = []
    for issue in issues:
        issue_claims = [
            claims_by_id[claim_id]
            for claim_id in issue.claim_ids
            if claim_id in claims_by_id
        ]
        negative_support_claim_ids = {
            claim.claim_id
            for claim in issue_claims
            if _claim_contradiction_supports_negative_issue(claim, issue)
        }
        unresolved = tuple(
            claim.claim_id
            for claim in issue_claims
            if claim.required
            and claim.decisive
            and claim.status != "supported"
            and claim.claim_id not in negative_support_claim_ids
        )
        if issue_claims and not unresolved:
            status: ClaimStatus = "supported"
        elif any(claim.status == "contradicted" for claim in issue_claims):
            status = "contradicted"
        elif any(claim.status == "partially_supported" for claim in issue_claims):
            status = "partially_supported"
        elif any(claim.status == "unsupported" for claim in issue_claims):
            status = "unsupported"
        else:
            status = issue.status
        supporting_evidence_ids = tuple(
            dict.fromkeys(
                [
                    *issue.supporting_evidence_ids,
                    *(
                        check.evidence_id
                        for claim in issue_claims
                        for check in claim.evidence_checks
                        if check.entailment in {
                            "supports",
                            "partially_supports",
                        }
                        or (
                            claim.claim_id in negative_support_claim_ids
                            and check.entailment == "contradicts"
                        )
                    ),
                ]
            )
        )
        normalized.append(
            ClaimCoverageIssue(
                issue_id=issue.issue_id,
                question=issue.question,
                status=status,
                polarity=issue.polarity,
                supported_answer=issue.supported_answer,
                claim_ids=issue.claim_ids,
                supporting_evidence_ids=supporting_evidence_ids,
                unresolved_claim_ids=unresolved,
                inference_reason=issue.inference_reason,
                gap="" if status == "supported" else issue.gap,
            )
        )
    return tuple(normalized)


def _claim_contradiction_supports_negative_issue(
    claim: ClaimCoverageClaim,
    issue: ClaimCoverageIssue,
) -> bool:
    if issue.polarity != "negative":
        return False
    if not claim.required or not claim.decisive:
        return False
    if claim.polarity == "negative":
        return False
    return claim.status == "contradicted" or any(
        check.entailment == "contradicts" and check.relevance in {"high", "medium"}
        for check in claim.evidence_checks
    )


def _claim_signature(claim: ClaimCoverageClaim) -> str:
    return json.dumps(
        {
            "question": normalize_card_name(claim.question),
            "branch": claim.branch,
            "claim_type": claim.claim_type,
            "semantic_role": normalize_card_name(claim.semantic_role or claim.claim),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _unique_stable_id(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    index = 2
    while f"{base}.{index}" in used:
        index += 1
    return f"{base}.{index}"


def _evidence_check_from_mapping(
    payload: Any,
    *,
    allowed_evidence_ids: set[str],
) -> ClaimEvidenceCheck:
    if not isinstance(payload, Mapping):
        raise ValueError("Evidence check must be an object.")
    evidence_id = _non_empty_string(payload, "evidence_id")
    if evidence_id not in allowed_evidence_ids:
        raise ValueError(f"Claim evidence check cites unknown evidence id: {evidence_id}.")
    raw_relevance = str(payload.get("relevance") or "")
    if raw_relevance not in EVIDENCE_RELEVANCE_VALUES:
        raise ValueError(f"Invalid evidence relevance for {evidence_id}: {raw_relevance}.")
    raw_entailment = str(payload.get("entailment") or "")
    if raw_entailment not in EVIDENCE_ENTAILMENT_VALUES:
        raise ValueError(f"Invalid evidence entailment for {evidence_id}: {raw_entailment}.")
    return ClaimEvidenceCheck(
        evidence_id=evidence_id,
        relevance=raw_relevance,  # type: ignore[arg-type]
        entailment=raw_entailment,  # type: ignore[arg-type]
        reason=str(payload.get("reason") or ""),
    )


def _gap_from_mapping(payload: Any, *, claim_ids: set[str]) -> ClaimCoverageGap:
    if not isinstance(payload, Mapping):
        raise ValueError("Claim gap must be an object.")
    claim_id = _non_empty_string(payload, "claim_id")
    if claim_id not in claim_ids:
        raise ValueError(f"Claim gap references unknown claim id: {claim_id}.")
    return ClaimCoverageGap(
        claim_id=claim_id,
        reason=str(payload.get("reason") or ""),
        suggested_queries=tuple(
            str(item).strip()
            for item in _list_value(payload, "suggested_queries")
            if str(item).strip()
        ),
    )


def _list_value(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array.")
    return value


def _non_empty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value.strip()


def aggregate_claim_coverage(
    report: ClaimCoverageReport,
    *,
    evidence_by_id: Mapping[str, EvidenceItem] | None = None,
) -> ClaimCoverageAggregation:
    claims_by_id = {claim.claim_id: claim for claim in report.required_claims}
    issues = _normalize_issues_against_claims(
        report.issues or _issues_from_claims(report.required_claims),
        report.required_claims,
    )
    supported_issue_ids = [
        issue.issue_id for issue in issues if issue.status == "supported"
    ]
    negative_issue_ids = [
        issue.issue_id
        for issue in issues
        if issue.status == "supported" and issue.polarity == "negative"
    ]
    unresolved_issue_ids = [
        issue.issue_id
        for issue in issues
        if issue.status != "supported" or issue.unresolved_claim_ids
    ]
    supported_negative_issue_by_id = {
        issue.issue_id
        for issue in issues
        if issue.status == "supported" and issue.polarity == "negative"
    }
    unsupported: list[str] = []
    contradicted: list[str] = []
    unknown: list[str] = []
    partial: list[str] = []
    dependency_blocks: list[str] = []
    inferred_supported: set[str] = set()
    for claim in report.required_claims:
        if not claim.required or not claim.decisive:
            continue
        dependencies_supported = bool(claim.depends_on) and all(
            (dependency := claims_by_id.get(dependency_id)) is not None
            and dependency.status == "supported"
            for dependency_id in claim.depends_on
        )
        dependency_inferred = (
            claim.claim_type == "hypothetical_branch"
            and claim.status == "supported"
            and dependencies_supported
            and bool(claim.inference_reason.strip())
        )
        if dependency_inferred:
            inferred_supported.add(claim.claim_id)
        has_supporting_check = any(
            check.entailment == "supports" for check in claim.evidence_checks
        ) or dependency_inferred
        supporting_evidence = [
            evidence_by_id[check.evidence_id]
            for check in claim.evidence_checks
            if evidence_by_id is not None
            and check.evidence_id in evidence_by_id
            and check.entailment == "supports"
        ]
        has_required_evidence_kind = True
        if claim.status == "supported":
            if claim.claim_type in {"ruling_fact", "case_application"}:
                has_required_evidence_kind = evidence_by_id is not None and any(
                    item.kind in MECHANISM_EVIDENCE_KINDS
                    for item in supporting_evidence
                )
            elif evidence_by_id is not None and claim.claim_type == "card_text":
                has_required_evidence_kind = any(
                    item.kind == "card_effect" for item in supporting_evidence
                )
            elif evidence_by_id is not None and claim.claim_type == "hypothetical_branch":
                has_required_evidence_kind = dependency_inferred or any(
                    item.kind in MECHANISM_EVIDENCE_KINDS
                    for item in supporting_evidence
                )
        has_contradicting_check = any(
            check.entailment == "contradicts" for check in claim.evidence_checks
        )
        if claim.status == "unsupported":
            unsupported.append(claim.claim_id)
        elif claim.status == "contradicted":
            if claim.issue_id in supported_negative_issue_by_id:
                continue
            contradicted.append(claim.claim_id)
        elif claim.status == "unknown":
            unknown.append(claim.claim_id)
        elif claim.status == "partially_supported":
            partial.append(claim.claim_id)
        elif claim.status == "supported" and not has_supporting_check:
            unsupported.append(claim.claim_id)
        elif claim.status == "supported" and not has_required_evidence_kind:
            unsupported.append(claim.claim_id)
        if has_contradicting_check:
            if claim.issue_id not in supported_negative_issue_by_id:
                contradicted.append(claim.claim_id)
        for dependency_id in claim.depends_on:
            dependency = claims_by_id.get(dependency_id)
            if dependency is not None and dependency.status != "supported":
                dependency_blocks.append(claim.claim_id)
                break
    blocking = tuple(dict.fromkeys([*unsupported, *contradicted, *unknown, *dependency_blocks]))
    unresolved_issue_ids = [
        issue_id
        for issue_id in unresolved_issue_ids
        if not any(
            claim.issue_id == issue_id and claim.claim_id in inferred_supported
            for claim in report.required_claims
        )
    ]
    suggested_queries = list(report.suggested_queries)
    for gap in report.gaps:
        suggested_queries.extend(gap.suggested_queries)
    sufficient = not blocking and not partial and not unresolved_issue_ids
    warnings: list[str] = []
    if contradicted:
        warnings.append("Contradicted claims block confident adjudication.")
    if unsupported or unknown:
        warnings.append("Unsupported or unknown claims require more evidence.")
    if partial:
        warnings.append("Partially supported claims require uncertainty or more evidence.")
    if any(
        claim.status == "supported"
        and claim.required
        and claim.decisive
        and not any(check.entailment == "supports" for check in claim.evidence_checks)
        for claim in report.required_claims
    ):
        warnings.append("Supported claims must include entailing evidence checks.")
    if evidence_by_id is not None and any(
        claim.status == "supported"
        and claim.required
        and claim.decisive
        and claim.claim_type in {"ruling_fact", "case_application", "hypothetical_branch"}
        and claim.claim_id in unsupported
        for claim in report.required_claims
    ):
        warnings.append("Ruling/application claims require FAQ, supplement, or rules evidence; card text alone is not enough.")
    if negative_issue_ids:
        warnings.append("Coverage includes supported negative adjudication issue outcomes.")
    if inferred_supported:
        warnings.append("Some application claims are supported by explicit dependency inference.")
    issue_polarities: dict[str, set[str]] = {}
    for issue in issues:
        if issue.status == "supported":
            issue_polarities.setdefault(issue.issue_id, set()).add(issue.polarity)
    hard_contradicted_issues = sorted(
        issue_id
        for issue_id, polarities in issue_polarities.items()
        if "affirmative" in polarities and "negative" in polarities
    )
    if hard_contradicted_issues:
        warnings.append(
            "Mutually incompatible issue outcomes are both supported: "
            + ", ".join(hard_contradicted_issues)
        )
        unresolved_issue_ids.extend(hard_contradicted_issues)
        sufficient = False
    return ClaimCoverageAggregation(
        sufficient=sufficient,
        blocking_claims=blocking,
        supported_issues=tuple(dict.fromkeys(supported_issue_ids)),
        negative_supported_issues=tuple(dict.fromkeys(negative_issue_ids)),
        unresolved_issues=tuple(dict.fromkeys(unresolved_issue_ids)),
        partial_claims=tuple(dict.fromkeys(partial)),
        unknown_claims=tuple(dict.fromkeys(unknown)),
        contradicted_claims=tuple(dict.fromkeys(contradicted)),
        unsupported_claims=tuple(dict.fromkeys(unsupported)),
        suggested_queries=tuple(dict.fromkeys(query for query in suggested_queries if query)),
        warnings=tuple(warnings),
    )


def parse_card_selection_response(
    text: str,
    *,
    allowed_ids: set[int],
) -> tuple[int | None, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM card selection output must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM card selection output must be a JSON object.")
    if not set(payload).issubset({"selected_card_id", "reason"}):
        raise ValueError("LLM card selection output contains unexpected fields.")
    raw_id = payload.get("selected_card_id")
    if raw_id is None or raw_id == "":
        return None, str(payload.get("reason") or "")
    try:
        selected_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM selected_card_id must be an integer or null.") from exc
    if selected_id not in allowed_ids:
        raise ValueError(f"LLM selected unknown card id: {selected_id}.")
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("LLM card selection must explain how the candidate matches the mention.")
    return selected_id, reason


def build_card_name_extraction_prompt(scenario_text: str) -> str:
    schema = {
        "mentions": [
            {
                "text": "可能的卡名字符串，不要填写卡片 id",
                "context": "这段文字在原问题附近的上下文，可为空",
            }
        ]
    }
    return "\n".join(
        [
            "你是游戏王 OCG 卡名抽取助手。",
            "从用户场景中抽取可能的卡片名称字符串。",
            "只抽取字符串，不要判断裁定，不要返回卡片 id。",
            "只返回严格 JSON，不要使用 Markdown 或额外说明。",
            "如果不确定，也可以列出可能的卡名片段；后续会由本地卡库检索确认。",
            "",
            "输出 JSON schema：",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "用户场景：",
            scenario_text,
        ]
    )


def build_card_selection_prompt(
    mention: str,
    candidates: Sequence[CardCandidate],
) -> str:
    payload = [
        {
            "card_id": candidate.card_id,
            "name": candidate.name,
            "score": candidate.score,
            "source": candidate.source,
            "match_reason": candidate.match_reason,
            "effect_text": candidate.effect_text,
        }
        for candidate in candidates
    ]
    schema = {
        "selected_card_id": "从候选 card_id 中选择一个；无法判断则为 null",
        "reason": "简短说明选择理由",
    }
    return "\n".join(
        [
            "你是游戏王 OCG 卡名消歧助手。",
            "只能在给定候选中选择，不得引入候选外的卡。",
            "只返回严格 JSON，不要使用 Markdown 或额外说明。",
            "",
            "待匹配字符串：",
            mention,
            "",
            "候选：",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "输出 JSON schema：",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ]
    )


def search_card_candidates(
    query: str,
    cards: Sequence[Card],
    *,
    top_k: int = 5,
    retriever: SparseCardRetriever | None = None,
    verified_aliases: Mapping[str, tuple[int, str]] | None = None,
) -> tuple[CardCandidate, ...]:
    normalized_query = normalize_card_name(query)
    candidates: dict[int, CardCandidate] = {}
    alias = (verified_aliases or {}).get(normalized_query)
    if alias is not None:
        alias_card_id, alias_source = alias
        for card in cards:
            if card.card_id != alias_card_id:
                continue
            normalized_name = normalize_card_name(card.name)
            candidates[card.card_id] = _validated_card_candidate(
                CardCandidate(
                    mention=query,
                    card_id=card.card_id,
                    name=card.name,
                    score=1.0,
                    source="verified_alias",
                    match_reason="verified_alias",
                    effect_text=card.description,
                    normalized_mention=normalized_query,
                    normalized_name=normalized_name,
                ),
                status="accepted",
                reason=f"verified alias from {alias_source}",
                source="alias",
            )
            break
    for card in cards:
        normalized_name = normalize_card_name(card.name)
        if normalized_query and normalized_name == normalized_query:
            candidates[card.card_id] = CardCandidate(
                mention=query,
                card_id=card.card_id,
                name=card.name,
                score=1.0,
                source="card_lookup",
                match_reason="normalized_name",
                effect_text=card.description,
                normalized_mention=normalized_query,
                normalized_name=normalized_name,
            )
        elif normalized_query and (
            normalized_query in normalized_name or normalized_name in normalized_query
        ):
            candidates.setdefault(
                card.card_id,
                CardCandidate(
                    mention=query,
                    card_id=card.card_id,
                    name=card.name,
                    score=0.92,
                    source="card_lookup",
                    match_reason="normalized_substring",
                    effect_text=card.description,
                    normalized_mention=normalized_query,
                    normalized_name=normalized_name,
                ),
            )

    if candidates:
        ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.card_id))
        return tuple(ordered[:top_k])

    if len(normalized_query) <= 24:
        retriever = retriever or SparseCardRetriever.from_cards(cards)
        for rank, result in enumerate(retriever.search(query, top_k=max(top_k, 10)), start=1):
            name = str(result.metadata.get("name") or "")
            normalized_name = normalize_card_name(name)
            if normalized_query and (
                normalized_query in normalized_name or normalized_name in normalized_query
            ):
                score = 0.9
                reason = "rag_name_overlap"
            else:
                score = max(0.15, 0.78 - rank * 0.04)
                reason = "sparse_rag"
            existing = candidates.get(result.card_id)
            if existing is None or score > existing.score:
                candidates[result.card_id] = CardCandidate(
                    mention=query,
                    card_id=result.card_id,
                    name=name,
                    score=score,
                    source="sparse",
                    match_reason=reason,
                    effect_text=str(result.metadata.get("desc") or result.metadata.get("text") or ""),
                    normalized_mention=normalized_query,
                    normalized_name=normalized_name,
                )
    ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.card_id))
    return tuple(ordered[:top_k])


def _validated_card_candidate(
    candidate: CardCandidate,
    *,
    status: CandidateValidationStatus,
    reason: str,
    source: str = "rules",
) -> CardCandidate:
    return CardCandidate(
        mention=candidate.mention,
        card_id=candidate.card_id,
        name=candidate.name,
        score=candidate.score,
        source=candidate.source,
        match_reason=candidate.match_reason,
        effect_text=candidate.effect_text,
        normalized_mention=candidate.normalized_mention or normalize_card_name(candidate.mention),
        normalized_name=candidate.normalized_name or normalize_card_name(candidate.name),
        validation_status=status,
        validation_reason=reason,
        validation_source=source,
    )


def validate_card_candidate(candidate: CardCandidate) -> CardCandidate:
    normalized_mention = candidate.normalized_mention or normalize_card_name(candidate.mention)
    normalized_name = candidate.normalized_name or normalize_card_name(candidate.name)
    if not normalized_mention or not normalized_name:
        return _validated_card_candidate(
            candidate,
            status="rejected",
            reason="missing normalized mention or candidate name",
        )
    if candidate.match_reason == "verified_alias":
        return _validated_card_candidate(
            candidate,
            status="accepted",
            reason=candidate.validation_reason if candidate.validation_status == "accepted" else "verified alias",
            source="alias",
        )
    if normalized_mention == normalized_name:
        return _validated_card_candidate(
            candidate,
            status="accepted",
            reason="exact normalized name match",
        )
    if len(normalized_mention) < SHORT_MENTION_NORMALIZED_LENGTH:
        return _validated_card_candidate(
            candidate,
            status="rejected",
            reason="short mention requires exact name or verified alias",
        )
    if normalized_mention in normalized_name or normalized_name in normalized_mention:
        overlap = min(len(normalized_mention), len(normalized_name))
        coverage = overlap / max(len(normalized_mention), len(normalized_name), 1)
        if (
            overlap >= SUBSTRING_MIN_NORMALIZED_LENGTH
            and coverage >= SUBSTRING_MIN_COVERAGE
        ):
            return _validated_card_candidate(
                candidate,
                status="accepted",
                reason=(
                    "substring coverage rule "
                    f"(overlap={overlap}, coverage={coverage:.2f})"
                ),
            )
        return _validated_card_candidate(
            candidate,
            status="rejected",
            reason=(
                "substring overlap below coverage rule "
                f"(overlap={overlap}, coverage={coverage:.2f})"
            ),
        )
    return _validated_card_candidate(
        candidate,
        status="needs_validation",
        reason="retrieval candidate has no lexical or alias compatibility",
    )


def select_card_resolution(
    mention: str,
    candidates: Sequence[CardCandidate],
    *,
    confidence_threshold: float,
    ambiguity_margin: float,
) -> CardResolution:
    ordered = tuple(
        sorted(
            (validate_card_candidate(candidate) for candidate in candidates),
            key=lambda item: (-item.score, item.card_id),
        )
    )
    accepted = tuple(candidate for candidate in ordered if candidate.validation_status == "accepted")
    if not ordered:
        return CardResolution(
            mention=mention,
            status="unresolved",
            candidates=ordered,
            selection_source="rules",
            selection_reason="no candidates",
        )
    if not accepted:
        status: CandidateStatus = (
            "ambiguous"
            if any(candidate.validation_status == "needs_validation" for candidate in ordered)
            else "unresolved"
        )
        return CardResolution(
            mention=mention,
            status=status,
            candidates=ordered,
            selection_source="rules",
            selection_reason="no candidate passed absolute validity gate",
        )
    if accepted[0].score < confidence_threshold:
        return CardResolution(
            mention=mention,
            status="unresolved",
            candidates=ordered,
            selection_source="rules",
            selection_reason="top valid candidate below confidence threshold",
        )
    strong_matches = tuple(
        candidate
        for candidate in accepted
        if candidate.match_reason in {"normalized_name", "verified_alias"}
        and candidate.score >= confidence_threshold
    )
    if len(strong_matches) == 1:
        return CardResolution(
            mention=mention,
            status="resolved",
            candidates=ordered,
            selected_card_id=strong_matches[0].card_id,
            selection_source="rules",
            selection_reason=strong_matches[0].validation_reason,
        )
    if len(accepted) > 1 and accepted[0].score - accepted[1].score <= ambiguity_margin:
        return CardResolution(
            mention=mention,
            status="ambiguous",
            candidates=ordered,
            selection_source="rules",
            selection_reason="top valid candidates within ambiguity margin",
        )
    return CardResolution(
        mention=mention,
        status="resolved",
        candidates=ordered,
        selected_card_id=accepted[0].card_id,
        selection_source="rules",
        selection_reason=accepted[0].validation_reason,
    )


class AdjudicationAgent:
    def __init__(
        self,
        cards: Sequence[Card],
        *,
        settings: Settings,
        ruling_index: YgocdbRulingIndex | None = None,
        rule_index: RuleIndex | None = None,
        ruling_retrievers: Sequence[PersistentRulingRetriever] = (),
        llm: Llm | None = None,
        verified_card_aliases: Mapping[str, tuple[int, str]] | None = None,
        progress_callback: AdjudicationProgressCallback | None = None,
    ) -> None:
        self.cards = list(cards)
        self.cards_by_id = {card.card_id: card for card in self.cards}
        self.card_retriever = SparseCardRetriever.from_cards(self.cards)
        self.settings = settings
        self.ruling_index = ruling_index
        self.rule_index = rule_index
        self.ruling_retrievers = tuple(ruling_retrievers)
        self.llm = llm
        self.progress_callback = progress_callback
        raw_aliases = dict(DEFAULT_VERIFIED_CARD_ALIASES)
        raw_aliases.update(verified_card_aliases or {})
        self.card_alias_mentions = tuple(raw_aliases)
        self.verified_card_aliases = {
            normalize_card_name(alias): (card_id, source)
            for alias, (card_id, source) in raw_aliases.items()
        }
        self._compiled_graph = self._compile_graph()

    def run(self, scenario_text: str) -> AdjudicationResponse:
        initial = self._replace(
            create_initial_agent_state(scenario_text),
            rule_chapter_catalog=tuple(
                entry.to_dict()
                for entry in (
                    self.rule_index.chapter_catalog()
                    if self.rule_index is not None
                    else ()
                )
            ),
        )
        if self.llm is not None:
            state = self._run_agent_loop(initial)
        elif self._compiled_graph is None:
            state = self._run_sequential(initial)
        else:
            result = self._compiled_graph.invoke({"agent_state": initial})
            state = result["agent_state"]
        scenario = state.scenario or ParsedScenario(original_text=scenario_text)
        simulation = SimulationStatus(status="not_requested")
        reasoning = (
            ReasoningStep(
                (
                    "Agent loop selected adjudication tools and gathered evidence."
                    if state.tool_history
                    else "LangGraph adjudication agent gathered evidence and checked coverage."
                ),
                tuple(item.evidence_id for item in state.evidence[:5]),
            ),
        )
        answer = state.final_answer or self._deterministic_answer(state)
        confidence = "medium" if state.coverage.sufficient else "low"
        warnings = tuple(dict.fromkeys([*state.warnings, *self._coverage_warnings(state.coverage)]))
        agent_coverage = state.coverage.to_dict()
        if state.claim_coverage_report is not None:
            agent_coverage["claim_coverage_report"] = state.claim_coverage_report.to_dict()
        if state.claim_coverage_aggregation is not None:
            agent_coverage["claim_coverage_aggregation"] = (
                state.claim_coverage_aggregation.to_dict()
            )
        return AdjudicationResponse(
            scenario=scenario,
            resolved_cards=state.resolved_cards,
            evidence=state.evidence,
            reasoning=reasoning,
            answer=answer,
            confidence=confidence,
            warnings=warnings,
            simulation=simulation,
            agent_coverage=agent_coverage,
            agent_trace=tuple(entry.to_dict() for entry in state.trace),
        )

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None
        graph = StateGraph(AgentGraphState)
        graph.add_node("extract", self._node_extract)
        graph.add_node("resolve_cards", self._node_resolve_cards)
        graph.add_node("retrieve_evidence", self._node_retrieve_evidence)
        graph.add_node("check_coverage", self._node_check_coverage)
        graph.add_node("targeted_retrieval", self._node_targeted_retrieval)
        graph.add_node("final_answer", self._node_final_answer)
        graph.set_entry_point("extract")
        graph.add_edge("extract", "resolve_cards")
        graph.add_edge("resolve_cards", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "check_coverage")
        graph.add_conditional_edges(
            "check_coverage",
            self._route_after_coverage,
            {"retry": "targeted_retrieval", "final": "final_answer"},
        )
        graph.add_edge("targeted_retrieval", "check_coverage")
        graph.add_edge("final_answer", END)
        return graph.compile()

    def _run_sequential(self, state: AdjudicationAgentState) -> AdjudicationAgentState:
        payload = {"agent_state": state}
        for node in (
            self._node_extract,
            self._node_resolve_cards,
            self._node_retrieve_evidence,
            self._node_check_coverage,
        ):
            payload = node(payload)
        while (
            self._route_after_coverage(payload) == "retry"
            and payload["agent_state"].iterations
            <= self.settings.adjudication_agent_max_iterations
        ):
            payload = self._node_targeted_retrieval(payload)
            payload = self._node_check_coverage(payload)
        payload = self._node_final_answer(payload)
        return payload["agent_state"]

    def _run_agent_loop(self, state: AdjudicationAgentState) -> AdjudicationAgentState:
        state = self._initialize_agent_loop_state(state)
        max_steps = max(
            1,
            self.settings.adjudication_agent_total_turn_limit,
            min(self.settings.adjudication_agent_max_iterations, 1),
        )
        invalid_actions = 0
        budget_exhausted = ""
        for step in range(1, max_steps + 1):
            try:
                action = self._plan_agent_action(state)
            except ValueError as exc:
                invalid_actions += 1
                observation = ToolObservation(
                    tool="agent_plan",
                    ok=False,
                    summary="planner output rejected",
                    validation_error=str(exc),
                    warnings=(str(exc),),
                )
                action = PlannerAction(
                    tool="critique_coverage",
                    arguments={},
                    rationale="planner output was invalid",
                )
                state = self._record_tool_step(
                    state,
                    step=step,
                    action=action,
                    observation=observation,
                    duplicate=False,
                )
                state = self._warn(state, str(exc))
                if invalid_actions >= 2:
                    budget_exhausted = "invalid_planner_output"
                    break
                continue
            self._emit_progress(
                "planner",
                f"规划器选择工具：{action.tool}",
                tool=action.tool,
            )
            preparation_observation = self._preparation_observation(state, action)
            forced_observation = self._submit_required_observation(state, action)
            budget_observation = self._budget_observation(state, action)
            signature = _tool_signature_for_state(action, state)
            duplicate = signature in state.tool_signatures
            if action.tool == "critique_coverage" and duplicate:
                duplicate = _last_critique_revision(state) == evidence_revision(state)
            repeated_critique_observation = self._repeated_critique_observation(
                state,
                action,
            )
            warnings = (f"Duplicate tool call: {action.tool}.",) if duplicate else ()
            state = self._trace(
                self._replace(
                    state,
                    planner_notes=(*state.planner_notes, *action.notes),
                    hypotheses=(*state.hypotheses, *action.hypotheses),
                    tool_signatures=(*state.tool_signatures, signature),
                    iterations=step,
                ),
                "agent_plan",
                f"selected {action.tool}",
                {
                    "step": step,
                    "action": action.to_dict(),
                    "duplicate": duplicate,
                    "warnings": list(warnings),
                },
            )
            if preparation_observation is not None:
                state, observation = state, preparation_observation
            elif repeated_critique_observation is not None:
                state, observation = (
                    self._warn(
                        state,
                        repeated_critique_observation.validation_error,
                    ),
                    repeated_critique_observation,
                )
            elif budget_observation is not None:
                state, observation = self._warn(state, budget_observation.validation_error), budget_observation
                budget_exhausted = budget_observation.data.get("budget", "unknown") if isinstance(budget_observation.data, Mapping) else "unknown"
            elif forced_observation is not None:
                state, observation = self._warn(state, forced_observation.validation_error), forced_observation
            elif (
                gap_observation := self._claim_gap_target_observation(
                    state,
                    action,
                    duplicate=duplicate,
                )
            ) is not None:
                state, observation = self._warn(state, gap_observation.validation_error), gap_observation
            else:
                state, observation = self._execute_agent_tool(state, action)
                if action.tool != "submit_ruling":
                    coverage = self._check_coverage(state)
                    state = self._replace(state, coverage=coverage)
                    self._emit_coverage(coverage)
            if warnings:
                observation = ToolObservation(
                    tool=observation.tool,
                    ok=observation.ok,
                    summary=observation.summary,
                    data=observation.data,
                    evidence_ids=observation.evidence_ids,
                    warnings=(*observation.warnings, *warnings),
                    validation_error=observation.validation_error,
                )
                for warning in warnings:
                    state = self._warn(state, warning)
            state = self._record_tool_step(
                state,
                step=step,
                action=action,
                observation=observation,
                duplicate=duplicate,
            )
            if action.tool == "submit_ruling" and observation.ok and state.agent_loop_complete:
                return state
            if preparation_observation is None and budget_observation is not None:
                break
        if state.final_answer is None:
            if not budget_exhausted:
                budget_exhausted = "total_turn_limit"
            state = self._warn(
                state,
                f"Agent loop ended without accepted submit_ruling; using fallback answer. Exhausted budget: {budget_exhausted}.",
            )
            coverage = self._check_coverage(state)
            self._emit_coverage(coverage)
            state = self._replace(
                state,
                coverage=coverage,
                final_answer=self._deterministic_answer(self._replace(state, coverage=coverage)),
            )
            state = self._trace(
                state,
                "agent_loop_fallback",
                "used deterministic fallback answer",
                {"coverage": coverage.to_dict(), "exhausted_budget": budget_exhausted},
            )
            self._emit_progress(
                "fallback",
                "未能提交有效裁定，已生成保守回退答案。",
                evidence_count=len(state.evidence),
                sufficient=coverage.sufficient,
                fallback=True,
            )
        return state

    def _budget_observation(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
    ) -> ToolObservation | None:
        retrieval_tools = {
            "search_card",
            "search_direct_faq",
            "search_related_faq",
            "search_card_supplement",
            "search_analogous_faq",
            "classify_rule_chapters",
            "search_rules",
        }
        counts = _tool_budget_counts(state.tool_history)
        budget_name = ""
        limit = 0
        used = 0
        if action.tool in retrieval_tools:
            budget_name = "retrieval"
            limit = max(1, self.settings.adjudication_agent_retrieval_limit)
            used = counts["retrieval"]
        elif action.tool == "critique_coverage":
            budget_name = "critique"
            limit = max(1, self.settings.adjudication_agent_critique_limit)
            used = counts["critique"]
        elif action.tool == "submit_ruling":
            budget_name = "submit"
            limit = max(1, self.settings.adjudication_agent_submit_limit)
            used = counts["submit"]
        if not budget_name or used < limit:
            return None
        message = (
            f"Agent {budget_name} budget exhausted before {action.tool}: "
            f"{used}/{limit} calls already used."
        )
        return ToolObservation(
            tool=action.tool,
            ok=False,
            summary=f"{budget_name} budget exhausted",
            data={"budget": budget_name, "used": used, "limit": limit},
            validation_error=message,
            warnings=(message,),
        )

    def _initialize_agent_loop_state(self, state: AdjudicationAgentState) -> AdjudicationAgentState:
        scenario = ParsedScenario(original_text=state.scenario_text)
        field_sources = _initial_scenario_field_sources()
        self._emit_progress(
            "parse",
            "场景原文已载入，等待规划器调用问题拆分和卡名抽取工具。",
        )
        return self._trace(
            self._replace(
                state,
                scenario=scenario,
                agent_loop_preparation_required=True,
                scenario_field_sources=field_sources,
                scenario_revision=0,
                mention_queries=(),
                extracted_card_mentions=(),
                card_resolution_loop=CardResolutionLoopTrace(),
                card_resolutions=(),
                resolved_cards=(),
                evidence=(),
                evidence_retrieval_stages=(),
                coverage=CoverageReport(sufficient=False, next_action="prepare"),
                claim_coverage_report=None,
                claim_coverage_aggregation=None,
                submit_attempts=(),
                final_answer=None,
                agent_loop_complete=False,
                rule_chapter_catalog=(
                    state.rule_chapter_catalog
                    or tuple(
                        entry.to_dict()
                        for entry in (
                            self.rule_index.chapter_catalog()
                            if self.rule_index is not None
                            else ()
                        )
                    )
                ),
            ),
            "initialize_agent_loop",
            "initialized raw scenario with parser tools pending",
            {
                "scenario_source": "raw",
                "scenario_revision": 0,
                "scenario_field_sources": {
                    field_name: dict(provenance)
                    for field_name, provenance in field_sources.items()
                },
            },
        )

    def _plan_agent_action(self, state: AdjudicationAgentState) -> PlannerAction:
        assert self.llm is not None
        try:
            response = self.llm.invoke(build_agent_planner_prompt(state))
        except Exception as exc:
            raise ValueError(f"规划器 LLM 请求失败：{exc}") from exc
        return parse_agent_planner_response(str(getattr(response, "content", response)))

    def _submit_required_observation(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
    ) -> ToolObservation | None:
        if (
            not state.coverage.sufficient
            or action.tool == "submit_ruling"
            or action.tool in SCENARIO_PREPARATION_TOOLS
        ):
            return None
        message = (
            "Coverage is sufficient; the next action must be submit_ruling. "
            f"Rejected non-submit tool: {action.tool}."
        )
        return ToolObservation(
            tool=action.tool,
            ok=False,
            summary="submit_ruling required",
            validation_error=message,
            warnings=(message,),
        )

    def _preparation_observation(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
    ) -> ToolObservation | None:
        guarded_tools = {
            "resolve_cards",
            "search_direct_faq",
            "search_related_faq",
            "search_card_supplement",
            "search_analogous_faq",
            "search_rules",
            "critique_coverage",
            "submit_ruling",
        }
        if action.tool not in guarded_tools:
            return None
        missing = _mandatory_preparation_gaps(state)
        if not missing:
            return None
        message = (
            "Scenario preparation required before evidence retrieval, coverage critique, "
            f"or submission. Call: {', '.join(missing)}."
        )
        return ToolObservation(
            tool=action.tool,
            ok=False,
            summary="scenario preparation required",
            data={"missing_preparation": list(missing)},
            validation_error=message,
            warnings=(message,),
        )

    def _claim_gap_target_observation(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
        *,
        duplicate: bool,
    ) -> ToolObservation | None:
        if state.claim_coverage_report is None or state.claim_coverage_aggregation is None:
            return None
        if action.tool in {"submit_ruling", "critique_coverage", "resolve_cards"}:
            return None
        retrieval_tools = {
            "search_card",
            "search_direct_faq",
            "search_related_faq",
            "search_card_supplement",
            "search_analogous_faq",
            "classify_rule_chapters",
            "search_rules",
        }
        if action.tool not in retrieval_tools:
            return None
        if duplicate:
            message = (
                "Repeated retrieval after Claim Coverage Report was rejected; "
                "the planner must target an unresolved claim gap with a new query."
            )
            return ToolObservation(
                tool=action.tool,
                ok=False,
                summary="claim gap targeted retrieval required",
                validation_error=message,
                warnings=(message,),
            )
        if action.tool == "search_card" and (
            state.coverage.missing_cards or state.coverage.ambiguous_cards
        ):
            return None
        if action.tool in {
            "search_direct_faq",
            "search_related_faq",
            "search_card_supplement",
        } and action.arguments.get("card_ids") is not None:
            return None
        if _action_targets_claim_gap(state, action):
            return None
        message = (
            "Retrieval after Claim Coverage Report must target a reported claim gap "
            "or contradiction. Use the report's claim_id, gap text, or suggested query."
        )
        return ToolObservation(
            tool=action.tool,
            ok=False,
            summary="claim gap targeted retrieval required",
            validation_error=message,
            warnings=(message,),
        )

    def _repeated_critique_observation(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
    ) -> ToolObservation | None:
        if action.tool != "critique_coverage":
            return None
        if state.claim_coverage_report is None:
            return None
        current_revision = evidence_revision(state)
        if _last_critique_revision(state) != current_revision:
            return None
        message = (
            "Claim Coverage Report 已在相同 evidence revision 上完成审查；"
            "重复 critique 不会增加信息。请按 blocking claim 的 gap 生成新检索，"
            "证据已充分时提交，无法取得新证据时明确结束并报告缺口。"
        )
        return ToolObservation(
            tool="critique_coverage",
            ok=False,
            summary="unchanged-evidence critique rejected",
            data={
                "evidence_revision": current_revision,
                "claim_report_hash": _claim_coverage_report_hash(
                    state.claim_coverage_report
                ),
            },
            validation_error=message,
            warnings=(message,),
        )

    def _record_tool_step(
        self,
        state: AdjudicationAgentState,
        *,
        step: int,
        action: PlannerAction,
        observation: ToolObservation,
        duplicate: bool,
    ) -> AdjudicationAgentState:
        entry = ToolHistoryEntry(
            step=step,
            action=action,
            observation=observation,
            duplicate=duplicate,
        )
        self._emit_progress(
            "tool",
            (
                f"工具 {action.tool} 完成，获得 {len(observation.evidence_ids)} 条证据。"
                if observation.ok
                else f"工具 {action.tool} 未完成。"
            ),
            tool=action.tool,
            ok=observation.ok,
            evidence_count=len(observation.evidence_ids),
        )
        return self._trace(
            self._replace(state, tool_history=(*state.tool_history, entry)),
            "execute_tool",
            observation.summary,
            {
                "step": step,
                "action": action.to_dict(),
                "observation": observation.to_dict(),
                "duplicate": duplicate,
            },
        )

    def _execute_agent_tool(
        self,
        state: AdjudicationAgentState,
        action: PlannerAction,
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        if preparation_observation := self._preparation_observation(state, action):
            return state, preparation_observation
        try:
            if action.tool == "extract_card_mentions":
                return self._tool_extract_card_mentions(state, action.arguments)
            if action.tool == "split_questions":
                return self._tool_split_questions(state, action.arguments)
            if action.tool == "extract_scenario_facts":
                return self._tool_extract_scenario_facts(state, action.arguments)
            if action.tool == "resolve_cards":
                return self._tool_resolve_cards(state, action.arguments)
            if action.tool == "search_card":
                return self._tool_search_card(state, action.arguments)
            if action.tool == "search_direct_faq":
                return self._tool_search_ruling_evidence(
                    state,
                    action.arguments,
                    tool="search_direct_faq",
                    stage="direct_faq",
                    relationship="direct_card",
                    allowed_kinds={"faq"},
                )
            if action.tool == "search_related_faq":
                return self._tool_search_ruling_evidence(
                    state,
                    action.arguments,
                    tool="search_related_faq",
                    stage="related_faq",
                    relationship="related_card",
                    allowed_kinds={"faq"},
                )
            if action.tool == "search_card_supplement":
                return self._tool_search_ruling_evidence(
                    state,
                    action.arguments,
                    tool="search_card_supplement",
                    stage="supplement",
                    relationship="card_supplement",
                    allowed_kinds={"card_supplement"},
                )
            if action.tool == "search_analogous_faq":
                return self._tool_search_ruling_evidence(
                    state,
                    action.arguments,
                    tool="search_analogous_faq",
                    stage="analogous_faq",
                    relationship="analogous",
                    allowed_kinds={"faq"},
                )
            if action.tool == "classify_rule_chapters":
                return self._tool_classify_rule_chapters(state, action.arguments)
            if action.tool == "search_rules":
                return self._tool_search_ruling_evidence(
                    state,
                    action.arguments,
                    tool="search_rules",
                    stage="rules",
                    relationship="rule_document",
                    allowed_kinds={"rule_passage"},
                )
            if action.tool == "critique_coverage":
                return self._tool_critique_coverage(state)
            if action.tool == "submit_ruling":
                return self._tool_submit_ruling(state, action.arguments)
        except ValueError as exc:
            observation = ToolObservation(
                tool=action.tool,
                ok=False,
                summary=f"{action.tool} rejected arguments or output",
                validation_error=str(exc),
                warnings=(str(exc),),
            )
            return self._warn(state, str(exc)), observation
        observation = ToolObservation(
            tool=action.tool,
            ok=False,
            summary=f"unknown tool {action.tool}",
            validation_error=f"Unknown adjudication tool: {action.tool}.",
        )
        return state, observation

    def _tool_extract_card_mentions(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        strategy = str(arguments.get("strategy") or "llm")
        focus = str(arguments.get("focus") or "").strip()
        self._emit_progress(
            "parse_tool",
            f"开始抽取场景卡名（{strategy}）。",
            tool="extract_card_mentions",
        )
        raw_response = ""
        try:
            if strategy == "llm":
                if self.llm is None:
                    raise ValueError("LLM provider is unavailable.")
                prompt = build_card_mention_extraction_prompt(
                    state.scenario_text,
                    focus=focus,
                )
                response = self.llm.invoke(prompt)
                raw_response = str(getattr(response, "content", response))
                try:
                    payload = json.loads(raw_response)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Card mention extractor returned invalid JSON."
                    ) from exc
                mentions = validate_card_mention_extraction_payload(
                    payload,
                    state.scenario_text,
                    known_cards=self.cards,
                )
            else:
                deterministic_scenario = parse_scenario(
                    state.scenario_text,
                    known_cards=self.cards,
                )
                mentions = tuple(
                    dict.fromkeys(
                        [
                            *deterministic_scenario.card_mentions,
                            *candidate_mention_queries(
                                state.scenario_text,
                                deterministic_scenario,
                                alias_mentions=self.card_alias_mentions,
                            ),
                        ]
                    )
                )
        except Exception as exc:
            message = (
                f"{exc} Retry extract_card_mentions with a narrower focus or "
                "strategy=deterministic."
            )
            self._emit_progress(
                "parse_tool",
                "卡名抽取失败，可调整关注点或改用确定性策略重试。",
                tool="extract_card_mentions",
                ok=False,
            )
            return state, ToolObservation(
                tool="extract_card_mentions",
                ok=False,
                summary="card mention extraction failed without state mutation",
                data={
                    "strategy": strategy,
                    "focus": focus,
                    "llm_call": {
                        "prompt_kind": "card_mentions",
                        "response": raw_response,
                    }
                    if strategy == "llm"
                    else None,
                },
                validation_error=message,
                warnings=(message,),
            )

        scenario = state.scenario or ParsedScenario(original_text=state.scenario_text)
        changed = tuple(scenario.card_mentions) != tuple(mentions)
        revision = state.scenario_revision + 1
        source = _updated_scenario_field_sources(
            state,
            "card_mentions",
            tool="extract_card_mentions",
            strategy=strategy,
            focus=focus,
            revision=revision,
        )
        if changed:
            for dependent_field in ("objects", "replacement_candidates"):
                source[dependent_field] = {
                    "status": "pending",
                    "tool": "extract_scenario_facts",
                    "strategy": "",
                    "focus": "",
                    "revision": 0,
                }
        extracted = tuple(
            ExtractedCardMention(text=mention, context=focus, source=strategy)
            for mention in mentions
        )
        changes: dict[str, Any] = {
            "scenario": _scenario_replacing(
                scenario,
                card_mentions=tuple(mentions),
                objects=(() if changed else scenario.objects),
                replacement_candidates=(
                    () if changed else scenario.replacement_candidates
                ),
            ),
            "scenario_field_sources": source,
            "scenario_revision": revision,
            "mention_queries": tuple(mentions),
            "extracted_card_mentions": extracted,
            "card_resolution_loop": CardResolutionLoopTrace(
                extracted_mentions=extracted
            ),
        }
        if changed:
            changes.update(
                {
                    "card_resolutions": (),
                    "resolved_cards": (),
                    **_scenario_revision_reset(state),
                }
            )
        new_state = self._replace(state, **changes)
        self._emit_progress(
            "parse_tool",
            f"卡名抽取完成，共识别 {len(mentions)} 个片段。",
            tool="extract_card_mentions",
            ok=True,
        )
        return new_state, ToolObservation(
            tool="extract_card_mentions",
            ok=True,
            summary=f"replaced card mentions with {len(mentions)} source-grounded values",
            data={
                "mentions": list(mentions),
                "strategy": strategy,
                "focus": focus,
                "scenario_revision": revision,
                "changed": changed,
                "llm_call": {
                    "prompt_kind": "card_mentions",
                    "response": raw_response,
                }
                if strategy == "llm"
                else None,
            },
        )

    def _tool_split_questions(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        strategy = str(arguments.get("strategy") or "llm")
        requested_strategy = strategy
        focus = str(arguments.get("focus") or "").strip()
        self._emit_progress(
            "parse_tool",
            f"开始拆分裁定问题（{strategy}）。",
            tool="split_questions",
        )
        raw_response = ""
        try:
            if strategy == "llm":
                if self.llm is None:
                    raise ValueError("LLM provider is unavailable.")
                prompt = build_question_split_prompt(
                    state.scenario_text,
                    focus=focus,
                )
                response = self.llm.invoke(prompt)
                raw_response = str(getattr(response, "content", response))
                try:
                    payload = json.loads(raw_response)
                except json.JSONDecodeError as exc:
                    raise ValueError("Question splitter returned invalid JSON.") from exc
                questions = validate_question_split_payload(
                    payload,
                    state.scenario_text,
                )
            else:
                questions = split_questions_deterministically(state.scenario_text)
                questions = validate_question_split_payload(
                    {"questions": list(questions)},
                    state.scenario_text,
                )
        except Exception as exc:
            deterministic_questions = split_questions_deterministically(
                state.scenario_text
            )
            can_use_exact_source_fallback = (
                strategy == "llm"
                and bool(state.scenario_text.strip())
                and not (state.scenario and state.scenario.questions)
                and deterministic_questions == (state.scenario_text.strip(),)
            )
            if can_use_exact_source_fallback:
                questions = validate_question_split_payload(
                    {"questions": list(deterministic_questions)},
                    state.scenario_text,
                )
                strategy = "deterministic_fallback"
            else:
                message = (
                    f"{exc} Retry split_questions with a narrower focus or "
                    "strategy=deterministic."
                )
                self._emit_progress(
                    "parse_tool",
                    "问题拆分失败，可调整关注点或改用确定性策略重试。",
                    tool="split_questions",
                    ok=False,
                )
                return state, ToolObservation(
                    tool="split_questions",
                    ok=False,
                    summary="question splitting failed without state mutation",
                    data={
                        "strategy": strategy,
                        "focus": focus,
                        "llm_call": {
                            "prompt_kind": "questions",
                            "response": raw_response,
                        }
                        if requested_strategy == "llm"
                        else None,
                    },
                    validation_error=message,
                    warnings=(message,),
                )

        scenario = state.scenario or ParsedScenario(original_text=state.scenario_text)
        changed = tuple(scenario.questions) != tuple(questions)
        revision = state.scenario_revision + 1
        changes = {
            "scenario": _scenario_replacing(scenario, questions=tuple(questions)),
            "scenario_field_sources": _updated_scenario_field_sources(
                state,
                "questions",
                tool="split_questions",
                strategy=strategy,
                focus=focus,
                revision=revision,
            ),
            "scenario_revision": revision,
        }
        if changed:
            changes.update(_scenario_revision_reset(state))
        new_state = self._replace(state, **changes)
        self._emit_progress(
            "parse_tool",
            f"问题拆分完成，共识别 {len(questions)} 个问题。",
            tool="split_questions",
            ok=True,
        )
        return new_state, ToolObservation(
            tool="split_questions",
            ok=True,
            summary=f"replaced questions with {len(questions)} source-grounded values",
            data={
                "questions": list(questions),
                "strategy": strategy,
                "focus": focus,
                "scenario_revision": revision,
                "changed": changed,
                "llm_call": {
                    "prompt_kind": "questions",
                    "response": raw_response,
                }
                if requested_strategy == "llm"
                else None,
            },
        )

    def _tool_extract_scenario_facts(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        field_name = str(arguments["field"])
        strategy = str(arguments.get("strategy") or "llm")
        focus = str(arguments.get("focus") or "").strip()
        scenario = state.scenario or ParsedScenario(original_text=state.scenario_text)
        self._emit_progress(
            "parse_tool",
            f"开始抽取场景字段 {field_name}（{strategy}）。",
            tool="extract_scenario_facts",
        )
        raw_response = ""
        try:
            if strategy == "llm":
                if self.llm is None:
                    raise ValueError("LLM provider is unavailable.")
                prompt = build_scenario_fact_extraction_prompt(
                    state.scenario_text,
                    field_name=field_name,
                    card_mentions=scenario.card_mentions,
                    focus=focus,
                )
                response = self.llm.invoke(prompt)
                raw_response = str(getattr(response, "content", response))
                try:
                    payload = json.loads(raw_response)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Scenario fact extractor returned invalid JSON."
                    ) from exc
                values = validate_scenario_fact_payload(
                    payload,
                    state.scenario_text,
                    field_name=field_name,
                    card_mentions=scenario.card_mentions,
                )
            else:
                values = extract_scenario_facts_deterministically(
                    state.scenario_text,
                    field_name=field_name,
                    card_mentions=scenario.card_mentions,
                )
                values = validate_scenario_fact_payload(
                    {"values": list(values)},
                    state.scenario_text,
                    field_name=field_name,
                    card_mentions=scenario.card_mentions,
                )
        except Exception as exc:
            message = (
                f"{exc} Retry extract_scenario_facts for {field_name} with a narrower "
                "focus or strategy=deterministic."
            )
            self._emit_progress(
                "parse_tool",
                f"场景字段 {field_name} 抽取失败，可调整参数重试。",
                tool="extract_scenario_facts",
                ok=False,
            )
            return state, ToolObservation(
                tool="extract_scenario_facts",
                ok=False,
                summary=f"{field_name} extraction failed without state mutation",
                data={
                    "field": field_name,
                    "strategy": strategy,
                    "focus": focus,
                    "llm_call": {
                        "prompt_kind": f"scenario_fact:{field_name}",
                        "response": raw_response,
                    }
                    if strategy == "llm"
                    else None,
                },
                validation_error=message,
                warnings=(message,),
            )

        previous = tuple(getattr(scenario, field_name))
        changed = previous != tuple(values)
        revision = state.scenario_revision + 1
        changes = {
            "scenario": _scenario_replacing(
                scenario,
                **{field_name: tuple(values)},
            ),
            "scenario_field_sources": _updated_scenario_field_sources(
                state,
                field_name,
                tool="extract_scenario_facts",
                strategy=strategy,
                focus=focus,
                revision=revision,
            ),
            "scenario_revision": revision,
        }
        if changed:
            changes.update(_derived_state_reset())
        new_state = self._replace(state, **changes)
        self._emit_progress(
            "parse_tool",
            f"场景字段 {field_name} 抽取完成，共 {len(values)} 项。",
            tool="extract_scenario_facts",
            ok=True,
        )
        return new_state, ToolObservation(
            tool="extract_scenario_facts",
            ok=True,
            summary=f"replaced scenario field {field_name} with {len(values)} values",
            data={
                "field": field_name,
                "values": list(values),
                "strategy": strategy,
                "focus": focus,
                "scenario_revision": revision,
                "changed": changed,
                "llm_call": {
                    "prompt_kind": f"scenario_fact:{field_name}",
                    "response": raw_response,
                }
                if strategy == "llm"
                else None,
            },
        )

    def _tool_resolve_cards(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        raw_mentions = arguments.get("mentions")
        mentions = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in (
                    raw_mentions
                    if isinstance(raw_mentions, list) and raw_mentions
                    else state.mention_queries
                )
                if str(item).strip()
            )
        )
        warnings = list(state.card_resolution_loop.warnings)
        resolutions = tuple(self._resolve_card_mention(mention, warnings=warnings) for mention in mentions)
        merged = self._merge_card_resolutions(state.card_resolutions, resolutions)
        resolved = self._resolved_cards_from_resolutions(merged)
        loop_trace = CardResolutionLoopTrace(
            extracted_mentions=state.card_resolution_loop.extracted_mentions,
            retry_count=state.card_resolution_loop.retry_count,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        card_effect_evidence = build_card_effect_evidence(resolved)
        resolved_card_ids = [
            card.matches[0].card_id
            for card in resolved
            if len(card.matches) == 1
        ]
        resolved_ruling_evidence: list[EvidenceItem] = []
        resolved_ruling_stages: list[dict[str, Any]] = []
        for stage, relationship, allowed_kinds in (
            ("direct_faq", "direct_card", {"faq"}),
            ("related_faq", "related_card", {"faq"}),
            ("supplement", "card_supplement", {"card_supplement"}),
        ):
            evidence, stages = self._search_tool_evidence(
                state.scenario_text,
                card_ids=resolved_card_ids,
                stage=stage,
                relationship=relationship,
                allowed_kinds=allowed_kinds,
            )
            resolved_ruling_evidence.extend(evidence)
            resolved_ruling_stages.extend(stages)
        new_state = self._replace(
            state,
            card_resolutions=merged,
            resolved_cards=resolved,
            evidence=tuple(
                dedupe_evidence(
                    [*state.evidence, *card_effect_evidence, *resolved_ruling_evidence]
                )
            ),
            evidence_retrieval_stages=(
                *state.evidence_retrieval_stages,
                *resolved_ruling_stages,
            ),
            card_resolution_loop=loop_trace,
        )
        observation = ToolObservation(
            tool="resolve_cards",
            ok=True,
            summary=f"resolved {len(resolved)} cards from {len(mentions)} mentions",
            data={
                "mentions": list(mentions),
                "resolutions": [resolution.to_dict() for resolution in resolutions],
                "resolved_card_ids": [
                    card.matches[0].card_id for card in resolved if len(card.matches) == 1
                ],
                "card_effect_evidence_ids": [
                    item.evidence_id for item in card_effect_evidence
                ],
                "supplement_evidence_ids": [
                    item.evidence_id
                    for item in resolved_ruling_evidence
                    if item.kind == "card_supplement"
                ],
                "resolved_ruling_evidence_ids": [
                    item.evidence_id for item in resolved_ruling_evidence
                ],
            },
            evidence_ids=tuple(
                item.evidence_id
                for item in (*card_effect_evidence, *resolved_ruling_evidence)
            ),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        self._emit_progress(
            "card_resolution",
            f"卡片解析完成：{len(resolved)} 张已确认，{len(mentions) - len(resolved)} 项待确认。",
        )
        return new_state, observation

    def _tool_search_card(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        query = str(arguments["query"]).strip()
        top_k = int(arguments.get("top_k") or self.settings.adjudication_card_resolution_top_k)
        candidates = tuple(
            validate_card_candidate(candidate)
            for candidate in search_card_candidates(
                query,
                self.cards,
                top_k=top_k,
                retriever=self.card_retriever,
                verified_aliases=self.verified_card_aliases,
            )
        )
        observation = ToolObservation(
            tool="search_card",
            ok=True,
            summary=f"returned {len(candidates)} card candidates",
            data={
                "query": query,
                "candidates": [candidate.to_dict() for candidate in candidates],
                "candidate_only": True,
            },
        )
        return state, observation

    def _tool_classify_rule_chapters(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        query = str(arguments["query"]).strip()
        warning = _rules_query_card_name_warning(query, state)
        if warning:
            return state, ToolObservation(
                tool="classify_rule_chapters",
                ok=False,
                summary="rule chapter classification requires mechanism-focused rewrite",
                data={"query": query, "reason": warning},
                validation_error=warning,
                warnings=(warning,),
            )
        if self.rule_index is None:
            message = "OCG rules EPUB chapter index is unavailable."
            return state, ToolObservation(
                tool="classify_rule_chapters",
                ok=False,
                summary="rule chapter index unavailable",
                validation_error=message,
                warnings=(message,),
            )
        top_k = int(arguments.get("top_k") or 5)
        candidates = tuple(
            candidate.to_dict()
            for candidate in self.rule_index.classify_chapters(query, top_k=top_k)
        )
        new_state = self._replace(state, rule_chapter_candidates=candidates)
        return new_state, ToolObservation(
            tool="classify_rule_chapters",
            ok=True,
            summary=f"classified issue into {len(candidates)} candidate rule chapters",
            data={"query": query, "candidates": list(candidates)},
        )

    def _tool_search_ruling_evidence(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
        *,
        tool: str,
        stage: str,
        relationship: str,
        allowed_kinds: set[str],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        query = str(arguments.get("query") or state.scenario_text).strip()
        card_ids = _argument_card_ids(arguments, default=self._resolved_card_ids(state))
        chapter_ids = tuple(
            dict.fromkeys(
                str(item).strip()
                for item in arguments.get("chapter_ids", ())
                if str(item).strip()
            )
        )
        if tool == "search_rules":
            warning = _rules_query_card_name_warning(query, state)
            if warning:
                return state, ToolObservation(
                    tool=tool,
                    ok=False,
                    summary="rules query requires mechanism-focused rewrite",
                    data={"query": query, "reason": warning},
                    validation_error=warning,
                    warnings=(warning,),
                )
            if chapter_ids:
                known_ids = {
                    str(chapter.get("chapter_id") or "")
                    for chapter in state.rule_chapter_catalog
                    if str(chapter.get("chapter_id") or "")
                }
                if self.rule_index is not None:
                    known_ids.update(self.rule_index.sections_by_chapter)
                unknown_ids = set(chapter_ids).difference(known_ids)
                if unknown_ids:
                    message = (
                        "search_rules.chapter_ids contains ids outside the known EPUB catalog: "
                        + ", ".join(sorted(unknown_ids))
                    )
                    return state, ToolObservation(
                        tool=tool,
                        ok=False,
                        summary="rules chapter filter contains unknown ids",
                        data={
                            "query": query,
                            "chapter_ids": list(chapter_ids),
                            "allowed_chapter_ids": sorted(known_ids),
                        },
                        validation_error=message,
                        warnings=(message,),
                    )
        if relationship in {"direct_card", "related_card", "card_supplement"} and not card_ids:
            stage_info = {
                "stage": stage,
                "requested_card_ids": [],
                "query_summary": query[:240],
                "returned_evidence_ids": [],
                "relationships": [],
                "skipped": True,
                "reason": "no resolved card ids",
                "tool": tool,
            }
            new_state = self._replace(
                state,
                evidence_retrieval_stages=(*state.evidence_retrieval_stages, stage_info),
            )
            return new_state, ToolObservation(
                tool=tool,
                ok=True,
                summary=f"{tool} skipped; no resolved card ids",
                data=stage_info,
            )
        evidence, stages = self._search_tool_evidence(
            query,
            card_ids=card_ids,
            stage=stage,
            relationship=relationship,
            allowed_kinds=allowed_kinds,
            chapter_ids=chapter_ids,
        )
        deduped = tuple(dedupe_evidence([*state.evidence, *evidence]))
        new_state = self._replace(
            state,
            evidence=deduped,
            evidence_retrieval_stages=(*state.evidence_retrieval_stages, *stages),
        )
        return new_state, ToolObservation(
            tool=tool,
            ok=True,
            summary=f"{tool} added {len(evidence)} evidence items",
            data={
                "query": query,
                "card_ids": list(card_ids),
                "chapter_ids": list(chapter_ids),
                "stage": stage,
                "returned_evidence": [item.to_dict() for item in evidence],
                "stages": stages,
            },
            evidence_ids=tuple(item.evidence_id for item in evidence),
        )

    def _tool_critique_coverage(
        self,
        state: AdjudicationAgentState,
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        legacy_coverage = self._check_coverage(state)
        suggestions: list[str] = []
        if legacy_coverage.missing_cards or legacy_coverage.ambiguous_cards:
            suggestions.append("resolve_cards or search_card for unresolved/ambiguous mentions")
        if legacy_coverage.missing_evidence:
            suggestions.append("search_direct_faq, search_related_faq, search_analogous_faq, or search_rules")
        if not suggestions and not state.evidence:
            suggestions.append("search_analogous_faq or search_rules with a mechanism-specific query")
        try:
            report = self._build_claim_coverage_report(state, legacy_coverage)
            aggregation = aggregate_claim_coverage(
                report,
                evidence_by_id={item.evidence_id: item for item in state.evidence},
            )
            coverage = self._coverage_from_claim_aggregation(
                state,
                legacy_coverage,
                report,
                aggregation,
            )
        except ValueError as exc:
            new_state = self._replace(state, coverage=legacy_coverage)
            return self._warn(new_state, str(exc)), ToolObservation(
                tool="critique_coverage",
                ok=False,
                summary="claim coverage report rejected",
                data=legacy_coverage.to_dict(),
                validation_error=str(exc),
                warnings=(str(exc),),
            )
        new_state = self._replace(state, coverage=coverage)
        new_state = self._replace(
            new_state,
            claim_coverage_report=report,
            claim_coverage_aggregation=aggregation,
        )
        suggestions.extend(aggregation.suggested_queries)
        return new_state, ToolObservation(
            tool="critique_coverage",
            ok=True,
            summary=coverage.next_action,
            data=coverage.to_dict()
            | {
                "evidence_revision": evidence_revision(new_state),
                "claim_coverage_report": report.to_dict(),
                "claim_coverage_aggregation": aggregation.to_dict(),
                "suggested_next_searches": list(dict.fromkeys(suggestions)),
            },
        )

    def _build_claim_coverage_report(
        self,
        state: AdjudicationAgentState,
        legacy_coverage: CoverageReport,
    ) -> ClaimCoverageReport:
        allowed = {item.evidence_id for item in state.evidence}
        evidence_by_id = {item.evidence_id: item for item in state.evidence}
        if self.llm is not None and state.evidence:
            try:
                response = self.llm.invoke(
                    build_claim_coverage_prompt(state, legacy_coverage)
                )
            except Exception as exc:
                raise ValueError(f"Coverage Critic LLM 请求失败：{exc}") from exc
            report = parse_claim_coverage_report(
                str(getattr(response, "content", response)),
                allowed_evidence_ids=allowed,
            )
            report = stabilize_claim_coverage_report(report, state)
            defects = claim_coverage_semantic_defects(report, state)
            if defects:
                try:
                    repair_response = self.llm.invoke(
                        build_claim_coverage_repair_prompt(
                            state,
                            legacy_coverage,
                            report,
                            defects,
                        )
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Coverage Critic 语义修复请求失败：{exc}"
                    ) from exc
                report = parse_claim_coverage_report(
                    str(getattr(repair_response, "content", repair_response)),
                    allowed_evidence_ids=allowed,
                )
                report = stabilize_claim_coverage_report(report, state)
                remaining_defects = claim_coverage_semantic_defects(report, state)
                if remaining_defects:
                    raise ValueError(
                        "Claim Coverage Report semantic validation failed after repair: "
                        + "; ".join(remaining_defects)
                    )
            report = ensure_actionable_claim_gaps(
                report,
                evidence_by_id=evidence_by_id,
            )
            return report
        report = self._deterministic_claim_coverage_report(state, legacy_coverage)
        return ensure_actionable_claim_gaps(
            report,
            evidence_by_id=evidence_by_id,
        )

    def _deterministic_claim_coverage_report(
        self,
        state: AdjudicationAgentState,
        legacy_coverage: CoverageReport,
    ) -> ClaimCoverageReport:
        questions = state.scenario.questions if state.scenario else ()
        evidence_ids = tuple(item.evidence_id for item in state.evidence[:3])
        status: ClaimStatus = (
            "supported"
            if state.evidence and not (
                legacy_coverage.missing_cards
                or legacy_coverage.ambiguous_cards
                or legacy_coverage.missing_evidence
            )
            else "unknown"
        )
        claims: list[ClaimCoverageClaim] = []
        gaps: list[ClaimCoverageGap] = []
        for index, question in enumerate(questions, start=1):
            branch = "hypothetical" if _has_hypothetical(question) else "main"
            claim_id = stable_claim_id(
                question,
                branch=branch,
                claim_type=("hypothetical_branch" if branch == "hypothetical" else "case_application"),
                semantic_role="answer_application",
            )
            issue_id = stable_issue_id(
                question,
                branch=branch,
                semantic_role="answer",
            )
            checks = tuple(
                ClaimEvidenceCheck(
                    evidence_id=evidence_id,
                    relevance="medium",
                    entailment=("supports" if status == "supported" else "unknown"),
                    reason="Deterministic fallback coverage uses available evidence as provisional support.",
                )
                for evidence_id in evidence_ids
            )
            claims.append(
                ClaimCoverageClaim(
                    claim_id=claim_id,
                    stable_id=claim_id,
                    issue_id=issue_id,
                    claim=f"Evidence is sufficient to answer: {question}",
                    claim_type=("hypothetical_branch" if branch == "hypothetical" else "case_application"),
                    status=status,
                    polarity="unknown",
                    question=question,
                    branch=branch,
                    evidence_checks=checks,
                    gap=("" if status == "supported" else "Need relevant ruling or rules evidence."),
                )
            )
            if status != "supported":
                gaps.append(
                    ClaimCoverageGap(
                        claim_id=claim_id,
                        reason="Need relevant ruling or rules evidence.",
                        suggested_queries=(question,),
                    )
                )
        return ClaimCoverageReport(
            required_claims=tuple(claims),
            issues=_issues_from_claims(claims),
            gaps=tuple(gaps),
            suggested_queries=tuple(query for gap in gaps for query in gap.suggested_queries),
            source="deterministic",
        )

    def _coverage_from_claim_aggregation(
        self,
        state: AdjudicationAgentState,
        legacy_coverage: CoverageReport,
        report: ClaimCoverageReport,
        aggregation: ClaimCoverageAggregation,
    ) -> CoverageReport:
        missing_evidence = tuple(
            dict.fromkeys(
                [
                    *(
                        item
                        for item in legacy_coverage.missing_evidence
                        if item != "Claim Coverage Report is required before final submission."
                    ),
                    *aggregation.blocking_claims,
                    *aggregation.unresolved_issues,
                ]
            )
        )
        missing_hypothetical = tuple(
            dict.fromkeys(
                [
                    *legacy_coverage.missing_hypothetical_evidence,
                    *(
                        claim.claim_id
                        for claim in report.required_claims
                        if claim.branch == "hypothetical" and claim.status != "supported"
                    ),
                ]
            )
        )
        sufficient = legacy_coverage.sufficient and aggregation.sufficient
        return CoverageReport(
            sufficient=sufficient,
            missing_cards=legacy_coverage.missing_cards,
            ambiguous_cards=legacy_coverage.ambiguous_cards,
            missing_evidence=missing_evidence,
            missing_hypothetical_evidence=missing_hypothetical,
            next_action=("final" if sufficient else "retry"),
        )

    def _tool_submit_ruling(
        self,
        state: AdjudicationAgentState,
        arguments: Mapping[str, Any],
    ) -> tuple[AdjudicationAgentState, ToolObservation]:
        coverage = self._check_coverage(state)
        answer = ""
        try:
            if "answer" not in arguments:
                raise SubmitValidationError(
                    (
                        SubmitValidationIssue(
                            code="required",
                            path="$.answer",
                            message="submit_ruling requires answer.",
                            retryable=True,
                        ),
                    )
                )
            raw_answer = _answer_argument_to_json(arguments["answer"])
            raw_payload = json.loads(raw_answer)
            _validate_explicit_submit_field_shapes(raw_payload)
            answer = self._normalize_answer_source_status(
                raw_answer,
                self._replace(state, coverage=coverage),
            )
            payload = json.loads(answer)
            _validate_non_blank_answer_references(payload)
            allowed = {item.evidence_id for item in state.evidence}
            _validate_answer_evidence_references(payload, allowed)
            validate_evidence_citations(answer, allowed)
            validate_adjudication_answer_json(answer, allowed)
            _validate_unresolved_mentions_accounted_for(answer, state)
            _validate_answer_claim_coverage(answer, state)
            _validate_answer_evidence_relationships(answer, state)
        except ValueError as exc:
            validation = _coerce_submit_validation_error(exc)
            validation_error = str(validation)
            rejected_answer = answer or _display_submit_answer(arguments.get("answer"))
            attempt = SubmitAttempt(
                accepted=False,
                answer=rejected_answer,
                validation_error=validation_error,
            )
            new_state = self._replace(
                self._warn(state, validation_error),
                submit_attempts=(*state.submit_attempts, attempt),
            )
            return new_state, ToolObservation(
                tool="submit_ruling",
                ok=False,
                summary="submit_ruling rejected final answer",
                data={
                    "validation_errors": [
                        issue.to_dict() for issue in validation.issues
                    ],
                    "consumes_submit_budget": validation.consumes_submit_budget,
                },
                validation_error=validation_error,
                warnings=(validation_error,),
            )
        answer = _finalize_answer_confidence(
            answer,
            sufficient=coverage.sufficient,
        )
        self._emit_coverage(coverage)
        attempt = SubmitAttempt(accepted=True, answer=answer)
        new_state = self._replace(
            state,
            submit_attempts=(*state.submit_attempts, attempt),
            final_answer=answer,
            coverage=coverage,
            agent_loop_complete=True,
        )
        self._emit_progress(
            "final",
            "最终裁定已通过校验。",
            evidence_count=len(state.evidence),
            sufficient=coverage.sufficient,
        )
        return new_state, ToolObservation(
            tool="submit_ruling",
            ok=True,
            summary="accepted final ruling answer",
            data={
                "coverage": coverage.to_dict(),
                "validation_errors": [],
                "consumes_submit_budget": True,
            },
        )

    def _search_tool_evidence(
        self,
        query: str,
        *,
        card_ids: Sequence[int],
        stage: str,
        relationship: str,
        allowed_kinds: set[str],
        chapter_ids: Sequence[str] = (),
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        evidence: list[EvidenceItem] = []
        stages: list[dict[str, Any]] = []
        if self.ruling_retrievers:
            for retriever in self.ruling_retrievers:
                result = retriever.search(
                    query,
                    card_ids=card_ids,
                    chapter_ids=chapter_ids,
                    relationship=relationship,
                    sparse_top_k=self.settings.ruling_sparse_top_k,
                    dense_top_k=self.settings.ruling_dense_top_k,
                    fused_top_k=self.settings.ruling_fused_top_k,
                    final_top_k=self.settings.ruling_final_top_k,
                    semantic=True,
                    rerank_provider=self.settings.ruling_rerank_provider,
                )
                candidates = [
                    candidate
                    for candidate in result.candidates
                    if candidate.document.kind in allowed_kinds
                    and (
                        relationship in candidate.relationship.split(",")
                        or (
                            relationship == "analogous"
                            and {
                                "direct_card",
                                "related_card",
                            }.intersection(candidate.relationship.split(","))
                        )
                    )
                ]
                stage_items = [
                    _with_evidence_stage(
                        candidate.to_evidence(),
                        stage,
                        query=query,
                        tool=stage,
                    )
                    for candidate in candidates
                ]
                evidence.extend(stage_items)
                stages.append(
                    {
                        "stage": stage,
                        "requested_card_ids": list(card_ids),
                        "query_summary": query[:240],
                        "chapter_ids": list(chapter_ids),
                        "returned_evidence_ids": [item.evidence_id for item in stage_items],
                        "relationships": [item.relationship for item in stage_items],
                        "skipped": False,
                        "tool": stage,
                    }
                )
            return evidence, stages
        if relationship == "direct_card" and self.ruling_index is not None:
            for card_id in card_ids:
                evidence.extend(
                    _with_evidence_stage(faq.to_evidence("direct_card"), stage, query=query, tool=stage)
                    for faq in self.ruling_index.direct_faqs(card_id)
                )
        elif relationship == "related_card" and self.ruling_index is not None:
            for card_id in card_ids:
                evidence.extend(
                    _with_evidence_stage(faq.to_evidence("related_card"), stage, query=query, tool=stage)
                    for faq in self.ruling_index.related_faqs(card_id)
                )
        elif relationship == "card_supplement" and self.ruling_index is not None:
            for card_id in card_ids:
                supplement = self.ruling_index.supplement(card_id)
                if supplement is not None:
                    evidence.append(_with_evidence_stage(supplement.to_evidence(), stage, query=query, tool=stage))
        elif relationship == "analogous" and self.ruling_index is not None:
            effect_tags = self._effect_tags_from_ids(card_ids) if card_ids else ()
            tag_map = {
                card.card_id: decode_effect_category_labels(card.category)
                for card in self.cards
            }
            evidence.extend(
                _with_evidence_stage(faq.to_evidence("analogous"), stage, query=query, tool=stage)
                for faq in self.ruling_index.search(
                    query,
                    top_k=self.settings.adjudication_ruling_top_k,
                    effect_tags=effect_tags,
                    card_effect_tags_by_id=tag_map,
                )
            )
        elif relationship == "rule_document" and self.rule_index is not None:
            evidence.extend(
                _with_evidence_stage(section.to_evidence(), stage, query=query, tool=stage)
                for section in self.rule_index.search(
                    query,
                    top_k=self.settings.adjudication_rules_top_k,
                    chapter_ids=chapter_ids,
                )
            )
        stages.append(
            {
                "stage": stage,
                "requested_card_ids": list(card_ids),
                "query_summary": query[:240],
                "chapter_ids": list(chapter_ids),
                "returned_evidence_ids": [item.evidence_id for item in evidence],
                "relationships": [item.relationship for item in evidence],
                "skipped": False,
                "tool": stage,
            }
        )
        return evidence, stages

    def _merge_card_resolutions(
        self,
        existing: Sequence[CardResolution],
        updates: Sequence[CardResolution],
    ) -> tuple[CardResolution, ...]:
        merged = {resolution.mention: resolution for resolution in existing}
        for resolution in updates:
            merged[resolution.mention] = resolution
        return tuple(merged.values())

    def _resolved_cards_from_resolutions(
        self,
        resolutions: Sequence[CardResolution],
    ) -> tuple[ResolvedCard, ...]:
        resolved_by_id: dict[int, ResolvedCard] = {}
        resolved_scores: dict[int, float] = {}
        for resolution in resolutions:
            if resolution.status != "resolved" or resolution.selected_card_id not in self.cards_by_id:
                continue
            card = self.cards_by_id[resolution.selected_card_id]
            score = resolution.candidates[0].score if resolution.candidates else 0.0
            existing = resolved_by_id.get(card.card_id)
            if (
                existing is None
                or score > resolved_scores.get(card.card_id, 0.0)
                or (
                    score == resolved_scores.get(card.card_id, 0.0)
                    and len(resolution.mention) < len(existing.mention)
                )
            ):
                resolved_by_id[card.card_id] = ResolvedCard(
                    mention=resolution.mention,
                    matches=(card,),
                )
                resolved_scores[card.card_id] = score
        return tuple(resolved_by_id.values())

    def _resolved_card_ids(self, state: AdjudicationAgentState) -> tuple[int, ...]:
        return tuple(
            card.matches[0].card_id
            for card in state.resolved_cards
            if len(card.matches) == 1
        )

    def _effect_tags_from_ids(self, card_ids: Sequence[int]) -> tuple[str, ...]:
        tags: list[str] = []
        for card_id in card_ids:
            card = self.cards_by_id.get(card_id)
            if card is not None:
                tags.extend(decode_effect_category_labels(card.category))
        return tuple(dict.fromkeys(tags))

    def _node_extract(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        warnings: list[str] = []
        try:
            scenario = (
                parse_scenario_with_llm(state.scenario_text, self.llm, known_cards=self.cards)
                if self.llm is not None
                else parse_scenario(state.scenario_text, known_cards=self.cards)
            )
        except Exception as exc:
            scenario = parse_scenario(state.scenario_text, known_cards=self.cards)
            state = self._warn(state, str(exc))
            warnings.append(str(exc))
        extracted_mentions = self._extract_card_mentions(state.scenario_text, scenario, warnings)
        queries = tuple(dict.fromkeys(mention.text for mention in extracted_mentions))
        loop_trace = CardResolutionLoopTrace(
            extracted_mentions=extracted_mentions,
            warnings=tuple(warnings),
        )
        self._emit_progress(
            "parse",
            f"场景解析完成，共识别 {len(scenario.questions)} 个问题。",
        )
        return {
            "agent_state": self._trace(
                self._replace(
                    state,
                    scenario=scenario,
                    mention_queries=queries,
                    extracted_card_mentions=extracted_mentions,
                    card_resolution_loop=loop_trace,
                ),
                "extract_scenario",
                f"parsed {len(queries)} mention queries",
                {
                    "queries": list(queries),
                    "questions": list(scenario.questions),
                    "card_mentions": [mention.to_dict() for mention in extracted_mentions],
                    "warnings": warnings,
                },
            )
        }

    def _extract_card_mentions(
        self,
        scenario_text: str,
        scenario: ParsedScenario,
        warnings: list[str],
    ) -> tuple[ExtractedCardMention, ...]:
        if (
            self.llm is not None
            and self.settings.adjudication_llm_card_extraction
        ):
            try:
                mentions = extract_card_mentions_with_llm(scenario_text, self.llm)
                if mentions:
                    return mentions
                warnings.append("LLM card-name extraction returned no mentions.")
            except Exception as exc:
                warnings.append(str(exc))
        return tuple(
            ExtractedCardMention(text=query, source="deterministic")
            for query in candidate_mention_queries(
                scenario_text,
                scenario,
                alias_mentions=self.card_alias_mentions,
            )
        )

    def _node_resolve_cards(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        warnings = list(state.card_resolution_loop.warnings)
        resolutions = [
            self._resolve_card_mention(mention, warnings=warnings)
            for mention in state.mention_queries
        ]
        retry_count = 0
        for retry_index in range(self.settings.adjudication_card_resolution_retry_limit):
            retry_items = [
                resolution
                for resolution in resolutions
                if resolution.status in {"unresolved", "ambiguous"}
            ]
            if not retry_items:
                break
            retry_count = retry_index + 1
            replacements: dict[str, CardResolution] = {}
            for resolution in retry_items:
                for query in self._targeted_card_queries(resolution.mention, state):
                    retry_resolution = self._resolve_card_mention(query, warnings=warnings)
                    if (
                        retry_resolution.status == "resolved"
                        and self._targeted_retry_explains_mention(
                            resolution.mention,
                            query,
                            retry_resolution,
                        )
                    ):
                        replacements[resolution.mention] = CardResolution(
                            mention=resolution.mention,
                            status="resolved",
                            candidates=retry_resolution.candidates,
                            selected_card_id=retry_resolution.selected_card_id,
                            selection_source=retry_resolution.selection_source,
                            selection_reason=f"targeted retry: {query}",
                        )
                        break
            if not replacements:
                break
            resolutions = [
                replacements.get(resolution.mention, resolution)
                for resolution in resolutions
            ]
        selected_ids = [
            resolution.selected_card_id
            for resolution in resolutions
            if resolution.status == "resolved" and resolution.selected_card_id is not None
        ]
        resolved_by_id: dict[int, ResolvedCard] = {}
        resolved_scores: dict[int, float] = {}
        for resolution in resolutions:
            if resolution.status != "resolved" or resolution.selected_card_id not in self.cards_by_id:
                continue
            card = self.cards_by_id[resolution.selected_card_id]
            existing = resolved_by_id.get(card.card_id)
            score = resolution.candidates[0].score if resolution.candidates else 0.0
            if (
                existing is None
                or score > resolved_scores.get(card.card_id, 0.0)
                or (
                    score == resolved_scores.get(card.card_id, 0.0)
                    and len(resolution.mention) < len(existing.mention)
                )
            ):
                resolved_by_id[card.card_id] = ResolvedCard(
                    mention=resolution.mention,
                    matches=(card,),
                )
                resolved_scores[card.card_id] = score
        resolved = tuple(resolved_by_id.values())
        loop_trace = CardResolutionLoopTrace(
            extracted_mentions=state.card_resolution_loop.extracted_mentions,
            retry_count=retry_count,
            warnings=tuple(dict.fromkeys(warnings)),
        )
        self._emit_progress(
            "card_resolution",
            f"卡片解析完成：{len(resolved)} 张已确认，{len(resolutions) - len(resolved)} 项待确认。",
        )
        state = self._replace(
            state,
            card_resolutions=tuple(resolutions),
            resolved_cards=resolved,
            card_resolution_loop=loop_trace,
        )
        return {
            "agent_state": self._trace(
                state,
                "resolve_cards",
                f"resolved {len(selected_ids)} cards",
                {
                    "selected_card_ids": selected_ids,
                    "resolutions": [resolution.to_dict() for resolution in resolutions],
                    "card_resolution_loop": loop_trace.to_dict(),
                },
            )
        }

    def _resolve_card_mention(
        self,
        mention: str,
        *,
        warnings: list[str],
    ) -> CardResolution:
        candidates = search_card_candidates(
            mention,
            self.cards,
            top_k=self.settings.adjudication_card_resolution_top_k,
            retriever=self.card_retriever,
            verified_aliases=self.verified_card_aliases,
        )
        resolution = select_card_resolution(
            mention,
            candidates,
            confidence_threshold=self.settings.adjudication_card_confidence_threshold,
            ambiguity_margin=self.settings.adjudication_card_ambiguity_margin,
        )
        if (
            resolution.status == "ambiguous"
            and self.llm is not None
            and self.settings.adjudication_llm_card_selection
        ):
            try:
                selected_id, reason = self._select_card_with_llm(mention, resolution.candidates)
                if selected_id is not None:
                    selected = next(
                        (
                            candidate
                            for candidate in resolution.candidates
                            if candidate.card_id == selected_id
                        ),
                        None,
                    )
                    if selected is None:
                        return resolution
                    candidates = tuple(
                        _validated_card_candidate(
                            candidate,
                            status="accepted",
                            reason=reason,
                            source="llm",
                        )
                        if candidate.card_id == selected_id
                        else candidate
                        for candidate in resolution.candidates
                    )
                    return CardResolution(
                        mention=mention,
                        status="resolved",
                        candidates=candidates,
                        selected_card_id=selected_id,
                        selection_source="llm",
                        selection_reason=reason,
                    )
            except ValueError as exc:
                warnings.append(str(exc))
        return resolution

    def _select_card_with_llm(
        self,
        mention: str,
        candidates: Sequence[CardCandidate],
    ) -> tuple[int | None, str]:
        assert self.llm is not None
        try:
            response = self.llm.invoke(
                build_card_selection_prompt(mention, candidates)
            )
        except Exception as exc:
            raise ValueError(f"卡名消歧 LLM 请求失败：{exc}") from exc
        return parse_card_selection_response(
            str(getattr(response, "content", response)),
            allowed_ids={candidate.card_id for candidate in candidates},
        )

    def _targeted_card_queries(
        self,
        mention: str,
        state: AdjudicationAgentState,
    ) -> tuple[str, ...]:
        parts = [
            mention,
            _strip_mention_noise(normalize_card_name(mention)),
        ]
        if state.scenario is not None:
            parts.extend(
                query
                for query in state.scenario.card_mentions
                if _mention_query_compatible(mention, query)
            )
        parts.extend(
            alias
            for alias in self.card_alias_mentions
            if alias in state.scenario_text and _mention_query_compatible(mention, alias)
        )
        queries = [part for part in parts if part and part.strip()]
        return tuple(dict.fromkeys(queries))

    def _targeted_retry_explains_mention(
        self,
        mention: str,
        query: str,
        resolution: CardResolution,
    ) -> bool:
        if resolution.selected_card_id is None:
            return False
        selected = next(
            (
                candidate
                for candidate in resolution.candidates
                if candidate.card_id == resolution.selected_card_id
            ),
            None,
        )
        if selected is None:
            return False
        mention_normalized = _strip_mention_noise(normalize_card_name(mention))
        query_normalized = _strip_mention_noise(normalize_card_name(query))
        name_normalized = normalize_card_name(selected.name)
        alias = self.verified_card_aliases.get(normalize_card_name(mention))
        if alias is not None and alias[0] == selected.card_id:
            return True
        query_alias = self.verified_card_aliases.get(normalize_card_name(query))
        if (
            query_alias is not None
            and query_alias[0] == selected.card_id
            and _normalized_mentions_overlap(mention_normalized, query_normalized)
        ):
            return True
        if _normalized_mentions_overlap(mention_normalized, name_normalized):
            return True
        if len(query_normalized) > max(len(mention_normalized), len(name_normalized)) + 4:
            return False
        return (
            _normalized_mentions_overlap(mention_normalized, query_normalized)
            and _normalized_mentions_overlap(query_normalized, name_normalized)
        )

    def _node_retrieve_evidence(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        evidence, stages = self._base_evidence(state)
        state = self._replace(
            state,
            evidence=tuple(dedupe_evidence(evidence)),
            evidence_retrieval_stages=tuple(stages),
        )
        return {
            "agent_state": self._trace(
                state,
                "retrieve_evidence",
                f"collected {len(state.evidence)} evidence items",
                {
                    "evidence_ids": [item.evidence_id for item in state.evidence],
                    "persistent_ruling_retriever_count": len(self.ruling_retrievers),
                    "used_persistent_ruling_retrieval": bool(self.ruling_retrievers),
                    "retrieval_stages": [dict(stage) for stage in stages],
                },
            )
        }

    def _node_check_coverage(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        coverage = self._check_coverage(state)
        self._emit_coverage(coverage)
        return {
            "agent_state": self._trace(
                self._replace(state, coverage=coverage),
                "check_coverage",
                coverage.next_action,
                coverage.to_dict(),
            )
        }

    def _node_targeted_retrieval(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        extra, stages = self._targeted_evidence(state)
        evidence = tuple(dedupe_evidence([*state.evidence, *extra]))
        state = self._replace(
            state,
            evidence=evidence,
            evidence_retrieval_stages=(
                *state.evidence_retrieval_stages,
                *stages,
            ),
            iterations=state.iterations + 1,
        )
        return {
            "agent_state": self._trace(
                state,
                "targeted_retrieval",
                f"added {len(extra)} targeted evidence items",
                {
                    "evidence_ids": [item.evidence_id for item in extra],
                    "persistent_ruling_retriever_count": len(self.ruling_retrievers),
                    "used_persistent_ruling_retrieval": bool(self.ruling_retrievers),
                    "retrieval_stages": [dict(stage) for stage in stages],
                },
            )
        }

    def _node_final_answer(self, payload: AgentGraphState) -> AgentGraphState:
        state = payload["agent_state"]
        answer = self._generate_final_answer(state)
        self._emit_progress(
            "final",
            "最终裁定已生成。",
            evidence_count=len(state.evidence),
            sufficient=state.coverage.sufficient,
            fallback=self.llm is None,
        )
        return {
            "agent_state": self._trace(
                self._replace(state, final_answer=answer),
                "final_answer",
                "generated final answer",
                {"coverage": state.coverage.to_dict()},
            )
        }

    def _route_after_coverage(self, payload: AgentGraphState) -> str:
        state = payload["agent_state"]
        if (
            not state.coverage.sufficient
            and state.coverage.next_action == "retry"
            and state.iterations < self.settings.adjudication_agent_max_iterations
        ):
            return "retry"
        return "final"

    def _base_evidence(self, state: AdjudicationAgentState) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        evidence: list[EvidenceItem] = build_card_effect_evidence(state.resolved_cards)
        stages: list[dict[str, Any]] = []
        card_ids = [card.matches[0].card_id for card in state.resolved_cards if len(card.matches) == 1]
        if self.ruling_retrievers:
            persistent, persistent_stages = self._persistent_ruling_evidence(
                state,
                card_ids=card_ids,
                extra_terms="",
            )
            evidence.extend(persistent)
            stages.extend(persistent_stages)
            return evidence, stages
        if self.ruling_index is not None:
            direct_start = len(evidence)
            for card_id in card_ids:
                evidence.extend(
                    faq.to_evidence("direct_card")
                    for faq in self.ruling_index.direct_faqs(card_id)
                )
            stages.append(
                _runtime_retrieval_stage("direct_faq", card_ids, evidence[direct_start:])
            )
            related_start = len(evidence)
            for card_id in card_ids:
                evidence.extend(
                    faq.to_evidence("related_card")
                    for faq in self.ruling_index.related_faqs(card_id)
                )
            stages.append(
                _runtime_retrieval_stage("related_faq", card_ids, evidence[related_start:])
            )
            supplement_start = len(evidence)
            for card_id in card_ids:
                supplement = self.ruling_index.supplement(card_id)
                if supplement is not None:
                    evidence.append(supplement.to_evidence())
            stages.append(
                _runtime_retrieval_stage(
                    "supplement", card_ids, evidence[supplement_start:]
                )
            )
            analogous_start = len(evidence)
            evidence.extend(self._analogous_faq_evidence(state, top_k=5))
            stages.append(
                _runtime_retrieval_stage(
                    "analogous_faq", (), evidence[analogous_start:]
                )
            )
        if self.rule_index is not None:
            rules_start = len(evidence)
            evidence.extend(self._rule_evidence(state, top_k=5))
            stages.append(
                _runtime_retrieval_stage("rules", (), evidence[rules_start:])
            )
        return evidence, stages

    def _targeted_evidence(self, state: AdjudicationAgentState) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        evidence: list[EvidenceItem] = []
        stages: list[dict[str, Any]] = []
        extra_terms = self._coverage_retry_terms(state)
        if self.ruling_retrievers:
            card_ids = [
                card.matches[0].card_id
                for card in state.resolved_cards
                if len(card.matches) == 1
            ]
            return self._persistent_ruling_evidence(
                state,
                card_ids=card_ids,
                extra_terms=extra_terms,
            )
        if self.ruling_index is not None:
            evidence.extend(
                self._analogous_faq_evidence(
                    state,
                    top_k=5,
                    extra_terms=extra_terms,
                )
            )
        if self.rule_index is not None:
            evidence.extend(
                self._rule_evidence(
                    state,
                    top_k=5,
                    extra_terms=extra_terms,
                )
            )
        return evidence, stages

    def _coverage_retry_terms(self, state: AdjudicationAgentState) -> str:
        return "\n".join(
            part
            for part in [
                *state.coverage.missing_evidence,
                *state.coverage.missing_hypothetical_evidence,
                *state.coverage.missing_cards,
                *state.coverage.ambiguous_cards,
                *(
                    state.claim_coverage_aggregation.suggested_queries
                    if state.claim_coverage_aggregation is not None
                    else ()
                ),
            ]
            if part
        )

    def _persistent_ruling_evidence(
        self,
        state: AdjudicationAgentState,
        *,
        card_ids: Sequence[int],
        extra_terms: str = "",
    ) -> tuple[list[EvidenceItem], list[dict[str, Any]]]:
        evidence: list[EvidenceItem] = []
        query = self._mechanism_query(state, extra_terms=extra_terms)
        stages: list[dict[str, Any]] = []
        stage_specs = [
            ("direct_faq", "direct_card", card_ids, {"faq"}, "direct_card"),
            ("related_faq", "related_card", card_ids, {"faq"}, "related_card"),
            ("supplement", "card_supplement", card_ids, {"card_supplement"}, "card_supplement"),
            ("analogous_faq", "analogous", (), {"faq"}, "analogous"),
            ("rules", "rule_document", (), {"rule_passage"}, "rule_document"),
        ]
        for retriever in self.ruling_retrievers:
            for stage_name, default_relationship, requested_card_ids, allowed_kinds, required_relationship in stage_specs:
                if requested_card_ids == card_ids and not card_ids:
                    stages.append(
                        {
                            "stage": stage_name,
                            "requested_card_ids": [],
                            "query_summary": query[:240],
                            "returned_evidence_ids": [],
                            "relationships": [],
                            "skipped": True,
                            "reason": "no resolved card ids",
                        }
                    )
                    continue
                result = retriever.search(
                    query,
                    card_ids=requested_card_ids,
                    relationship=default_relationship,
                    sparse_top_k=self.settings.ruling_sparse_top_k,
                    dense_top_k=self.settings.ruling_dense_top_k,
                    fused_top_k=self.settings.ruling_fused_top_k,
                    final_top_k=self.settings.ruling_final_top_k,
                    semantic=True,
                    rerank_provider=self.settings.ruling_rerank_provider,
                )
                stage_candidates = [
                    candidate
                    for candidate in result.candidates
                    if candidate.document.kind in allowed_kinds
                    and required_relationship in candidate.relationship.split(",")
                ]
                stage_items = [
                    _with_evidence_stage(candidate.to_evidence(), stage_name)
                    for candidate in stage_candidates
                ]
                evidence.extend(stage_items)
                stages.append(
                    {
                        "stage": stage_name,
                        "requested_card_ids": list(requested_card_ids),
                        "query_summary": query[:240],
                        "returned_evidence_ids": [item.evidence_id for item in stage_items],
                        "relationships": [item.relationship for item in stage_items],
                        "skipped": False,
                    }
                )
        return evidence, stages

    def _analogous_faq_evidence(
        self,
        state: AdjudicationAgentState,
        *,
        top_k: int,
        extra_terms: str = "",
    ) -> list[EvidenceItem]:
        if self.ruling_index is None or state.scenario is None:
            return []
        effect_tags = self._effect_tags(state)
        tag_map = {
            card.card_id: decode_effect_category_labels(card.category)
            for card in self.cards
        }
        query = self._mechanism_query(state, extra_terms=extra_terms)
        return [
            faq.to_evidence("analogous")
            for faq in self.ruling_index.search(
                query,
                top_k=top_k,
                effect_tags=effect_tags,
                card_effect_tags_by_id=tag_map,
            )
        ]

    def _rule_evidence(
        self,
        state: AdjudicationAgentState,
        *,
        top_k: int,
        extra_terms: str = "",
    ) -> list[EvidenceItem]:
        if self.rule_index is None:
            return []
        return [
            section.to_evidence()
            for section in self.rule_index.search(
                self._mechanism_query(state, extra_terms=extra_terms),
                top_k=top_k,
            )
        ]

    def _check_coverage(self, state: AdjudicationAgentState) -> CoverageReport:
        preparation_gaps = _mandatory_preparation_gaps(state)
        missing_cards = tuple(
            resolution.mention
            for resolution in state.card_resolutions
            if resolution.status == "unresolved"
            and not self._mention_covered_by_resolved_card(resolution.mention, state)
        )
        ambiguous_cards = tuple(
            resolution.mention
            for resolution in state.card_resolutions
            if resolution.status == "ambiguous"
            and not all(
                self._mention_covered_by_resolved_card(candidate.name, state)
                for candidate in resolution.candidates
            )
        )
        questions = state.scenario.questions if state.scenario else ()
        has_adjudication_evidence = any(
            item.kind in {"faq", "card_supplement", "rule_passage"}
            for item in state.evidence
        )
        missing_evidence = () if has_adjudication_evidence else tuple(questions)
        if not questions:
            missing_evidence = tuple(
                dict.fromkeys(
                    [*missing_evidence, "No prepared questions are available."]
                )
            )
        if preparation_gaps:
            missing_evidence = tuple(
                dict.fromkeys(
                    [
                        *missing_evidence,
                        "Scenario preparation gap: call "
                        + ", ".join(preparation_gaps)
                        + ".",
                    ]
                )
            )
        missing_hypothetical: tuple[str, ...] = ()
        if self.llm is not None and state.evidence and state.claim_coverage_aggregation is None:
            missing_evidence = tuple(
                dict.fromkeys(
                    [
                        *missing_evidence,
                        "Claim Coverage Report is required before final submission.",
                    ]
                )
            )
        if state.claim_coverage_aggregation is not None:
            aggregation = state.claim_coverage_aggregation
            missing_evidence = tuple(
                dict.fromkeys([*missing_evidence, *aggregation.blocking_claims])
            )
            missing_hypothetical = tuple(
                dict.fromkeys(
                    [
                        *missing_hypothetical,
                        *(
                            claim.claim_id
                            for claim in (
                                state.claim_coverage_report.required_claims
                                if state.claim_coverage_report is not None
                                else ()
                            )
                            if claim.branch == "hypothetical" and claim.status != "supported"
                        ),
                    ]
                )
            )
        sufficient = not (
            missing_cards or ambiguous_cards or missing_evidence or missing_hypothetical
        )
        if state.claim_coverage_aggregation is not None:
            sufficient = sufficient and state.claim_coverage_aggregation.sufficient
        next_action = "final" if sufficient else "retry"
        if preparation_gaps:
            next_action = "prepare"
        if (missing_cards or ambiguous_cards) and not preparation_gaps:
            next_action = "final"
        return CoverageReport(
            sufficient=sufficient,
            missing_cards=missing_cards,
            ambiguous_cards=ambiguous_cards,
            missing_evidence=missing_evidence,
            missing_hypothetical_evidence=missing_hypothetical,
            next_action=next_action,
        )

    def _generate_final_answer(self, state: AdjudicationAgentState) -> str:
        if self.llm is None or state.scenario is None or not state.evidence:
            return self._deterministic_answer(state)
        prompt = build_agent_adjudication_answer_prompt(state)
        allowed = {item.evidence_id for item in state.evidence}
        for attempt in range(2):
            response = self.llm.invoke(prompt)
            raw_answer = str(getattr(response, "content", response))
            try:
                answer = self._normalize_answer_source_status(raw_answer, state)
                validate_evidence_citations(answer, allowed)
                validate_adjudication_answer_json(answer, allowed)
                return _finalize_answer_confidence(
                    answer,
                    sufficient=state.coverage.sufficient,
                )
            except ValueError as exc:
                prompt = (
                    f"{prompt}\n\nPrevious output was invalid: {exc}\n"
                    "Return corrected strict JSON only."
                )
                if attempt == 1:
                    break
        return self._deterministic_answer(state)

    def _deterministic_answer(self, state: AdjudicationAgentState) -> str:
        source_status = self._source_status(state)
        insufficient = not state.coverage.sufficient
        payload = {
            "conclusion": (
                "当前证据不足，无法给出高置信度裁定。"
                if insufficient
                else "已汇总可验证证据；最终模型裁定不可用，以下为保守结果。"
            ),
            "source_status": source_status,
            "confidence": "low" if insufficient else "medium",
            "answers": [
                {
                    "question": question,
                    "case_conclusion": (
                        "现有证据不足，不能可靠确定本案结论。"
                        if insufficient
                        else "请依据下列已收集证据核对本案结论。"
                    ),
                    "hypothetical_conclusion": (
                        "现有证据不足，不能可靠确定该假设分支。"
                        if _has_hypothetical(question)
                        else ""
                    ),
                    "reasoning": [
                        {
                            "judgment": "保守回退不会推断超出已检索证据的事实。",
                            "evidence_ids": [item.evidence_id for item in state.evidence[:5]],
                        }
                    ],
                }
                for question in (
                    state.scenario.questions
                    if state.scenario and state.scenario.questions
                    else ()
                )
            ],
            "evidence_use": [
                {
                    "evidence_id": item.evidence_id,
                    "role": item.relationship or item.kind,
                    "use": item.title or item.kind,
                }
                for item in state.evidence[:10]
            ],
            "uncertainties": self._coverage_warnings(state.coverage),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _normalize_answer_source_status(
        self,
        answer: str,
        state: AdjudicationAgentState,
    ) -> str:
        status = self._source_status(state)
        return normalize_adjudication_answer_source_status(
            answer,
            state.evidence,
            faq_searched=status["faq"] != "not_searched",
            rules_searched=status["rules"] != "not_searched",
            sufficient=state.coverage.sufficient,
        )

    def _source_status(self, state: AdjudicationAgentState) -> dict[str, str]:
        faq_stages = {"direct_faq", "related_faq", "analogous_faq"}
        attempted_stages = {
            str(stage.get("stage") or "")
            for stage in state.evidence_retrieval_stages
        }
        attempted_tools = {entry.action.tool for entry in state.tool_history}
        faq_searched = bool(
            faq_stages.intersection(attempted_stages)
            or {
                "search_direct_faq",
                "search_related_faq",
                "search_analogous_faq",
            }.intersection(attempted_tools)
        )
        rules_searched = bool(
            "rules" in attempted_stages or "search_rules" in attempted_tools
        )
        return derive_adjudication_source_status(
            state.evidence,
            faq_searched=faq_searched,
            rules_searched=rules_searched,
            sufficient=state.coverage.sufficient,
        )

    def _mechanism_query(self, state: AdjudicationAgentState, *, extra_terms: str = "") -> str:
        scenario = state.scenario
        return "\n".join(
            part
            for part in [
                state.scenario_text,
                " ".join(scenario.effect_actions) if scenario else "",
                " ".join(scenario.replacement_candidates) if scenario else "",
                " ".join(self._effect_tags(state)),
                extra_terms,
            ]
            if part
        )

    def _effect_tags(self, state: AdjudicationAgentState) -> tuple[str, ...]:
        tags: list[str] = []
        for resolved in state.resolved_cards:
            if len(resolved.matches) == 1:
                tags.extend(decode_effect_category_labels(resolved.matches[0].category))
        return tuple(dict.fromkeys(tags))

    def _mention_covered_by_resolved_card(
        self,
        mention: str,
        state: AdjudicationAgentState,
    ) -> bool:
        normalized = _strip_mention_noise(normalize_card_name(mention))
        if len(normalized) < 3:
            return False
        for resolved in state.resolved_cards:
            if len(resolved.matches) != 1:
                continue
            card_name = normalize_card_name(resolved.matches[0].name)
            if card_name in normalized or normalized in card_name:
                return True
        return False

    def _coverage_warnings(self, coverage: CoverageReport) -> list[str]:
        warnings: list[str] = []
        if coverage.missing_cards:
            warnings.append(f"Unresolved card mentions: {', '.join(coverage.missing_cards)}")
        if coverage.ambiguous_cards:
            warnings.append(f"Ambiguous card mentions: {', '.join(coverage.ambiguous_cards)}")
        if coverage.missing_hypothetical_evidence:
            warnings.append("Hypothetical branch evidence is incomplete.")
        if coverage.missing_evidence:
            warnings.append("Evidence is incomplete for one or more questions.")
        if any(
            item.startswith("Scenario preparation gap:")
            for item in coverage.missing_evidence
        ):
            warnings.append(
                "Scenario preparation is incomplete; questions were not inferred from raw text."
            )
        return warnings

    def _replace(self, state: AdjudicationAgentState, **changes: Any) -> AdjudicationAgentState:
        data = {
            "scenario_text": state.scenario_text,
            "scenario": state.scenario,
            "agent_loop_preparation_required": state.agent_loop_preparation_required,
            "scenario_field_sources": state.scenario_field_sources,
            "scenario_revision": state.scenario_revision,
            "mention_queries": state.mention_queries,
            "extracted_card_mentions": state.extracted_card_mentions,
            "card_resolution_loop": state.card_resolution_loop,
            "card_resolutions": state.card_resolutions,
            "resolved_cards": state.resolved_cards,
            "evidence": state.evidence,
            "evidence_retrieval_stages": state.evidence_retrieval_stages,
            "rule_chapter_catalog": state.rule_chapter_catalog,
            "rule_chapter_candidates": state.rule_chapter_candidates,
            "planner_notes": state.planner_notes,
            "hypotheses": state.hypotheses,
            "tool_history": state.tool_history,
            "tool_signatures": state.tool_signatures,
            "submit_attempts": state.submit_attempts,
            "agent_loop_complete": state.agent_loop_complete,
            "coverage": state.coverage,
            "claim_coverage_report": state.claim_coverage_report,
            "claim_coverage_aggregation": state.claim_coverage_aggregation,
            "trace": state.trace,
            "warnings": state.warnings,
            "iterations": state.iterations,
            "final_answer": state.final_answer,
        }
        data.update(changes)
        return AdjudicationAgentState(**data)

    def _trace(
        self,
        state: AdjudicationAgentState,
        node: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> AdjudicationAgentState:
        return self._replace(
            state,
            trace=(
                *state.trace,
                TraceEntry(node=node, summary=summary, metadata=metadata or {}),
            ),
        )

    def _warn(self, state: AdjudicationAgentState, warning: str) -> AdjudicationAgentState:
        return self._replace(state, warnings=(*state.warnings, warning))

    def _emit_coverage(self, coverage: CoverageReport) -> None:
        self._emit_progress(
            "coverage",
            "证据覆盖检查完成：已满足。" if coverage.sufficient else "证据覆盖检查完成：仍有缺口。",
            sufficient=coverage.sufficient,
        )

    def _emit_progress(
        self,
        stage: str,
        message: str,
        *,
        tool: str = "",
        ok: bool | None = None,
        evidence_count: int | None = None,
        sufficient: bool | None = None,
        fallback: bool = False,
    ) -> None:
        if self.progress_callback is None:
            return
        event = AdjudicationProgressEvent(
            stage=stage,
            message=message,
            tool=tool,
            ok=ok,
            evidence_count=evidence_count,
            sufficient=sufficient,
            fallback=fallback,
        )
        try:
            self.progress_callback(event)
        except Exception:
            # Progress reporting must not interrupt adjudication.
            return


def build_agent_adjudication_answer_prompt(state: AdjudicationAgentState) -> str:
    base = build_adjudication_answer_prompt(
        state.scenario or ParsedScenario(original_text=state.scenario_text),
        state.evidence,
    )
    coverage = json.dumps(state.coverage.to_dict(), ensure_ascii=False, indent=2)
    trace_summary = json.dumps(
        _card_resolution_prompt_summary(state),
        ensure_ascii=False,
        indent=2,
    )
    return "\n".join(
        [
            base,
            "",
            "coverage_report:",
            coverage,
            "",
            "tool_trace_summary:",
            trace_summary,
        ]
    )


def _with_evidence_stage(
    item: EvidenceItem,
    stage: str,
    *,
    query: str = "",
    tool: str = "",
) -> EvidenceItem:
    stages = [
        str(value)
        for value in item.metadata.get("retrieval_stages", [])
        if str(value)
    ]
    stages.append(stage)
    return EvidenceItem(
        evidence_id=item.evidence_id,
        kind=item.kind,
        source=item.source,
        title=item.title,
        text=item.text,
        relationship=item.relationship,
        card_ids=item.card_ids,
        metadata=dict(item.metadata)
        | {
            "retrieval_stage": stage,
            "retrieval_stages": list(dict.fromkeys(stages)),
            "retrieval_query": query,
            "retrieval_tool": tool or stage,
        },
    )


def _runtime_retrieval_stage(
    stage: str,
    card_ids: Sequence[int],
    evidence: Sequence[EvidenceItem],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "requested_card_ids": list(card_ids),
        "query_summary": "",
        "returned_evidence_ids": [item.evidence_id for item in evidence],
        "relationships": [item.relationship for item in evidence],
        "skipped": False,
        "tool": stage,
    }


def _argument_card_ids(
    arguments: Mapping[str, Any],
    *,
    default: Sequence[int],
) -> tuple[int, ...]:
    value = arguments.get("card_ids")
    if value is None:
        return tuple(dict.fromkeys(int(item) for item in default))
    if not isinstance(value, list):
        raise ValueError("card_ids must be an array of integers.")
    return tuple(dict.fromkeys(int(item) for item in value))


def _answer_argument_to_json(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        json.loads(stripped)
        return stripped
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, indent=2)
    raise ValueError("submit_ruling answer must be a JSON string or object.")


def _display_submit_answer(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value or "")


def _validate_explicit_submit_field_shapes(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        return
    if "answers" in payload and (
        not isinstance(payload.get("answers"), list) or not payload["answers"]
    ):
        raise ValueError("LLM adjudication answer must include at least one answer.")
    if "source_status" not in payload:
        return
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


def _coerce_submit_validation_error(exc: ValueError) -> SubmitValidationError:
    if isinstance(exc, SubmitValidationError):
        return exc
    message = str(exc)
    lowered = message.lower()
    code = "invalid_submit_answer"
    path = "$.answer"
    retryable = True
    field_error = re.match(
        r"^LLM adjudication answer field (?P<path>\S+) (?P<reason>.+)$",
        message,
    )
    missing_field = re.search(r"missing required field: ([A-Za-z0-9_]+)", message)
    source_field = re.search(r"source_status\.([A-Za-z0-9_]+)", message)
    if field_error:
        relative_path = field_error.group("path").strip().lstrip(".")
        reason = field_error.group("reason").lower()
        path = f"$.answer.{relative_path}" if relative_path else "$.answer"
        if "required" in reason:
            code = "required"
        elif "non-empty" in reason or any(
            marker in reason
            for marker in ("must be one of", "must equal", "must contain", "invalid value")
        ):
            code = "invalid_value"
        elif "must be" in reason:
            code = "invalid_type"
        else:
            code = "invalid_value"
    elif missing_field:
        code = "required"
        path = f"$.answer.{missing_field.group(1)}"
    elif source_field:
        code = "invalid_type" if "must be a" in lowered else "invalid_value"
        path = f"$.answer.source_status.{source_field.group(1)}"
    elif "source_status must be an object" in lowered:
        code = "invalid_type"
        path = "$.answer.source_status"
    elif "source_status must contain" in lowered:
        code = "invalid_value"
        path = "$.answer.source_status"
    elif "must include at least one answer" in lowered:
        code = "invalid_value"
        path = "$.answer.answers"
    elif "valid json" in lowered:
        code = "invalid_json"
    elif "json string or object" in lowered:
        code = "invalid_type"
    elif "json object" in lowered or "must be an object" in lowered:
        code = "invalid_type"
    elif "unknown evidence ids" in lowered:
        code = "unknown_evidence_id"
        path = "$.answer.evidence_use"
    elif "unresolved or ambiguous" in lowered:
        code = "unresolved_card_reference"
        path = "$.answer.uncertainties"
    elif "unsupported claim" in lowered:
        code = "unsupported_claim"
        path = "$.answer.answers"
        retryable = False
    elif "contradicted claim" in lowered:
        code = "contradicted_claim"
        path = "$.answer.answers"
        retryable = False
    elif "unknown claim" in lowered:
        code = "unknown_claim_status"
        path = "$.answer.answers"
        retryable = False
    elif "partially supported claim" in lowered:
        code = "partially_supported_claim"
        path = "$.answer.answers"
        retryable = False
    return SubmitValidationError(
        (
            SubmitValidationIssue(
                code=code,
                path=path,
                message=message,
                retryable=retryable,
            ),
        )
    )


def _iter_answer_reference_fields(
    value: Any,
    *,
    path: str = "$.answer",
) -> Sequence[tuple[str, str, Any]]:
    references: list[tuple[str, str, Any]] = []
    singular_keys = {"claim_id", "issue_id", "evidence_id"}
    plural_keys = {"claim_ids", "issue_ids", "evidence_ids"}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_path = f"{path}.{key}"
            if key in singular_keys:
                references.append((key, nested_path, nested))
            elif key in plural_keys:
                if isinstance(nested, list):
                    singular = key[:-1]
                    references.extend(
                        (singular, f"{nested_path}[{index}]", item)
                        for index, item in enumerate(nested)
                    )
                else:
                    references.append((key, nested_path, nested))
            else:
                references.extend(
                    _iter_answer_reference_fields(nested, path=nested_path)
                )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            references.extend(
                _iter_answer_reference_fields(item, path=f"{path}[{index}]")
            )
    return references


def _validate_non_blank_answer_references(payload: Any) -> None:
    issues: list[SubmitValidationIssue] = []
    for field_name, path, value in _iter_answer_reference_fields(payload):
        if not isinstance(value, str):
            issues.append(
                SubmitValidationIssue(
                    code="invalid_type",
                    path=path,
                    message=f"{field_name} must be a string.",
                    retryable=True,
                )
            )
        elif not value.strip():
            issues.append(
                SubmitValidationIssue(
                    code="blank_reference_id",
                    path=path,
                    message=f"{field_name} must be a non-empty string.",
                    retryable=True,
                )
            )
    if issues:
        raise SubmitValidationError(issues)


def _validate_answer_evidence_references(payload: Any, allowed_ids: set[str]) -> None:
    unknown = [
        (path, value.strip())
        for field_name, path, value in _iter_answer_reference_fields(payload)
        if field_name == "evidence_id"
        and isinstance(value, str)
        and value.strip()
        and value.strip() not in allowed_ids
    ]
    if unknown:
        raise SubmitValidationError(
            tuple(
                SubmitValidationIssue(
                    code="unknown_evidence_id",
                    path=path,
                    message=f"Answer cited unknown evidence ids: {evidence_id}",
                    retryable=True,
                )
                for path, evidence_id in unknown
            )
        )


def _answer_reference_path(payload: Any, field_name: str, target: str) -> str:
    for reference_name, path, value in _iter_answer_reference_fields(payload):
        if (
            reference_name == field_name
            and isinstance(value, str)
            and value.strip() == target
        ):
            return path
    return "$.answer.answers"


def _finalize_answer_confidence(answer: str, *, sufficient: bool) -> str:
    if sufficient:
        return answer
    payload = json.loads(answer)
    if not isinstance(payload, Mapping):
        return answer
    normalized = dict(payload)
    normalized["confidence"] = "low"
    return json.dumps(normalized, ensure_ascii=False, indent=2)


def _validate_unresolved_mentions_accounted_for(
    answer: str,
    state: AdjudicationAgentState,
) -> None:
    unresolved = [
        resolution.mention
        for resolution in state.card_resolutions
        if resolution.status in {"unresolved", "ambiguous"}
    ]
    if not unresolved:
        return
    payload = json.loads(answer)
    if str(payload.get("confidence") or "").lower() == "low":
        return
    uncertainties = payload.get("uncertainties")
    if not isinstance(uncertainties, list):
        raise ValueError("Answer must list unresolved or ambiguous cards in uncertainties.")
    joined = "\n".join(str(item) for item in uncertainties)
    missing = [mention for mention in unresolved if mention not in joined]
    if missing:
        raise ValueError(
            "Answer ignored unresolved or ambiguous card mentions: "
            + ", ".join(missing)
        )


def _validate_answer_claim_coverage(answer: str, state: AdjudicationAgentState) -> None:
    if state.claim_coverage_report is None:
        return
    payload = json.loads(answer)
    claims_by_id = {
        claim.claim_id: claim
        for claim in state.claim_coverage_report.required_claims
    }
    issues_by_id = {
        issue.issue_id: issue
        for issue in _normalize_issues_against_claims(
            state.claim_coverage_report.issues,
            state.claim_coverage_report.required_claims,
        )
    }
    cited_claim_ids = _collect_json_claim_ids(payload)
    cited_issue_ids = _collect_json_issue_ids(payload)
    if not cited_claim_ids and not cited_issue_ids:
        raise SubmitValidationError(
            (
                SubmitValidationIssue(
                    code="missing_claim_reference",
                    path="$.answer.answers",
                    message="Answer must cite claim_id or issue_id values from the Claim Coverage Report.",
                    retryable=True,
                ),
            )
        )
    unknown = sorted(claim_id for claim_id in cited_claim_ids if claim_id not in claims_by_id)
    if unknown:
        raise SubmitValidationError(
            tuple(
                SubmitValidationIssue(
                    code="unknown_claim_id",
                    path=_answer_reference_path(payload, "claim_id", claim_id),
                    message=f"Answer cited unknown claim ids: {claim_id}",
                    retryable=True,
                )
                for claim_id in unknown
            )
        )
    unknown_issues = sorted(
        issue_id for issue_id in cited_issue_ids if issue_id not in issues_by_id
    )
    if unknown_issues:
        raise SubmitValidationError(
            tuple(
                SubmitValidationIssue(
                    code="unknown_issue_id",
                    path=_answer_reference_path(payload, "issue_id", issue_id),
                    message=f"Answer cited unknown issue ids: {issue_id}",
                    retryable=True,
                )
                for issue_id in unknown_issues
            )
        )
    uncertainties = "\n".join(str(item) for item in payload.get("uncertainties", []))
    confidence = str(payload.get("confidence") or "").lower()
    for issue_id in sorted(cited_issue_ids):
        issue = issues_by_id[issue_id]
        if issue.status == "supported":
            continue
        if confidence != "low" and issue_id not in uncertainties and issue.question not in uncertainties:
            raise SubmitValidationError(
                (
                    SubmitValidationIssue(
                        code=f"{issue.status}_issue",
                        path=_answer_reference_path(payload, "issue_id", issue_id),
                        message=f"Answer relies on {issue.status} issue without uncertainty: {issue_id}",
                        retryable=False,
                    ),
                )
            )
    for claim_id in sorted(cited_claim_ids):
        claim = claims_by_id[claim_id]
        if claim.status == "supported":
            continue
        if claim.status == "contradicted":
            raise SubmitValidationError(
                (
                    SubmitValidationIssue(
                        code="contradicted_claim",
                        path=_answer_reference_path(payload, "claim_id", claim_id),
                        message=f"Answer relies on contradicted claim: {claim_id}",
                        retryable=False,
                    ),
                )
            )
        if claim.status in {"unsupported", "unknown"}:
            if confidence != "low" and claim_id not in uncertainties and claim.claim not in uncertainties:
                raise SubmitValidationError(
                    (
                        SubmitValidationIssue(
                            code=f"{claim.status}_claim",
                            path=_answer_reference_path(payload, "claim_id", claim_id),
                            message=f"Answer relies on {claim.status} claim without uncertainty: {claim_id}",
                            retryable=False,
                        ),
                    )
                )
        if claim.status == "partially_supported":
            if confidence != "low" and claim_id not in uncertainties and claim.claim not in uncertainties:
                raise SubmitValidationError(
                    (
                        SubmitValidationIssue(
                            code="partially_supported_claim",
                            path=_answer_reference_path(payload, "claim_id", claim_id),
                            message=f"Answer relies on partially supported claim without uncertainty: {claim_id}",
                            retryable=False,
                        ),
                    )
                )
    for answer_index, answer_item in enumerate(payload.get("answers", [])):
        if not isinstance(answer_item, Mapping):
            continue
        hypothetical = str(answer_item.get("hypothetical_conclusion") or "").strip()
        if not hypothetical:
            continue
        branch_claim_ids = _collect_json_claim_ids(answer_item)
        branch_issue_ids = _collect_json_issue_ids(answer_item)
        if not any(
            claims_by_id[claim_id].branch != "main"
            or claims_by_id[claim_id].claim_type == "hypothetical_branch"
            for claim_id in branch_claim_ids
            if claim_id in claims_by_id and claims_by_id[claim_id].status == "supported"
        ) and not any(
            issues_by_id[issue_id].status == "supported"
            and (
                "hypothetical" in issues_by_id[issue_id].issue_id
                or "假设" in issues_by_id[issue_id].question
                or issues_by_id[issue_id].polarity == "conditional"
            )
            for issue_id in branch_issue_ids
            if issue_id in issues_by_id
        ):
            raise SubmitValidationError(
                (
                    SubmitValidationIssue(
                        code="unsupported_hypothetical_branch",
                        path=f"$.answer.answers[{answer_index}].hypothetical_conclusion",
                        message="Hypothetical conclusion must cite a supported hypothetical branch claim.",
                        retryable=False,
                    ),
                )
            )
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    for claim_id in cited_claim_ids:
        claim = claims_by_id[claim_id]
        if claim.status != "supported" or not claim.decisive:
            continue
        cited_evidence = [
            evidence_by_id[check.evidence_id]
            for check in claim.evidence_checks
            if check.evidence_id in evidence_by_id
            and check.entailment in {"supports", "partially_supports"}
        ]
        if cited_evidence and all(item.relationship == "analogous" for item in cited_evidence):
            if "类比" not in uncertainties and "analog" not in uncertainties.lower():
                raise SubmitValidationError(
                    (
                        SubmitValidationIssue(
                            code="analogous_evidence_requires_uncertainty",
                            path=_answer_reference_path(payload, "claim_id", claim_id),
                            message=f"Answer relies only on analogous evidence for decisive claim without uncertainty: {claim_id}",
                            retryable=True,
                        ),
                    )
                )


def _validate_answer_evidence_relationships(answer: str, state: AdjudicationAgentState) -> None:
    payload = json.loads(answer)
    evidence_by_id = {item.evidence_id: item for item in state.evidence}
    for index, item in enumerate(payload.get("evidence_use", [])):
        if not isinstance(item, Mapping):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        role = str(item.get("role") or "")
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None or not role:
            continue
        actual_roles = {part.strip() for part in evidence.relationship.split(",") if part.strip()}
        actual_roles.add(evidence.kind)
        if evidence.kind == "card_effect":
            actual_roles.add("direct_card")
        requested_roles = {part.strip() for part in role.split(",") if part.strip()}
        if actual_roles and requested_roles and not requested_roles.issubset(actual_roles):
            raise SubmitValidationError(
                (
                    SubmitValidationIssue(
                        code="invalid_evidence_relationship",
                        path=f"$.answer.evidence_use[{index}].role",
                        message=(
                            f"Answer misstated evidence relationship for {evidence_id}: "
                            f"{role} not in {evidence.relationship}"
                        ),
                        retryable=True,
                    ),
                )
            )


def _collect_json_claim_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "claim_id" and isinstance(nested, str) and nested.strip():
                ids.add(nested.strip())
            elif key == "claim_ids" and isinstance(nested, list):
                ids.update(
                    item.strip()
                    for item in nested
                    if isinstance(item, str) and item.strip()
                )
            else:
                ids.update(_collect_json_claim_ids(nested))
    elif isinstance(value, list):
        for item in value:
                ids.update(_collect_json_claim_ids(item))
    return ids


def _collect_json_issue_ids(value: Any) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "issue_id" and isinstance(nested, str) and nested.strip():
                ids.add(nested.strip())
            elif key == "issue_ids" and isinstance(nested, list):
                ids.update(
                    item.strip()
                    for item in nested
                    if isinstance(item, str) and item.strip()
                )
            else:
                ids.update(_collect_json_issue_ids(nested))
    elif isinstance(value, list):
        for item in value:
            ids.update(_collect_json_issue_ids(item))
    return ids


def _action_targets_claim_gap(
    state: AdjudicationAgentState,
    action: PlannerAction,
) -> bool:
    if state.claim_coverage_report is None or state.claim_coverage_aggregation is None:
        return True
    target_text = " ".join(
        part
        for part in [
            str(action.arguments.get("query") or ""),
            action.rationale,
            " ".join(action.notes),
            " ".join(action.hypotheses),
        ]
        if part
    )
    if not target_text.strip():
        return False
    blocking_ids = set(state.claim_coverage_aggregation.blocking_claims)
    blocking_ids.update(state.claim_coverage_aggregation.partial_claims)
    blocking_ids.update(state.claim_coverage_aggregation.unknown_claims)
    blocking_ids.update(state.claim_coverage_aggregation.unsupported_claims)
    blocking_ids.update(state.claim_coverage_aggregation.contradicted_claims)
    unresolved_claims = [
        claim
        for claim in state.claim_coverage_report.required_claims
        if claim.claim_id in blocking_ids or claim.status != "supported"
    ]
    if not unresolved_claims:
        return False
    if any(claim.claim_id in target_text for claim in unresolved_claims):
        return True
    gap_by_claim = {
        gap.claim_id: gap
        for gap in state.claim_coverage_report.gaps
    }
    target_terms: list[str] = []
    for claim in unresolved_claims:
        target_terms.extend([claim.claim, claim.gap, claim.question, claim.branch])
        gap = gap_by_claim.get(claim.claim_id)
        if gap is not None:
            target_terms.extend([gap.reason, *gap.suggested_queries])
    target_terms.extend(state.claim_coverage_report.suggested_queries)
    target_terms.extend(state.claim_coverage_report.contradictions)
    return any(
        _text_has_meaningful_overlap(target_text, term)
        for term in target_terms
        if term
    )


def _text_has_meaningful_overlap(left: str, right: str) -> bool:
    left_norm = normalize_card_name(left)
    right_norm = normalize_card_name(right)
    if not left_norm or not right_norm:
        return False
    if len(right_norm) >= 4 and right_norm in left_norm:
        return True
    if len(left_norm) >= 4 and left_norm in right_norm:
        return True
    left_ascii = set(re.findall(r"[a-z0-9]{3,}", left_norm))
    right_ascii = set(re.findall(r"[a-z0-9]{3,}", right_norm))
    if left_ascii and right_ascii and left_ascii.intersection(right_ascii):
        return True
    left_cjk = {char for char in left_norm if "\u4e00" <= char <= "\u9fff"}
    right_cjk = {char for char in right_norm if "\u4e00" <= char <= "\u9fff"}
    if left_cjk and right_cjk:
        common = len(left_cjk.intersection(right_cjk))
        return common >= 2 and common / max(1, min(len(left_cjk), len(right_cjk))) >= 0.25
    return False


def _rules_query_card_name_warning(
    query: str,
    state: AdjudicationAgentState,
) -> str:
    normalized_query = normalize_card_name(query)
    if not normalized_query:
        return "search_rules query must be a mechanism-focused non-empty query."
    matched_names: list[str] = []
    matched_name_tokens: set[str] = set()
    for card in state.resolved_cards:
        if len(card.matches) != 1:
            continue
        name = card.matches[0].name
        normalized_name = normalize_card_name(name)
        if normalized_name and normalized_name in normalized_query:
            matched_names.append(name)
            matched_name_tokens.update(re.findall(r"[A-Za-z0-9]{3,}", name.lower()))
    if not matched_names:
        return ""
    if any(normalized_query == normalize_card_name(name) for name in matched_names):
        return (
            "search_rules query is card-name-heavy. Rewrite it as a mechanism-focused query with OCG mechanics "
            "instead of resolved card names, e.g. destruction replacement, failed destruction, subsequent processing."
        )
    mechanism_tokens = {
        token
        for token in re.findall(r"[A-Za-z0-9]{3,}", query.lower())
        if token not in matched_name_tokens
    }
    cjk_mechanism_terms = [
        "破坏",
        "代替",
        "适用",
        "发动",
        "处理",
        "特殊召唤",
        "连锁",
        "对象",
        "伤害步骤",
        "效果",
        "不能",
        "成功",
        "未",
    ]
    mechanism_score = len(mechanism_tokens) + sum(
        1 for term in cjk_mechanism_terms if term in query
    )
    if len(matched_names) >= 2 or mechanism_score < 2:
        return (
            "search_rules query is card-name-heavy. Rewrite it as a mechanism-focused query with OCG mechanics "
            "instead of resolved card names, e.g. destruction replacement, failed destruction, subsequent processing."
        )
    return ""


def _tool_signature(action: PlannerAction) -> str:
    return json.dumps(
        {
            "tool": action.tool,
            "arguments": action.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _tool_signature_for_state(
    action: PlannerAction,
    state: AdjudicationAgentState,
) -> str:
    if action.tool != "critique_coverage":
        return _tool_signature(action)
    return json.dumps(
        {
            "tool": action.tool,
            "arguments": action.arguments,
            "evidence_revision": evidence_revision(state),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _last_critique_revision(state: AdjudicationAgentState) -> str:
    for entry in reversed(state.tool_history):
        if entry.action.tool == "critique_coverage":
            return str(entry.observation.data.get("evidence_revision") or "")
    return ""


def _tool_budget_counts(history: Sequence[ToolHistoryEntry]) -> dict[str, int]:
    retrieval_tools = {
        "search_card",
        "search_direct_faq",
        "search_related_faq",
        "search_card_supplement",
        "search_analogous_faq",
        "classify_rule_chapters",
        "search_rules",
    }
    counts = {"retrieval": 0, "critique": 0, "submit": 0}
    for entry in history:
        if entry.action.tool in retrieval_tools and entry.observation.ok:
            counts["retrieval"] += 1
        elif entry.action.tool == "critique_coverage" and entry.observation.ok:
            counts["critique"] += 1
        elif entry.action.tool == "submit_ruling" and (
            entry.observation.ok
            or bool(entry.observation.data.get("consumes_submit_budget", True))
        ):
            counts["submit"] += 1
    return counts


def stable_issue_id(
    question: str,
    *,
    branch: str,
    semantic_role: str,
) -> str:
    return ".".join(
        part
        for part in [
            "issue",
            _stable_slug(branch or "main"),
            _stable_slug(semantic_role or "answer"),
            _short_hash(question),
        ]
        if part
    )


def stable_claim_id(
    question: str,
    *,
    branch: str,
    claim_type: str,
    semantic_role: str,
) -> str:
    return ".".join(
        part
        for part in [
            "claim",
            _stable_slug(branch or "main"),
            _stable_slug(claim_type or "claim"),
            _stable_slug(semantic_role or "required"),
            _short_hash(question),
        ]
        if part
    )


def _stable_slug(value: str) -> str:
    ascii_words = re.findall(r"[a-z0-9]+", value.lower())
    if ascii_words:
        return "_".join(ascii_words[:5])[:48]
    normalized = normalize_card_name(value)
    if normalized:
        return _short_hash(normalized)
    return "x"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def evidence_revision(state: AdjudicationAgentState) -> str:
    payload = [
        {
            "evidence_id": item.evidence_id,
            "kind": item.kind,
            "relationship": item.relationship,
            "text_hash": _short_hash(item.text),
        }
        for item in state.evidence
    ]
    return _short_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _claim_coverage_report_hash(report: ClaimCoverageReport) -> str:
    return _short_hash(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def build_adjudication_evidence_context(
    state: AdjudicationAgentState,
    *,
    latest_limit: int = 8,
    analogous_limit: int = 4,
    rules_limit: int = 4,
) -> AdjudicationEvidenceContext:
    cited_by_evidence = _claim_cited_evidence_ids(state.claim_coverage_report)
    selected: list[tuple[str, EvidenceItem]] = []
    seen: set[str] = set()

    def add(section: str, item: EvidenceItem) -> None:
        key = f"{section}:{item.evidence_id}"
        if key in seen:
            return
        selected.append((section, item))
        seen.add(key)

    resolved_card_ids = set(_resolved_card_ids_from_state(state))
    for item in state.evidence:
        if item.kind == "card_effect":
            add("resolved_card_effects_full", item)
    for item in state.evidence:
        if item.kind == "card_supplement" and (
            not item.card_ids or resolved_card_ids.intersection(item.card_ids)
        ):
            add("resolved_card_supplements", item)
    for item in state.evidence:
        if item.evidence_id in cited_by_evidence:
            add("claim_cited_evidence", item)

    direct_count = related_count = analogous_count = rules_count = latest_count = 0
    for item in state.evidence:
        relationships = {part.strip() for part in item.relationship.split(",") if part.strip()}
        if item.kind == "faq" and "direct_card" in relationships:
            direct_count += 1
            if direct_count <= latest_limit:
                add("direct_faq", item)
        elif item.kind == "faq" and "related_card" in relationships:
            related_count += 1
            if related_count <= latest_limit:
                add("related_faq", item)
        elif item.kind == "faq" and "analogous" in relationships:
            analogous_count += 1
            if analogous_count <= analogous_limit:
                add("analogous_faq_bounded", item)
        elif item.kind == "rule_passage":
            rules_count += 1
            if rules_count <= rules_limit:
                add("rules_bounded", item)

    for item in reversed(state.evidence):
        latest_count += 1
        if latest_count > latest_limit:
            break
        add("latest_evidence", item)

    context_items = tuple(
        EvidenceContextItem(
            section=section,
            evidence_id=item.evidence_id,
            kind=item.kind,
            relationship=item.relationship,
            title=item.title,
            card_ids=item.card_ids,
            text=_evidence_context_text(section, item),
            cited_by_claim_ids=tuple(cited_by_evidence.get(item.evidence_id, ())),
        )
        for section, item in selected
    )
    return AdjudicationEvidenceContext(
        items=context_items,
        evidence_revision=evidence_revision(state),
    )


def _evidence_context_text(section: str, item: EvidenceItem) -> str:
    if section in {
        "resolved_card_effects_full",
        "resolved_card_supplements",
        "claim_cited_evidence",
    }:
        return item.text
    limit = 720 if item.kind in {"faq", "rule_passage"} else 520
    return item.text[:limit]


def _claim_cited_evidence_ids(
    report: ClaimCoverageReport | None,
) -> dict[str, tuple[str, ...]]:
    cited: dict[str, list[str]] = {}
    if report is None:
        return {}
    for claim in report.required_claims:
        for check in claim.evidence_checks:
            cited.setdefault(check.evidence_id, []).append(claim.claim_id)
    for issue in report.issues:
        for evidence_id in issue.supporting_evidence_ids:
            cited.setdefault(evidence_id, []).append(issue.issue_id)
    return {
        evidence_id: tuple(dict.fromkeys(claim_ids))
        for evidence_id, claim_ids in cited.items()
    }


def _resolved_card_ids_from_state(state: AdjudicationAgentState) -> tuple[int, ...]:
    return tuple(
        card.matches[0].card_id
        for card in state.resolved_cards
        if len(card.matches) == 1
    )


def build_claim_coverage_prompt(
    state: AdjudicationAgentState,
    legacy_coverage: CoverageReport,
) -> str:
    scenario = state.scenario or ParsedScenario(original_text=state.scenario_text)
    evidence_context = build_adjudication_evidence_context(state)
    schema = {
        "schema_version": 2,
        "issues": [
            {
                "issue_id": "稳定 issue id；同一问题后续 critique 必须沿用",
                "question": "用户的一个具体裁定问题",
                "status": "supported | partially_supported | unsupported | contradicted | unknown",
                "polarity": "affirmative | negative | conditional | unknown",
                "supported_answer": "证据支持的答案，例如 yes/no/conditional/unknown",
                "claim_ids": ["支撑该 issue 的 claim_id"],
                "supporting_evidence_ids": ["直接支撑该 issue 答案的 evidence_id"],
                "unresolved_claim_ids": ["仍未解决的 claim_id"],
                "inference_reason": "如果由依赖 claim 推出结论，说明推理链",
                "gap": "未解决时说明缺口",
            }
        ],
        "required_claims": [
            {
                "claim_id": "稳定 claim id；优先使用 claim.<branch>.<type>.<role>.<hash>",
                "stable_id": "同 claim_id；兼容旧字段时也必须给出",
                "claim": "需要证明或反驳的最小裁定命题",
                "claim_type": "card_text | ruling_fact | rule_principle | case_application | hypothetical_branch",
                "status": "supported | partially_supported | unsupported | contradicted | unknown",
                "issue_id": "该 claim 所属 issue_id",
                "semantic_role": "该 claim 在推理中的角色，例如 card_text_shenlong_replacement",
                "polarity": "affirmative | negative | conditional | unknown",
                "supported_answer": "该 claim 支持的答案方向，可为空",
                "question": "该 claim 服务的用户问题",
                "branch": "main 或 hypothetical:<name>",
                "required": True,
                "decisive": True,
                "depends_on": ["其他 claim_id；没有则为空数组"],
                "evidence_checks": [
                    {
                        "evidence_id": "必须来自 evidence_pool",
                        "relevance": "high | medium | low | none",
                        "entailment": "supports | partially_supports | contradicts | does_not_entail | unknown",
                        "reason": "分别说明相关性和能否推出 claim",
                    }
                ],
                "inference_reason": "hypothetical_branch 若由依赖推出，必须说明推理原因",
                "gap": "如果不是 supported，说明缺什么证据；supported 时为空",
            }
        ],
        "gaps": [
            {
                "claim_id": "有缺口的 claim_id",
                "reason": "缺口原因",
                "suggested_queries": ["下一步应检索的机制化 query"],
            }
        ],
        "suggested_queries": ["全局下一步 query，可为空"],
        "contradictions": ["矛盾说明，可为空"],
    }
    return "\n".join(
        [
            "你是游戏王 OCG 裁定证据覆盖审稿器。",
            "不要直接回答最终裁定；只输出 Claim Coverage Report，并维护稳定 issue/claim graph。",
            "不要输出 sufficient true/false；充分性由代码根据 claim status 计算。",
            "第一次 critique 需要初始化 issue/claim graph；后续 critique 必须更新已有 id 的状态、证据检查、gap 和 suggested_queries，不要把同一含义重新编号。",
            "如果发现新必要命题，只能追加新的稳定 id；不能把已有 id 改成另一种含义。",
            "必须先列出回答每个问题所需的 issues 和 required claims，包括假设分支。",
            "每个 issue 必须至少包含一个 required=true、decisive=true 的 case_application。case_application 必须把规则或 FAQ 应用到用户给出的具体时序和状态，不能只复述卡片文本或一般规则。",
            "card_text 只表示卡片印刷文本、卡片种类或固有属性，不得承担本案结论；hypothetical_branch 只用于用户以‘如果/若/假设/即使’明确提出的假设分支，main 分支不得标成 hypothetical_branch。",
            "建立 case_application 时必须逐项检查发动时、支付 cost 后、连锁处理中和效果处理时的关键状态；尤其检查卡片是否因 cost 或其他处理离场、翻面、转移区域，以及持续效果在处理时是否仍对它适用。没有发生这些变化时也要明确确认。",
            "在支持某个 case_application 前，必须主动尝试相反结论，并逐字核对所引用 FAQ/规则的关键句。证据写明离场后不再无效时，不得据此推出仍然无效；证据写明仍然无效时也不得推出恢复有效。",
            "issue 或 case_application 的 status=supported 时，supported_answer 和 polarity 必须明确表示答案方向，不得留空或填写 unknown。",
            "对每个 claim，分别判断 evidence relevance 和 evidence entailment。",
            "相关不等于支持：同卡、同机制或关键词相近只能说明 relevance，不能自动说明 entailment。",
            "把 evidence 对答案方向的作用说清楚：支持肯定、支持否定、条件支持、矛盾或无关。",
            "证据支持“不能适用/不能特殊召唤”等否定结论时，应建立 polarity=negative 且 status=supported 的 issue；不要只把正命题标成 unsupported。",
            "card_text 类型的 claim 必须优先引用 kind=card_effect 的证据；FAQ 只能证明裁定事实或类比规则，不能替代卡片效果文本本身。",
            "required=true 且 decisive=true 的 ruling_fact、case_application 必须至少有一条 entailment=supports 的直接证据，且证据 kind 必须严格为 faq、card_supplement 或 rule_passage；card_effect、其他 kind 和 depends_on 推理都不能替代。",
            "hypothetical_branch 仍须有上述机制证据，或由已 supported 的依赖 claim 推出并填写 depends_on 和 inference_reason。",
            "YGOCDB FAQ 中如果正文列出“以下效果也同样不能/可以适用”，且列出的卡名或效果编号包含本案已解析卡，该 FAQ 对该卡是 related_card/direct_card 级别证据，不要降格成普通 analogous。",
            "FAQ 的回答如果明确写出“不能适用/不能发动/不能特殊召唤/不进行后续处理”等否定裁定，必须优先建立 polarity=negative 且 status=supported 的 issue 或 claim；不要把它写成 unknown 或 partially_supported。",
            "hypothetical_branch 没有完全相同 FAQ 时，可以由已 supported 的 card_text、ruling_fact、rule_principle 依赖推出，但必须填写 depends_on 和 inference_reason；case_application 不得仅靠依赖推理标记为 supported。",
            "如果 evidence 推出相反结论，claim status 必须是 contradicted。",
            "只返回严格 JSON，不要 Markdown 或额外说明。",
            "",
            "ClaimStatus：supported, partially_supported, unsupported, contradicted, unknown",
            "Evidence relevance：high, medium, low, none",
            "Evidence entailment：supports, partially_supports, contradicts, does_not_entail, unknown",
            "Issue polarity：affirmative, negative, conditional, unknown",
            "",
            "输出 JSON schema：",
            json.dumps(schema, ensure_ascii=False, indent=2),
            "",
            "scenario：",
            state.scenario_text,
            "",
            "questions：",
            json.dumps(list(scenario.questions), ensure_ascii=False, indent=2),
            "",
            "legacy_coverage：",
            json.dumps(legacy_coverage.to_dict(), ensure_ascii=False, indent=2),
            "",
            "current_claim_graph（如果非 null，必须按稳定 id 更新）：",
            json.dumps(
                state.claim_coverage_report.to_dict()
                if state.claim_coverage_report is not None
                else None,
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "resolved_cards：",
            json.dumps(
                _card_resolution_prompt_summary(state)["card_resolution"]["resolved"],
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "evidence_pool：",
            json.dumps(evidence_context.to_dict(), ensure_ascii=False, indent=2),
        ]
    )


def build_claim_coverage_repair_prompt(
    state: AdjudicationAgentState,
    legacy_coverage: CoverageReport,
    rejected_report: ClaimCoverageReport,
    defects: Sequence[str],
) -> str:
    return "\n".join(
        [
            build_claim_coverage_prompt(state, legacy_coverage),
            "",
            "上一次候选 Claim Coverage Report 未通过语义结构校验。",
            "请独立复核证据原文并重写完整报告，不要只修改局部字段。",
            "必须修复以下缺陷：",
            json.dumps(list(defects), ensure_ascii=False, indent=2),
            "",
            "被拒绝的候选报告：",
            json.dumps(rejected_report.to_dict(), ensure_ascii=False, indent=2),
        ]
    )


def build_agent_planner_prompt(state: AdjudicationAgentState) -> str:
    scenario = state.scenario or ParsedScenario(original_text=state.scenario_text)
    preparation_status = {
        "mandatory_ready": not _mandatory_preparation_gaps(state),
        "missing_tools": list(_mandatory_preparation_gaps(state)),
        "scenario_revision": state.scenario_revision,
        "field_sources": {
            field_name: dict(provenance)
            for field_name, provenance in state.scenario_field_sources.items()
        },
        "current_fields": {
            "card_mentions": list(scenario.card_mentions),
            "questions": list(scenario.questions),
            "roles": list(scenario.roles),
            "effect_actions": list(scenario.effect_actions),
            "objects": list(scenario.objects),
            "replacement_candidates": list(scenario.replacement_candidates),
        },
    }
    summary = _card_resolution_prompt_summary(state)
    evidence_context = build_adjudication_evidence_context(state)
    recent_history = [
        {
            "step": entry.step,
            "tool": entry.action.tool,
            "arguments": entry.action.arguments,
            "observation": {
                "ok": entry.observation.ok,
                "summary": entry.observation.summary,
                "evidence_ids": list(entry.observation.evidence_ids),
                "validation_error": entry.observation.validation_error,
                "warnings": list(entry.observation.warnings),
            },
            "duplicate": entry.duplicate,
        }
        for entry in state.tool_history[-6:]
    ]
    output_schema = {
        "tool": "one of the allowed tool names",
        "arguments": "JSON object matching the selected tool schema",
        "rationale": "short reason for this next action",
        "notes": ["optional planner notes; not evidence"],
        "hypotheses": ["optional hypotheses; not evidence"],
    }
    coverage_policy = (
        "coverage.sufficient 为 true：下一步必须调用 submit_ruling。不要再调用 search_card、"
        "search_direct_faq、search_related_faq、search_card_supplement、search_analogous_faq、"
        "classify_rule_chapters、search_rules 或 critique_coverage。"
        if state.coverage.sufficient
        else "coverage.sufficient 为 false：先补齐缺失卡名或证据；证据足够后再 submit_ruling。"
    )
    claim_report = (
        state.claim_coverage_report.to_dict()
        if state.claim_coverage_report is not None
        else None
    )
    claim_aggregation = (
        state.claim_coverage_aggregation.to_dict()
        if state.claim_coverage_aggregation is not None
        else None
    )
    return "\n".join(
        [
            "你是游戏王 OCG 裁定 agent 的规划器。表达干练，结论先行，简洁准确，不加入聊天人格、撒娇或角色动作。",
            "你每轮只能选择一个工具。工具结果才是事实；你的 notes/hypotheses 只是工作假设。",
            "不要编造 card_id、evidence_id、别名或裁定事实。",
            "调用 resolve_cards、FAQ/supplement/analogous/rules 证据检索、"
            "critique_coverage 或 submit_ruling 前，必须先分别成功调用 split_questions "
            "和 extract_card_mentions；初始化中的原始场景文本不能直接充当已确认问题或卡名。",
            "准备完成前仅允许提前调用 search_card 和 classify_rule_chapters："
            "它们只使用显式 query，可用于 parser 失败后的恢复或诊断；"
            "不得把其候选结果当作已确认场景字段或裁定证据。",
            "解析工具失败是正常观察：可调整 focus 重试，或改用 strategy=deterministic。"
            "解析工具可重复成功调用，后一次结果会原子替换前一次结果。",
            "extract_scenario_facts 是可选的单字段工具；仅在机制事实有助于规划时调用，"
            "不要为了检索强制填满 roles、effect_actions、objects、replacement_candidates。",
            "如果卡名未解析或有歧义，优先调用 resolve_cards 或 search_card。",
            "FAQ 检索和规则检索的 query 目的不同：search_direct_faq/search_related_faq/search_card_supplement/search_analogous_faq 可以保留卡名、效果名、具体互动和 card_ids。",
            "规则检索应直接查看完整 rule_chapter_catalog，并从中选择真正相关的 chapter_id 传给 search_rules.chapter_ids；可选择目录中的任意有效 id，不受词法 top-k 候选限制。",
            "classify_rule_chapters 仅是可选的词法兼容回退，不是 search_rules 的前置条件，也不能裁剪完整目录。目录足以判断时直接调用 search_rules。",
            "问题跨章节或目录判断不确定时可以选择多个章节，也可以不带 chapter_ids 做全局检索。",
            "调用 classify_rule_chapters 时，query 每次只表达一个核心机制类别，例如“破坏代替”或“效果处理时对象消失”。",
            "classify_rule_chapters 和 search_rules 都必须把具体卡名抽象成机制化问题，例如“破坏被代替后后续处理是否适用”“效果处理时未成功破坏是否继续处理”，不要把卡名作为主要 query 词。",
            "如果 coverage gap 指向某个 stable issue/claim id，下一步检索必须围绕该 id 的缺口生成 query。",
            "只有证据足够时才调用 submit_ruling；submit_ruling 的 answer 必须严格符合 submit_ruling_answer_schema，且引用的 evidence_id 必须来自 evidence_pool。",
            "submit_ruling 的 reasoning 可以引用 claim_id 或 issue_id；引用的 id 必须来自最新 claim_coverage_report。",
            "如果上一次 submit_ruling 被拒绝，下一轮应修正 answer 后再次 submit_ruling，而不是继续检索，除非 coverage 明确显示证据不足。",
            coverage_policy,
            "只返回严格 JSON，不要 Markdown 或额外说明。",
            "",
            "输出 JSON schema：",
            json.dumps(output_schema, ensure_ascii=False, indent=2),
            "",
            "submit_ruling_answer_schema：",
            json.dumps(SUBMIT_RULING_ANSWER_SCHEMA, ensure_ascii=False, indent=2),
            "",
            "submit_ruling 调用示例：",
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": SUBMIT_RULING_ANSWER_SCHEMA},
                    "rationale": "coverage.sufficient 为 true，提交最终裁定。",
                },
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "allowed_tools：",
            json.dumps(AGENT_TOOL_SCHEMAS, ensure_ascii=False, indent=2),
            "",
            "scenario：",
            state.scenario_text,
            "",
            "scenario_preparation：",
            json.dumps(preparation_status, ensure_ascii=False, indent=2),
            "",
            "questions：",
            json.dumps(list(scenario.questions), ensure_ascii=False, indent=2),
            "",
            "card_state：",
            json.dumps(summary["card_resolution"], ensure_ascii=False, indent=2),
            "",
            "rule_chapter_catalog（完整可选目录）：",
            json.dumps(
                [dict(chapter) for chapter in state.rule_chapter_catalog],
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "rule_chapter_candidates：",
            json.dumps(
                [dict(candidate) for candidate in state.rule_chapter_candidates],
                ensure_ascii=False,
                indent=2,
            ),
            "",
            "coverage：",
            json.dumps(state.coverage.to_dict(), ensure_ascii=False, indent=2),
            "",
            "claim_coverage_report：",
            json.dumps(claim_report, ensure_ascii=False, indent=2),
            "",
            "claim_coverage_aggregation：",
            json.dumps(claim_aggregation, ensure_ascii=False, indent=2),
            "",
            "evidence_pool：",
            json.dumps(evidence_context.to_dict(), ensure_ascii=False, indent=2),
            "",
            "planner_notes：",
            json.dumps(list(state.planner_notes[-10:]), ensure_ascii=False, indent=2),
            "",
            "hypotheses：",
            json.dumps(list(state.hypotheses[-10:]), ensure_ascii=False, indent=2),
            "",
            "recent_tool_history：",
            json.dumps(recent_history, ensure_ascii=False, indent=2),
        ]
    )


def _card_resolution_prompt_summary(state: AdjudicationAgentState) -> dict[str, Any]:
    resolved_names = {
        resolved.matches[0].card_id: resolved.matches[0].name
        for resolved in state.resolved_cards
        if len(resolved.matches) == 1
    }
    resolved: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for item in state.card_resolutions:
        if item.status == "resolved":
            resolved.append(
                {
                    "mention": item.mention,
                    "card_id": item.selected_card_id,
                    "name": resolved_names.get(item.selected_card_id or -1, ""),
                    "selection_source": item.selection_source,
                    "selection_reason": item.selection_reason,
                    "candidate_validation": [
                        {
                            "card_id": candidate.card_id,
                            "validation_status": candidate.validation_status,
                            "validation_reason": candidate.validation_reason,
                        }
                        for candidate in item.candidates
                        if candidate.card_id == item.selected_card_id
                    ],
                }
            )
            continue
        target = ambiguous if item.status == "ambiguous" else unresolved
        target.append(
            {
                "mention": item.mention,
                "selection_reason": item.selection_reason,
                "top_candidates": [
                    {
                        "card_id": candidate.card_id,
                        "name": candidate.name,
                        "score": candidate.score,
                        "match_reason": candidate.match_reason,
                        "validation_status": candidate.validation_status,
                        "validation_reason": candidate.validation_reason,
                    }
                    for candidate in item.candidates[:3]
                ],
            }
        )
    return {
        "card_resolution": {
            "retry_count": state.card_resolution_loop.retry_count,
            "resolved": resolved,
            "ambiguous": ambiguous,
            "unresolved": unresolved,
            "warnings": list(state.card_resolution_loop.warnings),
        },
        "evidence_retrieval_stages": [
            dict(stage) for stage in state.evidence_retrieval_stages
        ],
    }


def _has_hypothetical(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in (
            "if",
            "suppose",
            "even if",
            "\u82e5",
            "\u5982\u679c",
            "\u5047\u8bbe",
            "\u5373\u4f7f",
        )
    )


def _mention_query_compatible(mention: str, query: str) -> bool:
    mention_normalized = _strip_mention_noise(normalize_card_name(mention))
    query_normalized = _strip_mention_noise(normalize_card_name(query))
    return _normalized_mentions_overlap(mention_normalized, query_normalized)


def _normalized_mentions_overlap(left: str, right: str) -> bool:
    if not left or not right:
        return False
    shorter = min(len(left), len(right))
    if shorter < SUBSTRING_MIN_NORMALIZED_LENGTH:
        return left == right
    return left in right or right in left


def _strip_mention_noise(value: str) -> str:
    for term in (
        "\u5bf9\u65b9",
        "\u6211\u65b9",
        "\u573a\u4e0a",
        "\u624b\u5361",
        "\u53d1\u52a8",
        "\u9009\u62e9",
        "\u9002\u7528",
        "\u7834\u574f",
        "\u6548\u679c",
        "\u8981\u5c06",
        "\u7684",
        "\u5c06",
        "\u8981",
        "\u62e9",
        "\u65b9",
        "\u65f6",
        "\u8fd8",
        "\u7528",
        "\u4e0a",
    ):
        value = value.replace(term, "")
    return value


def _has_hypothetical_evidence(evidence: Sequence[EvidenceItem]) -> bool:
    for item in evidence:
        text = "\n".join([item.evidence_id, item.title, item.text]).lower()
        if item.evidence_id.endswith(":24345") or "faq:24345" in item.evidence_id:
            return True
        if (
            ("not destroyed" in text or "\u6ca1\u6709\u88ab\u7834\u574f" in text or "\u6ca1\u6709\u7834\u574f" in text)
            and ("special summon" in text or "\u7279\u6b8a\u53ec\u5524" in text)
            and ("not applied" in text or "\u4e0d\u9002\u7528" in text or "\u4e0d\u80fd\u9002\u7528" in text)
        ):
            return True
    return False
