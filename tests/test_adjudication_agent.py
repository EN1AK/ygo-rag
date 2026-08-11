import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_agent.adjudication import (
    EvidenceItem,
    ParsedScenario,
    ResolvedCard,
    RuleIndex,
    RuleSection,
    YgocdbRulingIndex,
    build_question_split_prompt,
    build_scenario_parser_prompt,
    complete_scenario_questions,
    dedupe_evidence,
    parse_scenario,
    parse_scenario_with_llm,
    validate_question_split_payload,
)
from rag_agent.adjudication_agent import (
    AdjudicationAgent,
    CoverageReport,
    CardResolution,
    ClaimCoverageAggregation,
    ClaimCoverageClaim,
    ClaimCoverageGap,
    ClaimCoverageIssue,
    ClaimCoverageReport,
    ClaimEvidenceCheck,
    ExtractedCardMention,
    PlannerAction,
    ToolHistoryEntry,
    ToolObservation,
    TraceEntry,
    aggregate_claim_coverage,
    build_adjudication_evidence_context,
    build_agent_adjudication_answer_prompt,
    build_agent_planner_prompt,
    build_claim_coverage_prompt,
    candidate_mention_queries,
    claim_coverage_semantic_defects,
    parse_claim_coverage_report,
    create_initial_agent_state,
    evidence_revision,
    parse_agent_planner_response,
    parse_card_name_extraction_response,
    parse_card_selection_response,
    normalize_card_name,
    search_card_candidates,
    select_card_resolution,
)
from rag_agent.cards import Card
from rag_agent.config import Settings
from rag_agent.ruling_index import RulingCandidate, RulingDocument, RulingRetrievalResult
from rag_agent.ruling_index import PersistentRulingRetriever


UNCHAINED_SOUL_OF_RAGE = "\u7834\u68b0\u51a5\u5b98 \u7bc1"
UNCHAINED_ABOMINATION = "\u7834\u68b0\u7130\u9b54\u5929 \u960e\u6469"
CYBER_DRAGON = "\u5b8c\u7f8e\u7535\u5b50\u591a\u5143\u9a71\u52a8\u86c7\u00b7\u795e\u9f99"


class FakeLlm:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def _deterministic_preparation_actions():
    return [
        json.dumps(
            {
                "tool": "split_questions",
                "arguments": {"strategy": "deterministic"},
                "rationale": "Prepare authoritative question boundaries.",
            }
        ),
        json.dumps(
            {
                "tool": "extract_card_mentions",
                "arguments": {"strategy": "deterministic"},
                "rationale": "Prepare source-grounded card mentions.",
            }
        ),
    ]


def _with_active_revision_activity(
    agent,
    state,
    *,
    parser_tool: str,
):
    actions = (
        PlannerAction(tool=parser_tool, arguments={}),
        PlannerAction(tool="search_rules", arguments={"query": "old mechanism"}),
        PlannerAction(tool="critique_coverage", arguments={}),
        PlannerAction(tool="submit_ruling", arguments={"answer": {}}),
    )
    history = tuple(
        ToolHistoryEntry(
            step=index,
            action=action,
            observation=ToolObservation(
                tool=action.tool,
                ok=True,
                summary=f"old {action.tool}",
            ),
        )
        for index, action in enumerate(actions, start=1)
    )
    signatures = tuple(
        json.dumps(
            {"tool": action.tool, "arguments": action.arguments},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        for action in actions
    )
    trace = (
        *state.trace,
        *(
            TraceEntry(
                node="execute_tool",
                summary=f"old {action.tool}",
                metadata={"action": action.to_dict()},
            )
            for action in actions
        ),
    )
    return agent._replace(
        state,
        tool_history=history,
        tool_signatures=signatures,
        trace=trace,
    )


def _assert_active_revision_activity_was_reset(agent, state, *, parser_tool: str):
    assert [entry.action.tool for entry in state.tool_history] == [parser_tool]
    signature_tools = {
        json.loads(signature)["tool"] for signature in state.tool_signatures
    }
    assert signature_tools == {parser_tool}
    assert agent._budget_observation(
        state,
        PlannerAction(tool="search_rules", arguments={"query": "new mechanism"}),
    ) is None
    assert agent._budget_observation(
        state,
        PlannerAction(tool="critique_coverage", arguments={}),
    ) is None
    assert agent._budget_observation(
        state,
        PlannerAction(tool="submit_ruling", arguments={"answer": {}}),
    ) is None
    source_status = agent._source_status(state)
    assert source_status["faq"] == "not_searched"
    assert source_status["rules"] == "not_searched"
    audited_tools = {
        entry.metadata["action"]["tool"]
        for entry in state.trace
        if entry.node == "execute_tool" and "action" in entry.metadata
    }
    assert {
        parser_tool,
        "search_rules",
        "critique_coverage",
        "submit_ruling",
    }.issubset(audited_tools)


def _claim_report_payload(
    *,
    status: str = "supported",
    claim_id: str = "c1",
    evidence_id: str = "rule:1",
    entailment: str = "supports",
    relevance: str = "high",
    claim_type: str = "case_application",
    branch: str = "main",
    depends_on: list[str] | None = None,
    gaps: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "required_claims": [
            {
                "claim_id": claim_id,
                "claim": "Rule evidence entails the answer.",
                "claim_type": claim_type,
                "status": status,
                "question": "Can Alpha apply?",
                "branch": branch,
                "required": True,
                "decisive": True,
                "depends_on": depends_on or [],
                "evidence_checks": [
                    {
                        "evidence_id": evidence_id,
                        "relevance": relevance,
                        "entailment": entailment,
                        "reason": "The evidence is checked against the claim.",
                    }
                ],
                "gap": "" if status == "supported" else "Need more evidence.",
            }
        ],
        "gaps": gaps or [],
        "suggested_queries": [],
        "contradictions": [],
    }


def _answer_with_claim(
    claim_id: str,
    *,
    evidence_id: str = "rule:1",
    role: str = "rule_document",
    confidence: str | None = None,
    uncertainties: list[str] | None = None,
    hypothetical_conclusion: str = "",
) -> dict:
    payload = {
        "conclusion": "Use cited evidence.",
        "answers": [
            {
                "question": "Can Alpha apply?",
                "case_conclusion": "Use cited evidence.",
                "hypothetical_conclusion": hypothetical_conclusion,
                "reasoning": [
                    {
                        "judgment": "The claim supports this answer.",
                        "claim_id": claim_id,
                        "evidence_ids": [evidence_id],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": evidence_id,
                "role": role,
                "use": "Rule evidence.",
            }
        ],
        "uncertainties": uncertainties or [],
    }
    if confidence is not None:
        payload["confidence"] = confidence
    return payload


def test_initial_agent_state_is_serializable():
    state = create_initial_agent_state("Can Alpha apply?")

    payload = state.to_dict()

    assert payload["scenario_text"] == "Can Alpha apply?"
    assert payload["mention_queries"] == []
    assert payload["coverage"]["next_action"] == "extract"
    json.dumps(payload, ensure_ascii=False)


def test_deterministic_question_completeness_captures_whether_successfully_resolves():
    scenario = parse_scenario(
        "我方发动某个效果。随后对手发动虹光之宣告者的效果。此时虹光效果是否成功处理？"
    )
    completed = complete_scenario_questions(scenario)

    assert completed.questions == (
        "此时虹光效果是否成功处理？",
    )


def test_deterministic_parser_does_not_treat_generic_short_card_name_as_scenario_entity():
    scenario = parse_scenario(
        "「幻禄的天杯龙」的效果为什么没有被无效？",
        known_cards=(Card(1, "幻禄的天杯龙", ""), Card(2, "无效", "")),
    )

    assert scenario.card_mentions == ("幻禄的天杯龙",)


def test_deterministic_parser_does_not_treat_quoted_effect_text_as_card_name():
    scenario = parse_scenario(
        "「幻禄的天杯龙」发动『这张卡解放自己才能发动。之后从卡组特殊召唤。』的效果。",
        known_cards=(Card(1, "幻禄的天杯龙", ""),),
    )

    assert scenario.card_mentions == ("幻禄的天杯龙",)


@pytest.mark.parametrize("effect_text", ["无效", "不会被效果破坏", "这张卡不受效果影响"])
def test_deterministic_parser_does_not_treat_short_quoted_effect_as_card_name(effect_text):
    scenario = parse_scenario(
        f"「幻禄的天杯龙」发动『{effect_text}』。",
        known_cards=(Card(1, "幻禄的天杯龙", ""), Card(2, "无效", "")),
    )

    assert scenario.card_mentions == ("幻禄的天杯龙",)


def test_agent_initialization_performs_zero_llm_parsing_and_keeps_only_raw_source():
    scenario_text = (
        "我方发动某个效果。随后对手发动虹光之宣告者的效果。"
        "该效果能否发动？此时虹光效果是否成功处理？"
    )
    llm = FakeLlm(
        [
            json.dumps(
                {
                    "card_mentions": ["虹光之宣告者"],
                    "roles": ["我方", "对手"],
                    "effect_actions": ["发动"],
                    "objects": ["虹光之宣告者"],
                    "replacement_candidates": [],
                    "questions": ["该效果能否发动？"],
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "虹光之宣告者", "")],
        settings=_settings({"RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "1"}),
        llm=llm,
    )

    state = agent._initialize_agent_loop_state(
        create_initial_agent_state(scenario_text)
    )

    assert state.scenario is not None
    assert state.scenario.original_text == scenario_text
    assert state.scenario.card_mentions == ()
    assert state.scenario.questions == ()
    assert state.scenario.roles == ()
    assert state.scenario.effect_actions == ()
    assert state.mention_queries == ()
    assert llm.prompts == []
    assert state.trace[-1].metadata["scenario_source"] == "raw"
    assert state.scenario_field_sources["card_mentions"]["status"] == "pending"
    assert state.scenario_field_sources["questions"]["status"] == "pending"


def test_agent_loop_runs_focused_parser_tools_and_records_internal_llm_results():
    source = "Can Alpha Dragon activate?"
    llm = FakeLlm(
        [
            json.dumps(
                {
                    "tool": "split_questions",
                    "arguments": {"focus": "activation"},
                    "rationale": "Prepare questions.",
                }
            ),
            json.dumps({"questions": [source]}),
            json.dumps(
                {
                    "tool": "extract_card_mentions",
                    "arguments": {"focus": "Alpha"},
                    "rationale": "Prepare card mentions.",
                }
            ),
            json.dumps({"mentions": ["Alpha Dragon"]}),
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings({"RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT": "2"}),
        llm=llm,
    )

    response = agent.run(source)

    parser_steps = [
        entry
        for entry in response.agent_trace
        if entry["node"] == "execute_tool"
        and entry["metadata"]["action"]["tool"]
        in {"split_questions", "extract_card_mentions"}
    ]
    assert [entry["metadata"]["action"]["tool"] for entry in parser_steps] == [
        "split_questions",
        "extract_card_mentions",
    ]
    assert all(entry["metadata"]["observation"]["ok"] for entry in parser_steps)
    assert json.loads(
        parser_steps[0]["metadata"]["observation"]["data"]["llm_call"]["response"]
    ) == {"questions": [source]}
    assert json.loads(
        parser_steps[1]["metadata"]["observation"]["data"]["llm_call"]["response"]
    ) == {"mentions": ["Alpha Dragon"]}
    assert any("问题边界抽取器" in prompt for prompt in llm.prompts)
    assert any("卡名片段抽取器" in prompt for prompt in llm.prompts)


def test_split_questions_rejects_partial_multi_question_output_then_retries_full_set():
    source = "甲能发动吗？乙能适用吗？"
    llm = FakeLlm(
        [
            json.dumps({"questions": ["甲能发动吗？"]}, ensure_ascii=False),
            json.dumps(
                {"questions": ["甲能发动吗？", "乙能适用吗？"]},
                ensure_ascii=False,
            ),
        ]
    )
    agent = AdjudicationAgent([], settings=_settings(), llm=llm)
    initial = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    failed, failure = agent._execute_agent_tool(
        initial,
        PlannerAction(
            tool="split_questions",
            arguments={"focus": "检查第二问"},
        ),
    )
    corrected, success = agent._execute_agent_tool(
        failed,
        PlannerAction(
            tool="split_questions",
            arguments={"focus": "检查全部问题"},
        ),
    )

    assert failed is initial
    assert failure.ok is False
    assert "omitted a deterministic source question boundary" in failure.validation_error
    assert initial.scenario.questions == ()
    assert success.ok is True
    assert corrected.scenario.questions == ("甲能发动吗？", "乙能适用吗？")
    assert corrected.scenario_field_sources["questions"]["status"] == "ready"


def test_split_questions_prepares_full_source_when_no_question_boundary_exists():
    source = "Alpha Dragon activation timing requires review"
    prompt = build_question_split_prompt(source)
    assert "把完整原文原样作为唯一 issue 返回" in prompt
    assert "不得添加问号、改写、概括或补充答案" in prompt

    llm = FakeLlm(
        [
            json.dumps({"questions": [source]}),
            json.dumps({"questions": ["Alpha Dragon activation timing"]}),
        ]
    )
    llm_agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings(),
        llm=llm,
    )
    initial = llm_agent._initialize_agent_loop_state(
        create_initial_agent_state(source)
    )
    prepared, success = llm_agent._execute_agent_tool(
        initial,
        PlannerAction(tool="split_questions", arguments={}),
    )
    unchanged, rejected_summary = llm_agent._execute_agent_tool(
        prepared,
        PlannerAction(
            tool="split_questions",
            arguments={"focus": "activation timing"},
        ),
    )

    assert success.ok is True
    assert prepared.scenario.questions == (source,)
    assert prepared.scenario_field_sources["questions"]["status"] == "ready"
    assert rejected_summary.ok is False
    assert "exact full source" in rejected_summary.validation_error
    assert unchanged is prepared


def test_split_questions_invalid_llm_output_uses_exact_source_only_for_initial_no_boundary_case():
    source = "Alpha Dragon activation timing requires review"
    llm = FakeLlm([json.dumps({"questions": ["rewritten question"]})])
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings(),
        llm=llm,
    )
    initial = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    prepared, observation = agent._execute_agent_tool(
        initial,
        PlannerAction(tool="split_questions", arguments={}),
    )

    assert observation.ok is True
    assert observation.data["strategy"] == "deterministic_fallback"
    assert observation.data["llm_call"]["response"]
    assert prepared.scenario.questions == (source,)

    deterministic_agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings(),
    )
    deterministic = deterministic_agent._initialize_agent_loop_state(
        create_initial_agent_state(source)
    )
    deterministic, mention_result = deterministic_agent._execute_agent_tool(
        deterministic,
        PlannerAction(
            tool="extract_card_mentions",
            arguments={"strategy": "deterministic"},
        ),
    )
    deterministic, split_result = deterministic_agent._execute_agent_tool(
        deterministic,
        PlannerAction(
            tool="split_questions",
            arguments={"strategy": "deterministic"},
        ),
    )
    coverage = deterministic_agent._check_coverage(deterministic)

    assert mention_result.ok is True
    assert split_result.ok is True
    assert deterministic.scenario.questions == (source,)
    assert deterministic.scenario_field_sources["questions"]["status"] == "ready"
    assert not any(
        gap.startswith("Scenario preparation gap:")
        for gap in coverage.missing_evidence
    )
    assert source in coverage.missing_evidence


def test_split_questions_promotes_full_source_when_question_words_have_no_terminal_boundary():
    source = "Alpha 在场上存在时是否能直接攻击"

    assert validate_question_split_payload(
        {"questions": [source]},
        source,
    ) == (source,)


@pytest.mark.parametrize(
    ("source", "candidate", "accepted"),
    [
        (
            "Alpha-Dragon effect review",
            "Alpha-Dragon effect review",
            True,
        ),
        (
            "Alpha-Dragon effect review",
            "Alpha Dragon effect review",
            False,
        ),
        (
            "幻禄·天杯龙效果确认",
            "幻禄 天杯龙效果确认",
            False,
        ),
        (
            "Alpha Dragon effect review",
            "Alpha  Dragon effect review",
            False,
        ),
    ],
)
def test_no_boundary_question_fallback_requires_exact_full_source(
    source,
    candidate,
    accepted,
):
    payload = {"questions": [candidate]}

    if accepted:
        assert validate_question_split_payload(payload, source) == (source,)
    else:
        with pytest.raises(ValueError, match="exact full source"):
            validate_question_split_payload(payload, source)


def test_corrected_question_replaces_prior_value_and_invalidates_claim_state():
    source = "前提成立时甲能发动吗？"
    llm = FakeLlm(
        [
            json.dumps({"questions": ["甲能发动吗？"]}, ensure_ascii=False),
            json.dumps({"questions": [source]}, ensure_ascii=False),
        ]
    )
    agent = AdjudicationAgent(
        [],
        settings=_settings(
            {
                "RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT": "1",
                "RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT": "1",
                "RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT": "1",
            }
        ),
        llm=llm,
    )
    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))
    state, first = agent._execute_agent_tool(
        state,
        PlannerAction(tool="split_questions", arguments={}),
    )
    assert first.ok is True
    state = _with_active_revision_activity(
        agent,
        state,
        parser_tool="split_questions",
    )
    state = agent._replace(
        state,
        evidence=(
            EvidenceItem(
                evidence_id="rule:old",
                kind="rule_passage",
                source="old",
                text="Old question evidence.",
                relationship="rule_document",
            ),
        ),
        evidence_retrieval_stages=({"stage": "rules"},),
        rule_chapter_candidates=({"chapter_id": "old"},),
        coverage=CoverageReport(sufficient=True),
        claim_coverage_report=ClaimCoverageReport(),
        claim_coverage_aggregation=ClaimCoverageAggregation(sufficient=True),
        final_answer="stale",
        agent_loop_complete=True,
    )

    corrected, second = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="split_questions",
            arguments={"focus": "保留前提"},
        ),
    )

    assert second.ok is True
    assert corrected.scenario.questions == (source,)
    assert corrected.evidence == ()
    assert corrected.evidence_retrieval_stages == ()
    assert corrected.rule_chapter_candidates == ()
    assert corrected.claim_coverage_report is None
    assert corrected.claim_coverage_aggregation is None
    assert corrected.coverage.sufficient is False
    assert corrected.final_answer is None
    assert corrected.agent_loop_complete is False
    _assert_active_revision_activity_was_reset(
        agent,
        corrected,
        parser_tool="split_questions",
    )


def test_card_mention_correction_clears_entity_dependent_state():
    source = "Alpha Dragon 与 Beta Serpent 能发动吗？"
    llm = FakeLlm(
        [
            json.dumps({"mentions": ["Alpha Dragon"]}),
            json.dumps({"mentions": ["Beta Serpent"]}),
        ]
    )
    cards = [
        Card(1, "Alpha Dragon", "Alpha effect."),
        Card(2, "Beta Serpent", "Beta effect."),
    ]
    agent = AdjudicationAgent(
        cards,
        settings=_settings(
            {
                "RAG_ADJUDICATION_AGENT_RETRIEVAL_LIMIT": "1",
                "RAG_ADJUDICATION_AGENT_CRITIQUE_LIMIT": "1",
                "RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT": "1",
            }
        ),
        llm=llm,
    )
    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))
    state, first = agent._execute_agent_tool(
        state,
        PlannerAction(tool="extract_card_mentions", arguments={}),
    )
    assert first.ok is True
    state = _with_active_revision_activity(
        agent,
        state,
        parser_tool="extract_card_mentions",
    )
    state = agent._replace(
        state,
        card_resolutions=(
            CardResolution(mention="Alpha Dragon", status="resolved"),
        ),
        resolved_cards=(
            ResolvedCard(mention="Alpha Dragon", matches=(cards[0],)),
        ),
        evidence=(
            EvidenceItem(
                evidence_id="card:1",
                kind="card_effect",
                source="local",
                text="Alpha effect.",
                relationship="direct_card",
            ),
        ),
        evidence_retrieval_stages=({"stage": "direct_faq"},),
        rule_chapter_candidates=({"chapter_id": "old"},),
        coverage=CoverageReport(sufficient=True),
        claim_coverage_report=ClaimCoverageReport(),
        claim_coverage_aggregation=ClaimCoverageAggregation(sufficient=True),
        final_answer="stale",
    )

    corrected, second = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="extract_card_mentions",
            arguments={"focus": "Beta"},
        ),
    )

    assert second.ok is True
    assert corrected.scenario.card_mentions == ("Beta Serpent",)
    assert corrected.mention_queries == ("Beta Serpent",)
    assert corrected.card_resolutions == ()
    assert corrected.resolved_cards == ()
    assert corrected.evidence == ()
    assert corrected.evidence_retrieval_stages == ()
    assert corrected.rule_chapter_candidates == ()
    assert corrected.claim_coverage_report is None
    assert corrected.coverage.sufficient is False
    assert corrected.final_answer is None
    _assert_active_revision_activity_was_reset(
        agent,
        corrected,
        parser_tool="extract_card_mentions",
    )


def test_agent_loop_blocks_evidence_coverage_and_submit_before_preparation():
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings(),
        rule_index=RuleIndex([]),
    )
    state = agent._initialize_agent_loop_state(
        create_initial_agent_state("Can Alpha Dragon activate?")
    )

    _, search_card = agent._execute_agent_tool(
        state,
        PlannerAction(tool="search_card", arguments={"query": "Alpha Dragon"}),
    )
    assert search_card.ok is True
    _, classify = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="classify_rule_chapters",
            arguments={"query": "activation timing"},
        ),
    )
    assert classify.ok is True
    prompt = build_agent_planner_prompt(state)
    assert "准备完成前仅允许提前调用 search_card 和 classify_rule_chapters" in prompt
    assert "调用 resolve_cards、FAQ/supplement/analogous/rules 证据检索" in prompt

    for action in (
        PlannerAction(tool="resolve_cards", arguments={"mentions": ["Alpha Dragon"]}),
        PlannerAction(tool="search_direct_faq", arguments={}),
        PlannerAction(tool="search_related_faq", arguments={}),
        PlannerAction(tool="search_card_supplement", arguments={}),
        PlannerAction(
            tool="search_analogous_faq",
            arguments={"query": "activation timing"},
        ),
        PlannerAction(tool="search_rules", arguments={"query": "activation timing"}),
        PlannerAction(tool="critique_coverage", arguments={}),
        PlannerAction(tool="submit_ruling", arguments={"answer": {}}),
    ):
        blocked_state, observation = agent._execute_agent_tool(state, action)
        assert blocked_state is state
        assert observation.ok is False
        assert observation.summary == "scenario preparation required"
        assert observation.data["missing_preparation"] == [
            "extract_card_mentions",
            "split_questions",
        ]


def test_deterministic_card_strategy_recovers_after_malformed_llm_output():
    source = '"Alpha Dragon" can activate?'
    llm = FakeLlm([json.dumps({"mentions": [{"text": "Alpha Dragon"}]})])
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "")],
        settings=_settings(),
        llm=llm,
    )
    initial = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    failed, failure = agent._execute_agent_tool(
        initial,
        PlannerAction(tool="extract_card_mentions", arguments={}),
    )
    recovered, success = agent._execute_agent_tool(
        failed,
        PlannerAction(
            tool="extract_card_mentions",
            arguments={"strategy": "deterministic"},
        ),
    )

    assert failed is initial
    assert failure.ok is False
    assert "only strings" in failure.validation_error
    assert success.ok is True
    assert recovered.scenario.card_mentions == ("Alpha Dragon",)
    assert recovered.mention_queries == ("Alpha Dragon",)
    assert recovered.extracted_card_mentions[0].source == "deterministic"


def test_fuzzy_source_mention_reaches_unresolved_resolution_and_targeted_search():
    source = '"幻禄的天杯虫"能发动吗？'
    llm = FakeLlm(
        [json.dumps({"mentions": ["幻禄的天杯虫"]}, ensure_ascii=False)]
    )
    agent = AdjudicationAgent([], settings=_settings(), llm=llm)
    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))
    state, split_result = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="split_questions",
            arguments={"strategy": "deterministic"},
        ),
    )
    state, extraction = agent._execute_agent_tool(
        state,
        PlannerAction(tool="extract_card_mentions", arguments={}),
    )
    state, resolution = agent._execute_agent_tool(
        state,
        PlannerAction(tool="resolve_cards", arguments={}),
    )
    _, targeted = agent._execute_agent_tool(
        state,
        PlannerAction(tool="search_card", arguments={"query": "幻禄的天杯虫"}),
    )

    assert split_result.ok is True
    assert extraction.ok is True
    assert state.mention_queries == ("幻禄的天杯虫",)
    assert resolution.ok is True
    assert state.card_resolutions[0].status == "unresolved"
    assert targeted.ok is True
    assert targeted.data["query"] == "幻禄的天杯虫"


def test_scenario_fact_extraction_is_per_field_and_rejects_rulings_or_warnings():
    source = "我方让 Alpha Dragon 发动效果，会造成贯穿伤害吗？"
    llm = FakeLlm(
        [
            json.dumps({"values": ["我方"]}, ensure_ascii=False),
            json.dumps({"values": ["Alpha Dragon"]}),
            json.dumps({"values": ["会造成贯穿伤害"]}, ensure_ascii=False),
            json.dumps({"values": ["发动"], "warnings": []}, ensure_ascii=False),
        ]
    )
    agent = AdjudicationAgent([], settings=_settings(), llm=llm)
    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))
    state = agent._replace(
        state,
        scenario=ParsedScenario(
            original_text=source,
            card_mentions=("Alpha Dragon",),
        ),
    )

    state, roles = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="extract_scenario_facts",
            arguments={"field": "roles"},
        ),
    )
    state, objects = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="extract_scenario_facts",
            arguments={"field": "objects"},
        ),
    )
    unchanged, ruling = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="extract_scenario_facts",
            arguments={"field": "effect_actions"},
        ),
    )
    unchanged_again, warning = agent._execute_agent_tool(
        unchanged,
        PlannerAction(
            tool="extract_scenario_facts",
            arguments={"field": "effect_actions"},
        ),
    )

    assert roles.ok is True
    assert objects.ok is True
    assert state.scenario.roles == ("我方",)
    assert state.scenario.objects == ("Alpha Dragon",)
    assert ruling.ok is False
    assert unchanged is state
    assert warning.ok is False
    assert unchanged_again is state
    assert state.scenario.effect_actions == ()


def _scenario_parser_payload(**overrides):
    payload = {
        "card_mentions": ["虹光之宣告者"],
        "roles": ["我方", "对手"],
        "effect_actions": ["发动"],
        "objects": ["虹光之宣告者"],
        "replacement_candidates": [],
        "questions": ["该效果能否发动？"],
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _parse_llm_payload(source, payload, *, known_cards=()):
    return parse_scenario_with_llm(
        source,
        FakeLlm([json.dumps(payload, ensure_ascii=False)]),
        known_cards=known_cards,
    )


def test_scenario_parser_prompt_forbids_rulings_and_requires_extractive_json():
    prompt = build_scenario_parser_prompt("虹光之宣告者的效果能否发动？")

    assert "绝不裁定、绝不回答问题、绝不补充规则知识" in prompt
    assert "逐字抽取" in prompt
    assert "必须能在用户原文中定位" in prompt
    assert "warnings 必须是空数组" in prompt
    assert "不得增加 answer 等字段" in prompt


def test_llm_scenario_parser_rejects_object_card_mention_instead_of_stringifying_it():
    payload = _scenario_parser_payload(card_mentions=[{"name": "虹光之宣告者"}])

    with pytest.raises(ValueError, match="card_mentions.*only strings"):
        _parse_llm_payload(
            "我方发动某个效果。随后对手发动虹光之宣告者的效果。该效果能否发动？",
            payload,
        )


def test_agent_initialization_does_not_consume_fabricated_monolithic_parse_output():
    source = "我方发动虹光之宣告者的效果。该效果能否发动？"
    llm = FakeLlm(
        [
            json.dumps(
                _scenario_parser_payload(
                    card_mentions=[{"name": "污染卡名"}],
                    effect_actions=["发动", "会造成贯穿伤害"],
                ),
                ensure_ascii=False,
            )
        ]
    )
    agent = AdjudicationAgent(
        [],
        settings=_settings({"RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "1"}),
        llm=llm,
    )

    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    assert state.scenario is not None
    assert all("污染卡名" not in mention for mention in state.scenario.card_mentions)
    assert "会造成贯穿伤害" not in state.scenario.effect_actions
    assert state.trace[-1].metadata["scenario_source"] == "raw"
    assert llm.prompts == []
    assert state.warnings == ()


@pytest.mark.parametrize(
    ("payload_update", "error_match"),
    [
        ({"answer": "可以发动"}, "unexpected fields: answer"),
        ({"effect_actions": [{"action": "发动"}]}, "effect_actions.*only strings"),
        ({"questions": [{"question": "该效果能否发动？", "answer": "可以"}]}, "questions.*only strings"),
    ],
)
def test_llm_scenario_parser_rejects_answer_action_and_question_object_pollution(
    payload_update,
    error_match,
):
    payload = _scenario_parser_payload()
    payload.update(payload_update)

    with pytest.raises(ValueError, match=error_match):
        _parse_llm_payload(
            "我方发动某个效果。随后对手发动虹光之宣告者的效果。该效果能否发动？",
            payload,
        )


@pytest.mark.parametrize(
    "payload_update",
    [
        {"questions": ["该效果可以发动"]},
        {"effect_actions": ["会造成贯穿伤害"]},
        {"effect_actions": ["以实际卡名为准"]},
        {"warnings": ["该效果可以发动"]},
    ],
)
def test_llm_scenario_parser_rejects_string_rulings_and_nonempty_warnings(payload_update):
    payload = _scenario_parser_payload()
    payload.update(payload_update)

    with pytest.raises(ValueError):
        _parse_llm_payload(
            "我方发动某个效果。随后对手发动虹光之宣告者的效果。该效果能否发动？",
            payload,
        )


def test_llm_scenario_parser_rejects_source_grounded_declarative_answer_as_question():
    source = "如果这个效果可以发动，对方还能特殊召唤吗？"
    payload = {
        "card_mentions": [],
        "roles": ["对方"],
        "effect_actions": ["发动", "特殊召唤"],
        "objects": [],
        "replacement_candidates": [],
        "questions": ["这个效果可以发动"],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="questions must be source question fragments"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_rejects_source_grounded_english_statement_as_question():
    source = "Alpha is negated. Can Beta summon?"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": [],
        "objects": [],
        "replacement_candidates": [],
        "questions": ["Alpha is negated"],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="questions must be source question fragments"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_rejects_chinese_embedded_question_statement():
    source = "官方文档已经说明该效果是否发动的判断标准。"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["发动"],
        "objects": [],
        "replacement_candidates": [],
        "questions": [source],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="questions must be source question fragments"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_rejects_answer_appended_after_question_mark():
    source = "官方FAQ写着：能否发动？可以发动。"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["发动"],
        "objects": [],
        "replacement_candidates": [],
        "questions": ["能否发动？可以发动"],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="questions must be source question fragments"):
        _parse_llm_payload(source, payload)


def test_subject_led_chinese_question_without_question_mark_is_kept_as_specific_clause():
    source = "我方发动效果。该效果是否适用"
    scenario = parse_scenario(source)

    assert scenario.questions == ("该效果是否适用",)


def test_llm_scenario_parser_defers_question_without_mark_to_deterministic_completion():
    source = "该效果是否适用"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["适用"],
        "objects": [],
        "replacement_candidates": [],
        "questions": [],
        "warnings": [],
    }

    scenario = _parse_llm_payload(source, payload)

    assert scenario.questions == (source,)


def test_llm_scenario_parser_rejects_unmarked_ruling_statement_as_question():
    source = "裁定结论是该效果是否适用取决于具体条件。"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["适用"],
        "objects": [],
        "replacement_candidates": [],
        "questions": ["裁定结论是该效果是否适用取决于具体条件"],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="exactly one terminal question boundary"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_rejects_answer_sentence_prepended_to_question():
    source = "裁定结论是可以发动。问题是能否发动？"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["发动"],
        "objects": [],
        "replacement_candidates": [],
        "questions": [source],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="deterministic source question boundary"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_rejects_answer_clause_prepended_to_question():
    source = "裁定结论是可以发动，问题是能否发动？"
    payload = {
        "card_mentions": [],
        "roles": [],
        "effect_actions": ["发动"],
        "objects": [],
        "replacement_candidates": [],
        "questions": [source],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="deterministic source question boundary"):
        _parse_llm_payload(source, payload)

    assert parse_scenario(source).questions == ("问题是能否发动？",)


def test_llm_scenario_parser_rejects_source_grounded_answer_as_card_mention():
    source = "如果这个效果可以发动，对方还能特殊召唤吗？"
    payload = {
        "card_mentions": ["可以发动"],
        "roles": ["对方"],
        "effect_actions": ["发动", "特殊召唤"],
        "objects": [],
        "replacement_candidates": [],
        "questions": ["对方还能特殊召唤吗？"],
        "warnings": [],
    }

    with pytest.raises(ValueError, match="plausible card-name strings"):
        _parse_llm_payload(source, payload)


def test_llm_scenario_parser_accepts_real_case_seven_extractive_shape():
    source = (
        "我方场上存在「美丽的磁律机坏」，对方在「美丽的磁律机坏」连接区的"
        "「幻禄的天杯龙」的解放自己，从卡组特招天杯龙的效果为什么没有被无效？"
    )
    payload = {
        "card_mentions": ["美丽的磁律机坏", "幻禄的天杯龙"],
        "roles": ["我方", "对方"],
        "effect_actions": ["解放", "特招", "无效"],
        "objects": ["幻禄的天杯龙"],
        "replacement_candidates": [],
        "questions": ["从卡组特招天杯龙的效果为什么没有被无效？"],
        "warnings": [],
    }

    scenario = _parse_llm_payload(
        source,
        payload,
        known_cards=(
            Card(1, "美丽的磁律机坏", ""),
            Card(2, "幻禄的天杯龙", ""),
        ),
    )

    assert scenario.card_mentions == ("美丽的磁律机坏", "幻禄的天杯龙")
    assert scenario.effect_actions == ("解放", "特招", "无效")
    assert scenario.questions == ("从卡组特招天杯龙的效果为什么没有被无效？",)


@pytest.mark.parametrize("field_name", ["card_mentions", "objects", "replacement_candidates"])
def test_llm_scenario_parser_rejects_case_seven_answer_phrase_in_entity_fields(field_name):
    source = (
        "我方场上存在「美丽的磁律机坏」，对方在「美丽的磁律机坏」连接区的"
        "「幻禄的天杯龙」的解放自己，从卡组特招天杯龙的效果为什么没有被无效？"
    )
    payload = {
        "card_mentions": ["美丽的磁律机坏", "幻禄的天杯龙"],
        "roles": ["我方", "对方"],
        "effect_actions": ["解放", "特招", "无效"],
        "objects": ["幻禄的天杯龙"],
        "replacement_candidates": [],
        "questions": ["从卡组特招天杯龙的效果为什么没有被无效？"],
        "warnings": [],
    }
    payload[field_name] = ["没有被无效"]

    with pytest.raises(ValueError):
        _parse_llm_payload(
            source,
            payload,
            known_cards=(
                Card(1, "美丽的磁律机坏", ""),
                Card(2, "幻禄的天杯龙", ""),
            ),
        )


def test_llm_scenario_parser_preserves_unquoted_fuzzy_mention_explained_by_known_card():
    source = "幻禄的天杯龙发动效果时，该效果能否成功处理？"
    payload = {
        "card_mentions": ["幻禄的天杯龙"],
        "roles": [],
        "effect_actions": ["发动", "处理"],
        "objects": ["幻禄的天杯龙"],
        "replacement_candidates": [],
        "questions": ["该效果能否成功处理？"],
        "warnings": [],
    }

    scenario = _parse_llm_payload(
        source,
        payload,
        known_cards=(Card(1, "幻禄之天杯龙", ""),),
    )

    assert scenario.card_mentions == ("幻禄的天杯龙",)


def test_agent_initialization_does_not_preload_fuzzy_card_fact():
    source = "幻禄的天杯龙发动效果时，该效果能否成功处理？"
    llm = FakeLlm(
        [
            json.dumps(
                {
                    "card_mentions": ["幻禄的天杯龙"],
                    "roles": [],
                    "effect_actions": ["发动", "处理"],
                    "objects": ["幻禄的天杯龙"],
                    "replacement_candidates": [],
                    "questions": ["该效果能否成功处理？"],
                    "warnings": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "幻禄之天杯龙", "")],
        settings=_settings({"RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "1"}),
        llm=llm,
    )

    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    assert state.scenario is not None
    assert state.scenario.card_mentions == ()
    assert state.scenario.effect_actions == ()
    assert state.scenario.objects == ()
    assert llm.prompts == []


def test_agent_initialization_never_calls_unavailable_scenario_parser_provider():
    class FailingLlm:
        def invoke(self, prompt):
            raise RuntimeError("timeout")

    source = "我方发动虹光之宣告者的效果。该效果能否发动？"
    agent = AdjudicationAgent(
        [],
        settings=_settings({"RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "1"}),
        llm=FailingLlm(),
    )

    state = agent._initialize_agent_loop_state(create_initial_agent_state(source))

    assert state.scenario is not None
    assert state.trace[-1].metadata["scenario_source"] == "raw"
    assert state.scenario.card_mentions == ()
    assert state.scenario.questions == ()
    assert state.warnings == ()


def test_legacy_extract_node_falls_back_when_llm_scenario_parser_provider_fails():
    class FailingLlm:
        def invoke(self, prompt):
            raise RuntimeError("timeout")

    source = "我方发动虹光之宣告者的效果。该效果能否发动？"
    agent = AdjudicationAgent(
        [],
        settings=_settings(),
        llm=FailingLlm(),
    )

    payload = agent._node_extract({"agent_state": create_initial_agent_state(source)})
    state = payload["agent_state"]

    assert state.scenario is not None
    assert any("timeout" in warning for warning in state.warnings)


def test_llm_scenario_parser_keeps_source_mentions_and_completes_multiple_questions():
    source = (
        "我方发动某个效果。随后对手发动「虹光之 宣告者」的效果。"
        "该效果能否发动？此时虹光效果是否成功处理？"
    )
    scenario = _parse_llm_payload(
        source,
        _scenario_parser_payload(
            card_mentions=["虹光之·宣告者"],
            objects=["虹光之·宣告者"],
        ),
        known_cards=(Card(1, "虹光之宣告者", ""),),
    )

    assert scenario.card_mentions == ("虹光之·宣告者",)
    assert "该效果能否发动？" in scenario.questions
    assert any("是否成功处理" in question for question in scenario.questions)
    assert any("completeness checker" in warning for warning in scenario.warnings)


def test_question_completion_deduplicates_llm_questions_against_contextual_fragments():
    scenario = ParsedScenario(
        original_text=(
            "双方场上都只有一只怪兽的时候，对方发动了手卡破械冥官·篁的效果，"
            "要将场上的破械焰魔天·阎摩破坏，此时对方选择适用破械焰魔天·阎摩的效果"
            "想要破坏我方场上的完美电子多元驱动蛇·神龙，我方场上的完美电子多元驱动蛇·神龙"
            "可以作为被破坏的替代降低1000攻击力吗？如果适用降低1000攻击力，"
            "对方的破械冥官·篁还能特殊召唤吗？"
        ),
        questions=(
            "完美电子多元驱动蛇·神龙可以作为被破坏的替代降低1000攻击力吗？",
            "如果适用降低1000攻击力，对方的破械冥官·篁还能特殊召唤吗？",
            (
                "双方场上都只有一只怪兽的时候，对方发动了手卡破械冥官·篁的效果，"
                "要将场上的破械焰魔天·阎摩破坏，此时对方选择适用破械焰魔天·阎摩的效果"
                "想要破坏我方场上的完美电子多元驱动蛇·神龙，我方场上的完美电子多元驱动蛇·神龙"
                "可以作为被破坏的替代降低1000攻击力吗"
            ),
            "如果适用降低1000攻击力，对方的破械冥官·篁还能特殊召唤吗",
        ),
    )

    completed = complete_scenario_questions(scenario)

    assert completed.questions == (
        "完美电子多元驱动蛇·神龙可以作为被破坏的替代降低1000攻击力吗？",
        "如果适用降低1000攻击力，对方的破械冥官·篁还能特殊召唤吗？",
    )


def test_claim_coverage_report_parser_validates_schema_and_dependencies():
    report = parse_claim_coverage_report(
        json.dumps(
            {
                "schema_version": 1,
                "required_claims": [
                    {
                        "claim_id": "c1",
                        "claim": "Card text establishes the premise.",
                        "claim_type": "card_text",
                        "status": "supported",
                        "question": "Can Alpha apply?",
                        "branch": "main",
                        "required": True,
                        "decisive": True,
                        "depends_on": [],
                        "evidence_checks": [
                            {
                                "evidence_id": "rule:1",
                                "relevance": "high",
                                "entailment": "supports",
                                "reason": "Directly proves the premise.",
                            }
                        ],
                        "gap": "",
                    },
                    {
                        "claim_id": "c2",
                        "claim": "The premise applies to this case.",
                        "claim_type": "case_application",
                        "status": "supported",
                        "question": "Can Alpha apply?",
                        "branch": "main",
                        "required": True,
                        "decisive": True,
                        "depends_on": ["c1"],
                        "evidence_checks": [
                            {
                                "evidence_id": "rule:1",
                                "relevance": "high",
                                "entailment": "supports",
                                "reason": "Applies the premise.",
                            }
                        ],
                        "gap": "",
                    },
                ],
                "gaps": [],
                "suggested_queries": [],
                "contradictions": [],
            }
        ),
        allowed_evidence_ids={"rule:1"},
    )

    assert report.required_claims[1].depends_on == ("c1",)
    json.dumps(report.to_dict(), ensure_ascii=False)

    invalid_reports = [
        "not json",
        json.dumps(_claim_report_payload(status="invalid")),
        json.dumps(
            {
                "required_claims": [
                    {
                        "claim_id": "c1",
                        "claim": "Missing claim type.",
                        "status": "supported",
                        "evidence_checks": [],
                    }
                ]
            }
        ),
        json.dumps(
            _claim_report_payload(
                depends_on=["missing"],
            )
        ),
        json.dumps(_claim_report_payload(evidence_id="missing:e1")),
    ]
    for payload in invalid_reports:
        try:
            parse_claim_coverage_report(payload, allowed_evidence_ids={"rule:1"})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid claim coverage report should be rejected")


def test_claim_coverage_semantics_reject_main_hypothetical_and_missing_case_application():
    question = "Can Alpha apply?"
    state = replace(
        create_initial_agent_state(question),
        scenario=ParsedScenario(original_text=question, questions=(question,)),
    )
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="claim.main.hypothetical.alpha",
                claim="Alpha is a monster and therefore the effect applies.",
                claim_type="hypothetical_branch",
                status="supported",
                issue_id="issue.alpha",
                question=question,
                branch="main",
                decisive=True,
            ),
        ),
        issues=(
            ClaimCoverageIssue(
                issue_id="issue.alpha",
                question=question,
                status="supported",
                claim_ids=("claim.main.hypothetical.alpha",),
            ),
        ),
    )

    defects = claim_coverage_semantic_defects(report, state)

    assert any("缺少" in defect and "case_application" in defect for defect in defects)
    assert any("hypothetical_branch" in defect for defect in defects)


def test_claim_coverage_semantics_accepts_decisive_case_application():
    question = "Can Alpha apply?"
    state = replace(
        create_initial_agent_state(question),
        scenario=ParsedScenario(original_text=question, questions=(question,)),
    )
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="claim.main.case_application.alpha",
                claim="The rule applies to Alpha in the described chain.",
                claim_type="case_application",
                status="supported",
                issue_id="issue.alpha",
                question=question,
                branch="main",
                decisive=True,
            ),
        ),
        issues=(
            ClaimCoverageIssue(
                issue_id="issue.alpha",
                question=question,
                status="supported",
                claim_ids=("claim.main.case_application.alpha",),
            ),
        ),
    )

    assert claim_coverage_semantic_defects(report, state) == ()


def test_repeated_critique_is_rejected_when_evidence_revision_is_unchanged():
    evidence = EvidenceItem(
        evidence_id="rule:1",
        kind="rule_passage",
        source="ocgrules",
        text="Rule text.",
        relationship="rule_document",
    )
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="Apply the rule.",
                claim_type="case_application",
                status="unknown",
            ),
        )
    )
    state = replace(
        create_initial_agent_state("Can Alpha apply?"),
        evidence=(evidence,),
        claim_coverage_report=report,
    )
    revision = evidence_revision(state)
    state = replace(
        state,
        tool_history=(
            ToolHistoryEntry(
                step=1,
                action=PlannerAction(tool="critique_coverage", arguments={}),
                observation=ToolObservation(
                    tool="critique_coverage",
                    ok=True,
                    summary="checked",
                    data={"evidence_revision": revision},
                ),
            ),
        ),
    )
    agent = AdjudicationAgent([], settings=_settings())

    observation = agent._repeated_critique_observation(
        state,
        PlannerAction(tool="critique_coverage", arguments={}),
    )

    assert observation is not None
    assert observation.ok is False
    assert observation.data["evidence_revision"] == revision


def test_claim_coverage_aggregation_handles_statuses_and_entailment():
    supported = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="Supported claim.",
                claim_type="case_application",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="rule:1",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
        )
    )
    assert aggregate_claim_coverage(supported).sufficient is False
    assert (
        aggregate_claim_coverage(
            supported,
            evidence_by_id={
                "rule:1": EvidenceItem(
                    evidence_id="rule:1",
                    kind="rule_passage",
                    source="ocgrules",
                    text="Rule text.",
                    relationship="rule_document",
                )
            },
        ).sufficient
        is True
    )

    expected_blocked = {
        "partially_supported": "partial_claims",
        "unsupported": "unsupported_claims",
        "contradicted": "contradicted_claims",
        "unknown": "unknown_claims",
    }
    for status, field_name in expected_blocked.items():
        report = ClaimCoverageReport(
            required_claims=(
                ClaimCoverageClaim(
                    claim_id=f"c-{status}",
                    claim=f"{status} claim.",
                    claim_type="case_application",
                    status=status,
                    evidence_checks=(
                        ClaimEvidenceCheck(
                            evidence_id="rule:1",
                            relevance="high",
                            entailment=(
                                "partially_supports"
                                if status == "partially_supported"
                                else "contradicts"
                                if status == "contradicted"
                                else "does_not_entail"
                            ),
                        ),
                    ),
                ),
            )
        )
        aggregation = aggregate_claim_coverage(report)
        assert aggregation.sufficient is False
        assert getattr(aggregation, field_name)


def test_relevant_evidence_without_entailment_does_not_make_claim_sufficient():
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="Topically similar evidence proves the answer.",
                claim_type="case_application",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="faq:1",
                        relevance="high",
                        entailment="does_not_entail",
                        reason="Same mechanism, but it does not prove this conclusion.",
                    ),
                ),
            ),
        )
    )

    aggregation = aggregate_claim_coverage(report)

    assert aggregation.sufficient is False
    assert aggregation.unsupported_claims == ("c1",)


@pytest.mark.parametrize("claim_type", ["case_application", "ruling_fact"])
def test_decisive_ruling_claim_rejects_card_effect_only_support(claim_type):
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="The card text applies to this ruling case.",
                claim_type=claim_type,
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="card:1:effect",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
        )
    )
    aggregation = aggregate_claim_coverage(
        report,
        evidence_by_id={
            "card:1:effect": EvidenceItem(
                evidence_id="card:1:effect",
                kind="card_effect",
                source="cards.cdb",
                text="Card text.",
                relationship="resolved_card",
            )
        },
    )

    assert aggregation.sufficient is False
    assert aggregation.unsupported_claims == ("c1",)

    card_text_report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c2",
                claim="The card has this text.",
                claim_type="card_text",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="card:1:effect",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
        )
    )
    assert (
        aggregate_claim_coverage(
            card_text_report,
            evidence_by_id={
                "card:1:effect": EvidenceItem(
                    evidence_id="card:1:effect",
                    kind="card_effect",
                    source="cards.cdb",
                    text="Card text.",
                    relationship="resolved_card",
                )
            },
        ).sufficient
        is True
    )


@pytest.mark.parametrize("claim_type", ["case_application", "ruling_fact"])
def test_decisive_ruling_claim_rejects_dependency_only_support(claim_type):
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="rule",
                claim="The rule establishes the mechanism.",
                claim_type="rule_principle",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="rule:1",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
            ClaimCoverageClaim(
                claim_id="application",
                claim="The mechanism applies to this case.",
                claim_type=claim_type,
                status="supported",
                depends_on=("rule",),
                inference_reason="The supported mechanism entails the application.",
            ),
        )
    )

    aggregation = aggregate_claim_coverage(
        report,
        evidence_by_id={
            "rule:1": EvidenceItem(
                evidence_id="rule:1",
                kind="rule_passage",
                source="ocgrules",
                text="Rule text.",
                relationship="rule_document",
            )
        },
    )

    assert aggregation.sufficient is False
    assert aggregation.unsupported_claims == ("application",)


@pytest.mark.parametrize("claim_type", ["case_application", "ruling_fact"])
@pytest.mark.parametrize("kind", ["faq", "card_supplement", "rule_passage"])
def test_decisive_ruling_claim_accepts_direct_mechanism_evidence(claim_type, kind):
    evidence_id = f"evidence:{kind}"
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="Mechanism evidence directly entails this ruling.",
                claim_type=claim_type,
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id=evidence_id,
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
        )
    )

    aggregation = aggregate_claim_coverage(
        report,
        evidence_by_id={
            evidence_id: EvidenceItem(
                evidence_id=evidence_id,
                kind=kind,
                source="test",
                text="Mechanism evidence.",
                relationship="rule_document",
            )
        },
    )

    assert aggregation.sufficient is True
    assert aggregation.unsupported_claims == ()


def test_evidence_context_retains_card_effects_and_claim_cited_evidence_after_many_items():
    agent = AdjudicationAgent([], settings=_settings())
    card_effect = EvidenceItem(
        evidence_id="card:1:effect",
        kind="card_effect",
        source="cards.cdb",
        text="Full Alpha Dragon effect text that must not be truncated away.",
        relationship="resolved_card",
        card_ids=(1,),
    )
    cited_faq = EvidenceItem(
        evidence_id="faq:cited",
        kind="faq",
        source="ygocdb",
        text="Cited FAQ text.",
        relationship="direct_card",
        card_ids=(1,),
    )
    late_rules = tuple(
        EvidenceItem(
            evidence_id=f"rule:{index}",
            kind="rule_passage",
            source="ocgrules",
            text=f"Late rules passage {index}.",
            relationship="rule_document",
        )
        for index in range(24)
    )
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                claim="FAQ proves the ruling fact.",
                claim_type="ruling_fact",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="faq:cited",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
        )
    )
    state = agent._replace(
        create_initial_agent_state("Can Alpha apply?"),
        evidence=(card_effect, cited_faq, *late_rules),
        claim_coverage_report=report,
    )

    context = build_adjudication_evidence_context(state).to_dict()
    prompt = build_agent_planner_prompt(state)

    assert "card:1:effect" in prompt
    assert "Full Alpha Dragon effect text" in prompt
    assert "faq:cited" in prompt
    assert context["sections"]["resolved_card_effects_full"][0]["evidence_id"] == "card:1:effect"
    assert context["sections"]["claim_cited_evidence"][0]["cited_by_claim_ids"] == ["c1"]


def test_claim_coverage_parser_accepts_issue_polarity_and_prompt_is_chinese():
    report = parse_claim_coverage_report(
        json.dumps(
            {
                "schema_version": 2,
                "issues": [
                    {
                        "issue_id": "issue.main.answer.alpha",
                        "question": "Can Alpha apply?",
                        "status": "supported",
                        "polarity": "negative",
                        "supported_answer": "no",
                        "claim_ids": ["c1"],
                        "supporting_evidence_ids": ["rule:1"],
                        "unresolved_claim_ids": [],
                    }
                ],
                "required_claims": [
                    {
                        "claim_id": "c1",
                        "stable_id": "claim.main.case_application.alpha",
                        "claim": "Alpha cannot apply.",
                        "claim_type": "case_application",
                        "status": "supported",
                        "issue_id": "issue.main.answer.alpha",
                        "semantic_role": "alpha_application",
                        "polarity": "negative",
                        "supported_answer": "no",
                        "evidence_checks": [
                            {
                                "evidence_id": "rule:1",
                                "relevance": "high",
                                "entailment": "supports",
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        allowed_evidence_ids={"rule:1"},
    )

    state = AdjudicationAgent([], settings=_settings())._replace(
        create_initial_agent_state("Can Alpha apply?"),
        evidence=(
            EvidenceItem(
                evidence_id="rule:1",
                kind="rule_passage",
                source="ocgrules",
                text="Rule text.",
                relationship="rule_document",
            ),
        ),
        claim_coverage_report=report,
    )
    prompt = build_claim_coverage_prompt(state, CoverageReport(sufficient=False))

    assert report.issues[0].polarity == "negative"
    assert report.required_claims[0].stable_id == "claim.main.case_application.alpha"
    assert "维护稳定 issue/claim graph" in prompt
    assert "支持否定" in prompt


def test_aggregation_allows_supported_negative_issue_and_hypothetical_dependency_inference():
    evidence_by_id = {
        "card:1:effect": EvidenceItem(
            evidence_id="card:1:effect",
            kind="card_effect",
            source="cards.cdb",
            text="Card text.",
            relationship="resolved_card",
        ),
        "rule:1": EvidenceItem(
            evidence_id="rule:1",
            kind="rule_passage",
            source="ocgrules",
            text="Rule text.",
            relationship="rule_document",
        ),
    }
    negative_report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="c1",
                issue_id="issue.alpha",
                claim="Alpha can apply.",
                claim_type="case_application",
                status="contradicted",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="rule:1",
                        relevance="high",
                        entailment="contradicts",
                    ),
                ),
            ),
        ),
        issues=(
            ClaimCoverageIssue(
                issue_id="issue.alpha",
                question="Can Alpha apply?",
                status="supported",
                polarity="negative",
                supported_answer="no",
                claim_ids=("c1",),
                supporting_evidence_ids=("rule:1",),
            ),
        ),
    )

    negative_aggregation = aggregate_claim_coverage(
        negative_report,
        evidence_by_id=evidence_by_id,
    )

    assert negative_aggregation.sufficient is True
    assert negative_aggregation.negative_supported_issues == ("issue.alpha",)

    dependency_report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="card_text",
                claim="Alpha has the relevant text.",
                claim_type="card_text",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="card:1:effect",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
            ClaimCoverageClaim(
                claim_id="rule_fact",
                claim="The rule principle applies.",
                claim_type="rule_principle",
                status="supported",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="rule:1",
                        relevance="high",
                        entailment="supports",
                    ),
                ),
            ),
            ClaimCoverageClaim(
                claim_id="hypothetical",
                claim="The hypothetical follows from the supported parent claims.",
                claim_type="hypothetical_branch",
                status="supported",
                branch="hypothetical:alpha",
                depends_on=("card_text", "rule_fact"),
                inference_reason="Card text plus rule principle entail this branch.",
            ),
        )
    )

    dependency_aggregation = aggregate_claim_coverage(
        dependency_report,
        evidence_by_id=evidence_by_id,
    )

    assert dependency_aggregation.sufficient is True
    assert "hypothetical" not in dependency_aggregation.unsupported_claims


def test_contradicted_affirmative_claim_supports_negative_issue():
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="C1",
                issue_id="I1",
                claim="Alpha can apply its replacement effect.",
                claim_type="case_application",
                status="contradicted",
                polarity="affirmative",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="faq:deny",
                        relevance="high",
                        entailment="contradicts",
                        reason="FAQ says this replacement effect cannot apply.",
                    ),
                ),
            ),
        ),
        issues=(
            ClaimCoverageIssue(
                issue_id="I1",
                question="Can Alpha apply?",
                status="contradicted",
                polarity="negative",
                supported_answer="no",
                claim_ids=("C1",),
                supporting_evidence_ids=("faq:deny",),
                unresolved_claim_ids=("C1",),
            ),
        ),
    )

    aggregation = aggregate_claim_coverage(
        report,
        evidence_by_id={
            "faq:deny": EvidenceItem(
                evidence_id="faq:deny",
                kind="faq",
                source="ygocdb",
                text="It cannot apply.",
                relationship="related_card",
            )
        },
    )

    assert aggregation.sufficient is True
    assert aggregation.negative_supported_issues == ("I1",)
    assert aggregation.contradicted_claims == ()
    assert aggregation.blocking_claims == ()


def test_submit_ruling_accepts_contradicted_affirmative_claim_via_negative_issue():
    agent = AdjudicationAgent([], settings=_settings())
    evidence = (
        EvidenceItem(
            evidence_id="faq:deny",
            kind="faq",
            source="ygocdb",
            text="It cannot apply.",
            relationship="related_card",
        ),
    )
    report = ClaimCoverageReport(
        required_claims=(
            ClaimCoverageClaim(
                claim_id="C1",
                issue_id="I1",
                claim="Alpha can apply its replacement effect.",
                claim_type="case_application",
                status="contradicted",
                polarity="affirmative",
                evidence_checks=(
                    ClaimEvidenceCheck(
                        evidence_id="faq:deny",
                        relevance="high",
                        entailment="contradicts",
                    ),
                ),
            ),
        ),
        issues=(
            ClaimCoverageIssue(
                issue_id="I1",
                question="Can Alpha apply?",
                status="contradicted",
                polarity="negative",
                supported_answer="no",
                claim_ids=("C1",),
                supporting_evidence_ids=("faq:deny",),
                unresolved_claim_ids=("C1",),
            ),
        ),
    )
    state = agent._replace(
        create_initial_agent_state("Can Alpha apply?"),
        scenario=ParsedScenario(
            original_text="Can Alpha apply?",
            questions=("Can Alpha apply?",),
        ),
        evidence=evidence,
        claim_coverage_report=report,
        claim_coverage_aggregation=aggregate_claim_coverage(
            report,
            evidence_by_id={item.evidence_id: item for item in evidence},
        ),
    )
    answer = {
        "conclusion": "Alpha cannot apply.",
        "answers": [
            {
                "question": "Can Alpha apply?",
                "case_conclusion": "Alpha cannot apply.",
                "hypothetical_conclusion": "",
                "reasoning": [
                    {
                        "judgment": "The affirmative application claim is contradicted by FAQ evidence, supporting the negative issue.",
                        "issue_id": "I1",
                        "evidence_ids": ["faq:deny"],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": "faq:deny",
                "role": "related_card",
                "use": "FAQ denies the replacement effect.",
            }
        ],
        "uncertainties": [],
    }

    _, observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="submit_ruling",
            arguments={"answer": answer},
        ),
    )

    assert observation.ok is True


def test_agent_planner_response_parser_validates_tool_schema():
    action = parse_agent_planner_response(
        json.dumps(
            {
                "tool": "search_rules",
                "arguments": {"query": "怪兽效果发动限制 非同调"},
                "rationale": "Need rules evidence.",
                "notes": ["query generated by planner"],
                "hypotheses": ["activation restriction"],
            },
            ensure_ascii=False,
        )
    )

    assert action.tool == "search_rules"
    assert action.arguments["query"] == "怪兽效果发动限制 非同调"
    assert action.notes == ("query generated by planner",)

    invalid_outputs = [
        "not json",
        json.dumps({"tool": "unknown", "arguments": {}}),
        json.dumps({"tool": "search_rules", "arguments": {}}),
        json.dumps({"tool": "search_card", "arguments": {"query": "Alpha", "extra": True}}),
    ]
    for output in invalid_outputs:
        try:
            parse_agent_planner_response(output)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid planner output should be rejected")


def test_normalized_middle_dot_card_name_resolves_to_space_name():
    cards = [
        Card(68625623, UNCHAINED_SOUL_OF_RAGE, "Destroy one card."),
        Card(94014327, UNCHAINED_ABOMINATION, "Destroy one monster."),
    ]

    candidates = search_card_candidates("\u7834\u68b0\u51a5\u5b98\u00b7\u7bc1", cards)
    resolution = select_card_resolution(
        "\u7834\u68b0\u51a5\u5b98\u00b7\u7bc1",
        candidates,
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )

    assert normalize_card_name("\u7834\u68b0\u51a5\u5b98\u00b7\u7bc1") == normalize_card_name(
        UNCHAINED_SOUL_OF_RAGE
    )
    assert resolution.status == "resolved"
    assert resolution.selected_card_id == 68625623


def test_ambiguous_card_resolution_is_not_silently_selected():
    candidates = search_card_candidates(
        "Alpha",
        [
            Card(1, "Alpha Dragon", "Destroy one card."),
            Card(2, "Alpha Serpent", "Destroy one monster."),
        ],
    )

    resolution = select_card_resolution(
        "Alpha",
        candidates,
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )

    assert resolution.status == "ambiguous"
    assert resolution.selected_card_id is None
    assert len(resolution.candidates) >= 2


def test_agent_tools_collect_relationship_labels_and_trace():
    cards = [
        Card(68625623, UNCHAINED_SOUL_OF_RAGE, "Destroy one card."),
        Card(94014327, UNCHAINED_ABOMINATION, "Destroy one monster."),
        Card(13203964, CYBER_DRAGON, "Lose 1000 ATK instead of being destroyed."),
    ]
    ruling_index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 94014327,
                "faqs": [
                    {
                        "fid": "24345",
                        "title": "Replacement consequence",
                        "question": "If the target was not destroyed?",
                        "answer": "If the target was not destroyed, the later special summon is not applied.",
                        "refer": {"68625623": True},
                    }
                ],
                "supplement": {"text": "Supplement text.", "refer": {}, "date": "2026"},
            }
        ]
    )
    agent = AdjudicationAgent(
        cards,
        settings=_settings(),
        ruling_index=ruling_index,
    )

    response = agent.run(
        "\u5982\u679c\u7834\u68b0\u51a5\u5b98\u00b7\u7bc1\u8981\u7834\u574f"
        "\u7834\u68b0\u7130\u9b54\u5929\u00b7\u960e\u6469\uff0c"
        "\u53ef\u4ee5\u7528"
        "\u5b8c\u7f8e\u7535\u5b50\u591a\u5143\u9a71\u52a8\u86c7\u00b7\u795e\u9f99"
        "\u4ee3\u66ff\u5417\uff1f\u5982\u679c\u6ca1\u6709\u7834\u574f\uff0c"
        "\u8fd8\u80fd\u7279\u6b8a\u53ec\u5524\u5417\uff1f"
    )

    relationships = {item.relationship for item in response.evidence}
    assert any("related_card" in relationship for relationship in relationships)
    assert response.agent_coverage is not None
    assert response.agent_trace
    assert any(entry["node"] == "check_coverage" for entry in response.agent_trace)
    assert any(item.evidence_id == "ygocdb:faq:24345" for item in response.evidence)


def test_llm_card_name_extraction_parsing_rejects_ids_and_malformed_output():
    mentions = parse_card_name_extraction_response(
        json.dumps(
            {
                "mentions": [
                    {"text": UNCHAINED_SOUL_OF_RAGE, "context": "发动效果"},
                    {"text": "  ", "context": "ignored"},
                ]
            },
            ensure_ascii=False,
        )
    )

    assert mentions == (
        ExtractedCardMention(
            text=UNCHAINED_SOUL_OF_RAGE,
            context="发动效果",
            source="llm",
        ),
    )

    for output in (
        "not json",
        json.dumps({"mentions": [{"text": UNCHAINED_SOUL_OF_RAGE, "card_id": 68625623}]}),
        json.dumps({"mentions": [], "card_id": 68625623}),
    ):
        try:
            parse_card_name_extraction_response(output)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid extraction output should be rejected")


def test_constrained_llm_card_selection_rejects_unknown_card_ids():
    selected, reason = parse_card_selection_response(
        json.dumps({"selected_card_id": 1, "reason": "name matches"}),
        allowed_ids={1, 2},
    )

    assert selected == 1
    assert reason == "name matches"

    try:
        parse_card_selection_response(
            json.dumps({"selected_card_id": 999, "reason": "invented"}),
            allowed_ids={1, 2},
        )
    except ValueError as exc:
        assert "unknown card id" in str(exc)
    else:
        raise AssertionError("unknown selected_card_id should be rejected")

    try:
        parse_card_selection_response(
            json.dumps({"selected_card_id": 1, "reason": ""}),
            allowed_ids={1, 2},
        )
    except ValueError as exc:
        assert "explain" in str(exc)
    else:
        raise AssertionError("LLM selected_card_id without reason should be rejected")


def test_sparse_top1_without_name_compatibility_is_not_resolved():
    class FakeSparseRetriever:
        def search(self, *args, **kwargs):
            class Result:
                card_id = 1
                metadata = {
                    "name": "Disaster Lion",
                    "desc": "Thunder break effect text.",
                }

            return [Result()]

    cards = [Card(1, "Disaster Lion", "Thunder break effect text.")]

    candidates = search_card_candidates("雷破", cards, retriever=FakeSparseRetriever())
    resolution = select_card_resolution(
        "雷破",
        candidates,
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )

    assert resolution.status in {"ambiguous", "unresolved"}
    assert resolution.selected_card_id is None
    assert resolution.candidates[0].validation_status in {"needs_validation", "rejected"}


def test_exact_alias_substring_and_short_mention_validation():
    cards = [
        Card(1, "Alpha Dragon", "Destroy one card."),
        Card(2, "Beta Serpent", "Cannot be destroyed."),
    ]

    exact = select_card_resolution(
        "Alpha Dragon",
        search_card_candidates("Alpha Dragon", cards),
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )
    assert exact.status == "resolved"
    assert exact.selection_reason == "exact normalized name match"

    alias = select_card_resolution(
        "阿龙",
        search_card_candidates(
            "阿龙",
            cards,
            verified_aliases={normalize_card_name("阿龙"): (1, "manual")},
        ),
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )
    assert alias.status == "resolved"
    assert "manual" in alias.selection_reason

    substring = select_card_resolution(
        "Alpha Dragon applies",
        search_card_candidates("Alpha Dragon applies", cards),
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )
    assert substring.status == "resolved"
    assert "substring coverage rule" in substring.selection_reason

    short = select_card_resolution(
        "龙",
        search_card_candidates("龙", [Card(3, "神龙", "Effect.")]),
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )
    assert short.status != "resolved"
    assert short.selected_card_id is None


def test_unique_exact_match_wins_over_substring_candidate_inside_ambiguity_margin():
    cards = [
        Card(5821478, "拓扑逻辑轰炸龙", "After damage calculation, inflict damage."),
        Card(20586572, "轰炸龙", "Destroy cards."),
    ]

    resolution = select_card_resolution(
        "拓扑逻辑轰炸龙",
        search_card_candidates("拓扑逻辑轰炸龙", cards),
        confidence_threshold=0.75,
        ambiguity_margin=0.08,
    )

    assert resolution.status == "resolved"
    assert resolution.selected_card_id == 5821478
    assert resolution.selection_reason == "exact normalized name match"


def test_candidate_mentions_filter_mechanism_terms_and_covered_short_names():
    scenario = ParsedScenario(
        original_text="",
        card_mentions=("拓扑逻辑轰炸龙", "轰炸龙", "无效"),
    )

    queries = candidate_mention_queries(
        "对方在「拓扑逻辑轰炸龙」处理后试图无效。",
        scenario,
    )

    assert "拓扑逻辑轰炸龙" in queries
    assert "轰炸龙" not in queries
    assert "无效" not in queries


def test_default_verified_aliases_resolve_common_case_variants():
    cards = [
        Card(71808988, "青眼卡通究极龙", "Cannot attack directly unless conditions apply."),
        Card(43387895, "霸王眷龙 凶饿毒", "Copy name and effect."),
        Card(23657016, "幻禄之天杯龙", "Tribute this card; Special Summon."),
        Card(31603289, "奶油蛋糕杯猫★味美喵奖品", "Return this card to the Extra Deck."),
    ]
    agent = AdjudicationAgent(cards, settings=_settings())

    for mention, card_id in (
        ("卡通青眼究极龙", 71808988),
        ("霸王眷龙 凶恶毒", 43387895),
        ("幻禄的天杯龙", 23657016),
        ("奶油蛋糕喵★黯蜜味", 31603289),
    ):
        resolution = agent._resolve_card_mention(mention, warnings=[])
        assert resolution.status == "resolved"
        assert resolution.selected_card_id == card_id
        assert resolution.selection_source == "rules"
        assert "verified alias" in resolution.selection_reason


def test_targeted_retry_rejects_unrelated_broad_sparse_replacement():
    cards = [
        Card(79791878, "光之黄金柜", "Lists related monsters."),
        Card(43387895, "霸王眷龙 凶饿毒", "Copy name and effect."),
    ]
    agent = AdjudicationAgent(cards, settings=_settings())
    retry_resolution = CardResolution(
        mention="自己场上有霸王眷龙 凶恶毒和光之黄金柜",
        status="resolved",
        selected_card_id=79791878,
        candidates=search_card_candidates("光之黄金柜", cards),
    )

    assert not agent._targeted_retry_explains_mention(
        "霸王眷龙 凶恶毒",
        "自己场上有霸王眷龙 凶恶毒和光之黄金柜",
        retry_resolution,
    )


def test_invalid_agent_planner_output_falls_back_with_warning():
    cards = [Card(68625623, UNCHAINED_SOUL_OF_RAGE, "Destroy one card.")]
    agent = AdjudicationAgent(cards, settings=_settings(), llm=FakeLlm(["not json"]))

    response = agent.run(f"{UNCHAINED_SOUL_OF_RAGE} 发动效果。")

    assert not response.resolved_cards
    assert any("Agent planner output" in warning for warning in response.warnings)
    assert any(entry["node"] == "initialize_agent_loop" for entry in response.agent_trace)
    assert any(entry["node"] == "agent_loop_fallback" for entry in response.agent_trace)


def test_targeted_retry_can_resolve_initially_ambiguous_mention():
    cards = [
        Card(1, "Alpha Dragon", "Destroy one card."),
        Card(2, "Alpha Serpent", "Destroy one monster."),
    ]
    agent = AdjudicationAgent(cards, settings=_settings())
    state = create_initial_agent_state("Alpha Dragon applies.")
    state = agent._replace(
        state,
        scenario=ParsedScenario(
            original_text=state.scenario_text,
            card_mentions=("Alpha Dragon",),
            questions=("Alpha?",),
        ),
        mention_queries=("Alpha",),
        extracted_card_mentions=(ExtractedCardMention(text="Alpha"),),
        card_resolutions=(
            CardResolution(
                mention="Alpha",
                status="ambiguous",
                candidates=search_card_candidates("Alpha", cards),
            ),
        ),
    )

    result = agent._node_resolve_cards({"agent_state": state})["agent_state"]

    assert result.card_resolution_loop.retry_count == 1
    assert result.card_resolutions[0].status == "resolved"
    assert result.card_resolutions[0].selected_card_id == 1


def test_persistent_retrieval_uses_final_resolved_ids_after_retry():
    class RecordingRulingRetriever:
        def __init__(self):
            self.card_ids = []

        def search(self, *args, **kwargs):
            self.card_ids.append(tuple(kwargs["card_ids"]))
            return RulingRetrievalResult(
                candidates=(
                    RulingCandidate(
                        evidence_id="ygocdb:faq:retry",
                        score=1.0,
                        source="hybrid",
                        relationship="analogous",
                        document=RulingDocument(
                            evidence_id="ygocdb:faq:retry",
                            kind="faq",
                            source="ygocdb",
                            title="Retry evidence",
                            text="Resolved card evidence.",
                            card_ids=tuple(kwargs["card_ids"]),
                        ),
                    ),
                )
            )

    retriever = RecordingRulingRetriever()
    cards = [
        Card(1, "Alpha Dragon", "Destroy one card."),
        Card(2, "Alpha Serpent", "Destroy one monster."),
    ]
    agent = AdjudicationAgent(
        cards,
        settings=_settings(),
        ruling_retrievers=(retriever,),
    )

    response = agent.run("Alpha Dragon applies. If not destroyed, can it special summon?")

    assert (1,) in retriever.card_ids
    assert any(item.evidence_id == "ygocdb:faq:retry" for item in response.evidence)


def test_persistent_retrieval_preserves_staged_relationships_and_trace():
    documents = [
        RulingDocument(
            evidence_id="ygocdb:faq:direct",
            kind="faq",
            source="ygocdb",
            title="Direct",
            text="Alpha Dragon direct FAQ.",
            card_ids=(1,),
            primary_card_ids=(1,),
        ),
        RulingDocument(
            evidence_id="ygocdb:faq:related",
            kind="faq",
            source="ygocdb",
            title="Related",
            text="Alpha Dragon related FAQ.",
            card_ids=(2,),
            refer_card_ids=(1,),
        ),
        RulingDocument(
            evidence_id="ygocdb:supplement:1",
            kind="card_supplement",
            source="ygocdb",
            title="Supplement",
            text="Alpha Dragon supplement.",
            card_ids=(1,),
        ),
        RulingDocument(
            evidence_id="rules:alpha",
            kind="rule_passage",
            source="ocg_rules",
            title="Rules",
            text="Alpha Dragon rules passage.",
        ),
        RulingDocument(
            evidence_id="ygocdb:faq:analogous",
            kind="faq",
            source="ygocdb",
            title="Analogous",
            text="Alpha Dragon analogous FAQ.",
        ),
    ]
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
        ruling_retrievers=(PersistentRulingRetriever(documents),),
    )

    response = agent.run("Alpha Dragon applies. Can the later processing happen?")

    relationships = {item.evidence_id: item.relationship for item in response.evidence}
    assert relationships["ygocdb:faq:direct"].split(",")[0] == "direct_card"
    assert "related_card" in relationships["ygocdb:faq:related"]
    assert relationships["ygocdb:supplement:1"] == "card_supplement"
    assert relationships["rules:alpha"] == "rule_document"
    assert "analogous" in relationships["ygocdb:faq:analogous"]
    retrieve_trace = [
        entry for entry in response.agent_trace if entry["node"] == "retrieve_evidence"
    ][0]
    stages = retrieve_trace["metadata"]["retrieval_stages"]
    assert {stage["stage"] for stage in stages} >= {
        "direct_faq",
        "related_faq",
        "supplement",
        "analogous_faq",
        "rules",
    }
    assert any(
        item.metadata.get("retrieval_stages")
        for item in response.evidence
        if item.evidence_id == "ygocdb:faq:direct"
    )


def test_agent_tools_search_card_is_candidate_only_and_submit_validates():
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
    )
    state = create_initial_agent_state("Alpha Dragon applies.")
    state = agent._replace(
        state,
        scenario=ParsedScenario(
            original_text=state.scenario_text,
            questions=("Can Alpha apply?",),
        ),
    )

    state, observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="search_card",
            arguments={"query": "Alpha Dragon"},
            rationale="candidate search only",
        ),
    )

    assert observation.ok is True
    assert observation.data["candidate_only"] is True
    assert not state.resolved_cards

    state, observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="submit_ruling",
            arguments={
                "answer": {
                    "conclusion": "Unknown.",
                    "answers": [
                        {
                            "question": "Can Alpha apply?",
                            "case_conclusion": "Unknown.",
                            "hypothetical_conclusion": "",
                            "reasoning": [
                                {"judgment": "No evidence.", "evidence_ids": ["missing:e1"]}
                            ],
                        }
                    ],
                    "evidence_use": [
                        {"evidence_id": "missing:e1", "role": "rule_document", "use": "bad"}
                    ],
                    "uncertainties": [],
                }
            },
            rationale="submit bad answer",
        ),
    )

    assert observation.ok is False
    assert "unknown evidence" in observation.validation_error


def test_resolve_cards_tool_adds_card_effect_evidence():
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.", atk=2500, defense=2000)],
        settings=_settings(),
    )
    state = create_initial_agent_state("Can Alpha Dragon apply?")

    state, observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="resolve_cards",
            arguments={"mentions": ["Alpha Dragon"]},
        ),
    )

    assert observation.ok is True
    assert observation.evidence_ids == ("card:1:effect",)
    card_effect = next(item for item in state.evidence if item.evidence_id == "card:1:effect")
    assert "ATK 2500" in card_effect.text
    assert card_effect.metadata["atk"] == 2500


def test_resolve_cards_tool_adds_resolved_card_supplement_evidence():
    ruling_index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 1,
                "faqs": [
                    {
                        "fid": "faq1",
                        "title": "Alpha ruling",
                        "question": "Can Alpha apply?",
                        "answer": "Alpha cannot apply.",
                        "refer": {},
                    }
                ],
                "supplement": {
                    "text": "Alpha supplement text.",
                    "refer": {},
                    "date": "2026",
                },
            }
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
        ruling_index=ruling_index,
    )
    state = create_initial_agent_state("Can Alpha Dragon apply?")

    state, observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="resolve_cards",
            arguments={"mentions": ["Alpha Dragon"]},
        ),
    )

    assert observation.ok is True
    assert "ygocdb:faq:faq1" in observation.evidence_ids
    assert "ygocdb:supplement:1" in observation.evidence_ids
    assert any(item.evidence_id == "ygocdb:faq:faq1" for item in state.evidence)
    assert any(item.evidence_id == "ygocdb:supplement:1" for item in state.evidence)


def test_submit_ruling_validates_claim_ids_statuses_and_relationships():
    agent = AdjudicationAgent([], settings=_settings())
    evidence = (
        EvidenceItem(
            evidence_id="rule:1",
            kind="rule_passage",
            source="ocgrules",
            text="Rule text.",
            relationship="rule_document",
        ),
    )

    for status, expected in (
        ("unsupported", "unsupported claim"),
        ("unknown", "unknown claim"),
        ("partially_supported", "partially supported claim"),
        ("contradicted", "contradicted claim"),
    ):
        report = parse_claim_coverage_report(
            json.dumps(
                _claim_report_payload(
                    status=status,
                    entailment=(
                        "partially_supports"
                        if status == "partially_supported"
                        else "contradicts"
                        if status == "contradicted"
                        else "does_not_entail"
                    ),
                )
            ),
            allowed_evidence_ids={"rule:1"},
        )
        state = agent._replace(
            create_initial_agent_state("Can Alpha apply?"),
            scenario=ParsedScenario(
                original_text="Can Alpha apply?",
                questions=("Can Alpha apply?",),
            ),
            evidence=evidence,
            claim_coverage_report=report,
            claim_coverage_aggregation=aggregate_claim_coverage(report),
        )
        _, observation = agent._execute_agent_tool(
            state,
            PlannerAction(
                tool="submit_ruling",
                arguments={"answer": _answer_with_claim("c1")},
            ),
        )
        assert observation.ok is False
        assert expected in observation.validation_error

    supported = parse_claim_coverage_report(
        json.dumps(_claim_report_payload()),
        allowed_evidence_ids={"rule:1"},
    )
    supported_state = agent._replace(
        create_initial_agent_state("Can Alpha apply?"),
        scenario=ParsedScenario(
            original_text="Can Alpha apply?",
            questions=("Can Alpha apply?",),
        ),
        evidence=evidence,
        claim_coverage_report=supported,
        claim_coverage_aggregation=aggregate_claim_coverage(
            supported,
            evidence_by_id={item.evidence_id: item for item in evidence},
        ),
    )

    _, unknown_claim = agent._execute_agent_tool(
        supported_state,
        PlannerAction(
            tool="submit_ruling",
            arguments={"answer": _answer_with_claim("missing")},
        ),
    )
    assert unknown_claim.ok is False
    assert "unknown claim ids" in unknown_claim.validation_error

    _, relationship_mismatch = agent._execute_agent_tool(
        supported_state,
        PlannerAction(
            tool="submit_ruling",
            arguments={"answer": _answer_with_claim("c1", role="direct_card")},
        ),
    )
    assert relationship_mismatch.ok is False
    assert "misstated evidence relationship" in relationship_mismatch.validation_error

    unsupported = parse_claim_coverage_report(
        json.dumps(
            _claim_report_payload(
                status="unknown",
                entailment="unknown",
            )
        ),
        allowed_evidence_ids={"rule:1"},
    )
    uncertain_state = agent._replace(
        supported_state,
        claim_coverage_report=unsupported,
        claim_coverage_aggregation=aggregate_claim_coverage(unsupported),
    )
    _, uncertain = agent._execute_agent_tool(
        uncertain_state,
        PlannerAction(
            tool="submit_ruling",
            arguments={
                "answer": _answer_with_claim(
                    "c1",
                    confidence="low",
                    uncertainties=["c1 cannot be determined from available evidence."],
                )
            },
        ),
    )
    assert uncertain.ok is True


def test_submit_ruling_returns_structured_field_errors_for_blank_references():
    agent = AdjudicationAgent([], settings=_settings())
    evidence = (
        EvidenceItem(
            evidence_id="rule:1",
            kind="rule_passage",
            source="ocgrules",
            text="Rule text.",
            relationship="rule_document",
        ),
    )
    report = parse_claim_coverage_report(
        json.dumps(_claim_report_payload()),
        allowed_evidence_ids={"rule:1"},
    )
    state = agent._replace(
        create_initial_agent_state("Can Alpha apply?"),
        scenario=ParsedScenario(
            original_text="Can Alpha apply?",
            questions=("Can Alpha apply?",),
        ),
        evidence=evidence,
        claim_coverage_report=report,
        claim_coverage_aggregation=aggregate_claim_coverage(
            report,
            evidence_by_id={item.evidence_id: item for item in evidence},
        ),
    )
    answer = _answer_with_claim(" ")
    answer["answers"][0]["reasoning"][0]["issue_id"] = ""

    _, observation = agent._execute_agent_tool(
        state,
        PlannerAction(tool="submit_ruling", arguments={"answer": answer}),
    )

    assert observation.ok is False
    assert "unknown claim ids" not in observation.validation_error
    assert observation.data["consumes_submit_budget"] is False
    assert observation.data["validation_errors"] == [
        {
            "code": "blank_reference_id",
            "path": "$.answer.answers[0].reasoning[0].claim_id",
            "message": "claim_id must be a non-empty string.",
            "retryable": True,
        },
        {
            "code": "blank_reference_id",
            "path": "$.answer.answers[0].reasoning[0].issue_id",
            "message": "issue_id must be a non-empty string.",
            "retryable": True,
        },
    ]


def test_submit_ruling_structures_schema_error_at_missing_field():
    agent = AdjudicationAgent([], settings=_settings())
    _, observation = agent._execute_agent_tool(
        create_initial_agent_state("Can Alpha apply?"),
        PlannerAction(
            tool="submit_ruling",
            arguments={
                "answer": {
                    "conclusion": "Incomplete answer.",
                    "evidence_use": [],
                    "uncertainties": [],
                }
            },
        ),
    )

    assert observation.ok is False
    assert observation.validation_error
    assert observation.data["validation_errors"] == [
        {
            "code": "required",
            "path": "$.answer.answers",
            "message": "LLM adjudication answer field answers is required.",
            "retryable": True,
        }
    ]
    assert observation.data["consumes_submit_budget"] is False


@pytest.mark.parametrize(
    ("field", "value", "expected_path", "expected_code"),
    [
        ("source_status", [], "$.answer.source_status", "invalid_type"),
        ("source_status", {"faq": "found"}, "$.answer.source_status", "invalid_value"),
        (
            "source_status",
            {"faq": "invalid", "rules": "found", "failure_reason": ""},
            "$.answer.source_status.faq",
            "invalid_value",
        ),
        (
            "source_status",
            {"faq": "found", "rules": "found", "failure_reason": None},
            "$.answer.source_status.failure_reason",
            "invalid_type",
        ),
        ("answers", "invalid", "$.answer.answers", "invalid_value"),
        ("answers", [], "$.answer.answers", "invalid_value"),
    ],
)
def test_submit_ruling_structures_present_field_shape_errors(
    field,
    value,
    expected_path,
    expected_code,
):
    agent = AdjudicationAgent([], settings=_settings())
    answer = _answer_with_claim("c1")
    answer[field] = value

    _, observation = agent._execute_agent_tool(
        create_initial_agent_state("Can Alpha apply?"),
        PlannerAction(tool="submit_ruling", arguments={"answer": answer}),
    )

    assert observation.ok is False
    assert observation.data["validation_errors"][0]["path"] == expected_path
    assert observation.data["validation_errors"][0]["code"] == expected_code
    assert observation.data["validation_errors"][0]["retryable"] is True
    assert observation.data["consumes_submit_budget"] is False


@pytest.mark.parametrize(
    ("field", "value", "relative_path", "reason", "expected_code"),
    [
        (
            "conclusion",
            [],
            "conclusion",
            "must be a non-empty string.",
            "invalid_value",
        ),
        (
            "answers",
            ["bad"],
            "answers[0]",
            "must be an object.",
            "invalid_type",
        ),
        (
            "evidence_use",
            "bad",
            "evidence_use",
            "must be an array.",
            "invalid_type",
        ),
        (
            "uncertainties",
            "bad",
            "uncertainties",
            "must be an array.",
            "invalid_type",
        ),
    ],
)
def test_submit_ruling_structures_unified_validator_field_errors(
    monkeypatch,
    field,
    value,
    relative_path,
    reason,
    expected_code,
):
    validator_calls = []

    def reject_with_unified_field_error(answer, allowed_ids):
        validator_calls.append((json.loads(answer), allowed_ids))
        raise ValueError(
            f"LLM adjudication answer field {relative_path} {reason}"
        )

    monkeypatch.setattr(
        "rag_agent.adjudication_agent.validate_adjudication_answer_json",
        reject_with_unified_field_error,
    )
    answer = {
        "conclusion": "Use cited evidence.",
        "answers": [
            {
                "question": "Can Alpha apply?",
                "case_conclusion": "Use cited evidence.",
                "hypothetical_conclusion": "",
                "reasoning": [],
            }
        ],
        "evidence_use": [],
        "uncertainties": [],
    }
    answer[field] = value
    agent = AdjudicationAgent([], settings=_settings())

    _, observation = agent._execute_agent_tool(
        create_initial_agent_state("Can Alpha apply?"),
        PlannerAction(tool="submit_ruling", arguments={"answer": answer}),
    )

    assert len(validator_calls) == 1
    assert observation.ok is False
    assert observation.data["validation_errors"] == [
        {
            "code": expected_code,
            "path": f"$.answer.{relative_path}",
            "message": (
                f"LLM adjudication answer field {relative_path} {reason}"
            ),
            "retryable": True,
        }
    ]
    assert observation.data["consumes_submit_budget"] is False


def test_substantive_submit_rejection_consumes_submit_budget():
    agent = AdjudicationAgent(
        [],
        settings=_settings({"RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT": "1"}),
    )
    evidence = (
        EvidenceItem(
            evidence_id="rule:1",
            kind="rule_passage",
            source="ocgrules",
            text="Rule text.",
            relationship="rule_document",
        ),
    )
    report = parse_claim_coverage_report(
        json.dumps(
            _claim_report_payload(
                status="unsupported",
                entailment="does_not_entail",
            )
        ),
        allowed_evidence_ids={"rule:1"},
    )
    state = agent._replace(
        create_initial_agent_state("Can Alpha apply?"),
        scenario=ParsedScenario(
            original_text="Can Alpha apply?",
            questions=("Can Alpha apply?",),
        ),
        evidence=evidence,
        claim_coverage_report=report,
        claim_coverage_aggregation=aggregate_claim_coverage(
            report,
            evidence_by_id={item.evidence_id: item for item in evidence},
        ),
    )
    action = PlannerAction(
        tool="submit_ruling",
        arguments={"answer": _answer_with_claim("c1")},
    )

    rejected_state, observation = agent._execute_agent_tool(state, action)
    history_state = agent._record_tool_step(
        rejected_state,
        step=1,
        action=action,
        observation=observation,
        duplicate=False,
    )
    budget_observation = agent._budget_observation(history_state, action)

    assert observation.ok is False
    assert observation.data["validation_errors"][0]["code"] == "unsupported_claim"
    assert observation.data["validation_errors"][0]["retryable"] is False
    assert observation.data["consumes_submit_budget"] is True
    assert budget_observation is not None
    assert budget_observation.data == {"budget": "submit", "used": 1, "limit": 1}


def test_submit_ruling_requires_supported_hypothetical_branch_claim():
    agent = AdjudicationAgent([], settings=_settings())
    evidence = (
        EvidenceItem(
            evidence_id="rule:1",
            kind="rule_passage",
            source="ocgrules",
            text="Rule text.",
            relationship="rule_document",
        ),
    )
    main_only = parse_claim_coverage_report(
        json.dumps(_claim_report_payload()),
        allowed_evidence_ids={"rule:1"},
    )
    branch = parse_claim_coverage_report(
        json.dumps(
            _claim_report_payload(
                claim_type="hypothetical_branch",
                branch="hypothetical:if destroyed",
            )
        ),
        allowed_evidence_ids={"rule:1"},
    )
    base_state = agent._replace(
        create_initial_agent_state("If Alpha applies, can Beta summon?"),
        scenario=ParsedScenario(
            original_text="If Alpha applies, can Beta summon?",
            questions=("If Alpha applies, can Beta summon?",),
        ),
        evidence=evidence,
    )
    main_state = agent._replace(
        base_state,
        claim_coverage_report=main_only,
        claim_coverage_aggregation=aggregate_claim_coverage(
            main_only,
            evidence_by_id={item.evidence_id: item for item in evidence},
        ),
    )
    _, rejected = agent._execute_agent_tool(
        main_state,
        PlannerAction(
            tool="submit_ruling",
            arguments={
                "answer": _answer_with_claim(
                    "c1",
                    hypothetical_conclusion="If the assumption holds, use cited evidence.",
                )
            },
        ),
    )
    assert rejected.ok is False
    assert "Hypothetical conclusion" in rejected.validation_error

    branch_state = agent._replace(
        base_state,
        claim_coverage_report=branch,
        claim_coverage_aggregation=aggregate_claim_coverage(branch),
    )
    _, accepted = agent._execute_agent_tool(
        branch_state,
        PlannerAction(
            tool="submit_ruling",
            arguments={
                "answer": _answer_with_claim(
                    "c1",
                    hypothetical_conclusion="If the assumption holds, use cited evidence.",
                )
            },
        ),
    )
    assert accepted.ok is True


def test_agent_loop_generates_distinct_rules_queries_and_accepts_submission():
    answer = {
        "conclusion": "Use cited rules.",
        "answers": [
            {
                "question": "Can Alpha Dragon activate?",
                "case_conclusion": "Use cited rules.",
                "hypothetical_conclusion": "",
                "reasoning": [
                    {
                        "judgment": "Activation limits need rule evidence.",
                        "evidence_ids": ["rule:activation"],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": "rule:activation",
                "role": "rule_document",
                "use": "Activation restriction rule.",
            }
        ],
        "uncertainties": [],
    }
    llm = FakeLlm(
        [
            *_deterministic_preparation_actions(),
            json.dumps(
                {
                    "tool": "resolve_cards",
                    "arguments": {"mentions": ["Alpha Dragon"]},
                    "rationale": "Need card identity.",
                }
            ),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "monster effect activation restriction extra deck non-synchro"},
                    "rationale": "Need first mechanism query.",
                }
            ),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "restriction checks effect activator or effect content"},
                    "rationale": "Refine rule query.",
                }
            ),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": answer},
                    "rationale": "Evidence is sufficient.",
                }
            ),
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings({"RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "5"}),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:activation",
                    title="Activation restriction",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="activation",
                    text="monster effect activation restriction extra deck non-synchro",
                ),
                RuleSection(
                    rule_id="rule:restriction-target",
                    title="Restriction target",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="restriction",
                    text="restriction checks effect activator or effect content",
                ),
            ]
        ),
        llm=llm,
    )

    response = agent.run("Can Alpha Dragon activate?")

    answer_payload = json.loads(response.answer)
    assert answer_payload["conclusion"] == answer["conclusion"]
    assert answer_payload["source_status"]["faq"] == "not_found"
    assert answer_payload["source_status"]["rules"] == "found"
    assert any(card.matches[0].card_id == 1 for card in response.resolved_cards)
    assert {"rule:activation", "rule:restriction-target"}.issubset(
        {item.evidence_id for item in response.evidence}
    )
    search_rule_queries = [
        entry["metadata"]["action"]["arguments"]["query"]
        for entry in response.agent_trace
        if entry["node"] == "agent_plan"
        and entry["metadata"]["action"]["tool"] == "search_rules"
    ]
    assert search_rule_queries == [
        "monster effect activation restriction extra deck non-synchro",
        "restriction checks effect activator or effect content",
    ]


def test_agent_planner_prompt_includes_tools_history_and_evidence_pool():
    state = create_initial_agent_state("Can Alpha apply?")
    state = AdjudicationAgent([], settings=_settings())._replace(
        state,
        evidence=(
            EvidenceItem(
                evidence_id="rule:1",
                kind="rule_passage",
                source="ocgrules",
                text="Rule text.",
                relationship="rule_document",
                metadata={"retrieval_stage": "rules"},
            ),
        ),
    )

    prompt = build_agent_planner_prompt(state)

    assert "allowed_tools" in prompt
    assert "search_rules" in prompt
    assert "evidence_pool" in prompt
    assert "rule:1" in prompt


def test_agent_planner_prompt_includes_submit_answer_schema_and_sufficient_policy():
    state = create_initial_agent_state("Can Alpha apply?")
    state = AdjudicationAgent([], settings=_settings())._replace(
        state,
        scenario=ParsedScenario(
            original_text=state.scenario_text,
            questions=("Can Alpha apply?",),
        ),
        evidence=(
            EvidenceItem(
                evidence_id="rule:1",
                kind="rule_passage",
                source="ocgrules",
                text="Rule text.",
                relationship="rule_document",
            ),
        ),
        coverage=CoverageReport(sufficient=True),
    )

    prompt = build_agent_planner_prompt(state)

    assert "submit_ruling_answer_schema" in prompt
    assert '"conclusion"' in prompt
    assert '"answers"' in prompt
    assert '"evidence_use"' in prompt
    assert '"uncertainties"' in prompt
    assert "下一步必须调用 submit_ruling" in prompt


def test_claim_coverage_prompt_requires_direct_mechanism_evidence_in_chinese():
    state = create_initial_agent_state("Can Alpha apply?")

    prompt = build_claim_coverage_prompt(
        state,
        CoverageReport(sufficient=False),
    )

    assert "ruling_fact、case_application 必须至少有一条 entailment=supports 的直接证据" in prompt
    assert "kind 必须严格为 faq、card_supplement 或 rule_passage" in prompt
    assert "card_effect、其他 kind 和 depends_on 推理都不能替代" in prompt
    assert "hypothetical_branch 仍须有上述机制证据" in prompt


def test_agent_loop_rejects_non_submit_tool_when_coverage_is_sufficient():
    answer = {
        "conclusion": "Use cited evidence.",
        "answers": [
            {
                "question": "Can Alpha apply?",
                "case_conclusion": "Use cited evidence.",
                "hypothetical_conclusion": "",
                "reasoning": [
                    {
                        "judgment": "Rule evidence is enough.",
                        "claim_id": "c1",
                        "evidence_ids": ["rule:1"],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": "rule:1",
                "role": "rule_document",
                "use": "Rule evidence.",
            }
        ],
        "uncertainties": [],
    }
    llm = FakeLlm(
        [
            *_deterministic_preparation_actions(),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "Alpha rule"},
                    "rationale": "Need evidence.",
                }
            ),
            json.dumps(
                {
                    "tool": "critique_coverage",
                    "arguments": {},
                    "rationale": "Check claim coverage.",
                }
            ),
            json.dumps(_claim_report_payload()),
            json.dumps(
                {
                    "tool": "search_card",
                    "arguments": {"query": "Alpha Dragon"},
                    "rationale": "Incorrectly keeps searching after claim coverage.",
                }
            ),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": answer},
                    "rationale": "Submit after rejected non-submit action.",
                }
            ),
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings({"RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "5"}),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:1",
                    title="Alpha rule",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="alpha",
                    text="Alpha rule",
                ),
            ]
        ),
        llm=llm,
    )

    response = agent.run("Can Alpha apply?")

    answer_payload = json.loads(response.answer)
    assert answer_payload["conclusion"] == answer["conclusion"]
    assert answer_payload["source_status"] == {
        "faq": "not_searched",
        "rules": "found",
        "failure_reason": "",
    }
    assert any(
        entry["node"] == "execute_tool"
        and entry["metadata"]["observation"]["summary"] == "submit_ruling required"
        for entry in response.agent_trace
    )
    assert any(
        "Coverage is sufficient; the next action must be submit_ruling" in warning
        for warning in response.warnings
    )


def test_agent_loop_blank_submit_reference_does_not_consume_submit_budget():
    valid_answer = _answer_with_claim("c1")
    blank_answer = _answer_with_claim(" ")
    llm = FakeLlm(
        [
            *_deterministic_preparation_actions(),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "Alpha activation rule"},
                    "rationale": "Need mechanism evidence.",
                }
            ),
            json.dumps(
                {
                    "tool": "critique_coverage",
                    "arguments": {},
                    "rationale": "Check claim coverage.",
                }
            ),
            json.dumps(_claim_report_payload()),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": blank_answer},
                    "rationale": "Submit with a repairable blank reference.",
                }
            ),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": valid_answer},
                    "rationale": "Retry with the corrected reference.",
                }
            ),
        ]
    )
    agent = AdjudicationAgent(
        [],
        settings=_settings(
                {
                    "RAG_ADJUDICATION_AGENT_TOTAL_TURN_LIMIT": "6",
                "RAG_ADJUDICATION_AGENT_SUBMIT_LIMIT": "1",
            }
        ),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:1",
                    title="Alpha activation rule",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="alpha",
                    text="Alpha activation rule",
                )
            ]
        ),
        llm=llm,
    )

    response = agent.run("Can Alpha apply?")

    assert json.loads(response.answer)["conclusion"] == valid_answer["conclusion"]
    submit_observations = [
        entry["metadata"]["observation"]
        for entry in response.agent_trace
        if entry["node"] == "execute_tool"
        and entry["metadata"]["action"]["tool"] == "submit_ruling"
    ]
    assert len(submit_observations) == 2
    assert submit_observations[0]["data"]["consumes_submit_budget"] is False
    assert submit_observations[1]["ok"] is True
    assert not any("submit budget exhausted" in warning for warning in response.warnings)


def test_agent_loop_claim_gaps_require_targeted_followup_retrieval():
    answer = _answer_with_claim("c1", evidence_id="rule:target")
    first_report = _claim_report_payload(
        status="unknown",
        evidence_id="rule:first",
        entailment="does_not_entail",
        gaps=[
            {
                "claim_id": "c1",
                "reason": "Need timing evidence for Alpha.",
                "suggested_queries": ["Alpha timing missing"],
            }
        ],
    )
    first_report["required_claims"][0]["gap"] = "Need timing evidence for Alpha."
    first_report["suggested_queries"] = ["Alpha timing missing"]
    llm = FakeLlm(
        [
            *_deterministic_preparation_actions(),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "first evidence"},
                    "rationale": "Need initial evidence.",
                }
            ),
            json.dumps(
                {
                    "tool": "critique_coverage",
                    "arguments": {},
                    "rationale": "Find claim gaps.",
                }
            ),
            json.dumps(first_report),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "unrelated broad query"},
                    "rationale": "This does not target the reported gap.",
                }
            ),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "Alpha timing missing"},
                    "rationale": "Targets c1 gap.",
                }
            ),
            json.dumps(
                {
                    "tool": "critique_coverage",
                    "arguments": {},
                    "rationale": "Re-check claim coverage.",
                }
            ),
            json.dumps(_claim_report_payload(evidence_id="rule:target")),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": answer},
                    "rationale": "Claim coverage is sufficient.",
                }
            ),
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings({"RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "6"}),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:first",
                    title="First rule",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="first",
                    text="first evidence",
                ),
                RuleSection(
                    rule_id="rule:target",
                    title="Target rule",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="target",
                    text="Alpha timing missing",
                ),
            ]
        ),
        llm=llm,
    )

    response = agent.run("Can Alpha apply?")

    answer_payload = json.loads(response.answer)
    assert answer_payload["conclusion"] == answer["conclusion"]
    assert answer_payload["source_status"]["rules"] == "found"
    assert "claim_coverage_report" in response.agent_coverage
    assert any(item.evidence_id == "rule:target" for item in response.evidence)
    assert any(
        entry["node"] == "execute_tool"
        and entry["metadata"]["observation"]["summary"]
        == "claim gap targeted retrieval required"
        for entry in response.agent_trace
    )
    critique_entries = [
        entry
        for entry in response.agent_trace
        if entry["node"] == "agent_plan"
        and entry["metadata"]["action"]["tool"] == "critique_coverage"
    ]
    assert len(critique_entries) == 2
    assert all(not entry["metadata"]["duplicate"] for entry in critique_entries)


def test_search_rules_rejects_card_name_heavy_query_and_accepts_mechanism_query():
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:destroy",
                    title="Destruction replacement",
                    chapter_path=("Rules",),
                    xhtml_path="rules.xhtml",
                    anchor="destroy",
                    text="破坏 被代替 后续处理",
                )
            ]
        ),
    )
    state = agent._replace(
        create_initial_agent_state("Can Alpha Dragon apply?"),
        resolved_cards=(
            ResolvedCard(
                mention="Alpha Dragon",
                matches=(Card(1, "Alpha Dragon", "Destroy one card."),),
            ),
        ),
    )

    _, rejected = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="search_rules",
            arguments={"query": "Alpha Dragon"},
        ),
    )
    _, accepted = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="search_rules",
            arguments={"query": "破坏 被代替 后续处理"},
        ),
    )

    assert rejected.ok is False
    assert "mechanism-focused" in rejected.validation_error
    assert accepted.ok is True
    assert accepted.evidence_ids == ("rule:destroy",)


def test_agent_classifies_rule_chapters_then_searches_selected_chapter():
    replacement_chapter = "c03/resistance.xhtml"
    chain_chapter = "c02/chain.xhtml"
    agent = AdjudicationAgent(
        [],
        settings=_settings(),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:replacement",
                    title="代替破坏",
                    chapter_path=("深入OCG规则", "卡片的抗性", "代替破坏"),
                    xhtml_path=replacement_chapter,
                    anchor="replace",
                    text="破坏被代替后的后续处理。",
                ),
                RuleSection(
                    rule_id="rule:chain",
                    title="如何结算连锁",
                    chapter_path=("OCG规则基础", "连锁基础", "如何结算连锁"),
                    xhtml_path=chain_chapter,
                    anchor="resolve",
                    text="连锁中的破坏处理。",
                ),
            ]
        ),
    )
    state = create_initial_agent_state("破坏被代替后是否继续处理？")

    classified_state, classified = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="classify_rule_chapters",
            arguments={"query": "破坏代替 后续处理", "top_k": 2},
        ),
    )
    searched_state, searched = agent._execute_agent_tool(
        classified_state,
        PlannerAction(
            tool="search_rules",
            arguments={
                "query": "破坏代替 后续处理",
                "chapter_ids": [replacement_chapter],
            },
        ),
    )

    assert classified.ok is True
    assert classified_state.rule_chapter_candidates[0]["chapter_id"] == replacement_chapter
    assert searched.ok is True
    assert searched.evidence_ids == ("rule:replacement",)
    assert searched_state.evidence_retrieval_stages[-1]["chapter_ids"] == [
        replacement_chapter
    ]
    assert "rule_chapter_candidates" in build_agent_planner_prompt(classified_state)


def test_agent_planner_sees_full_catalog_and_can_search_outside_lexical_top_k():
    replacement_chapter = "c03/resistance.xhtml"
    chain_chapter = "c02/chain.xhtml"
    rule_index = RuleIndex(
        [
            RuleSection(
                rule_id="rule:replacement",
                title="代替破坏",
                chapter_path=("深入OCG规则", "卡片的抗性", "代替破坏"),
                xhtml_path=replacement_chapter,
                anchor="replace",
                text="破坏被代替后的后续处理。",
            ),
            RuleSection(
                rule_id="rule:chain",
                title="如何结算连锁",
                chapter_path=("OCG规则基础", "连锁基础", "如何结算连锁"),
                xhtml_path=chain_chapter,
                anchor="resolve",
                text="连锁处理按照逆序结算。",
            ),
        ]
    )
    agent = AdjudicationAgent([], settings=_settings(), rule_index=rule_index)
    state = agent._initialize_agent_loop_state(
        create_initial_agent_state("破坏被代替后是否继续处理？")
    )
    state, split_observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="split_questions",
            arguments={"strategy": "deterministic"},
        ),
    )
    state, mention_observation = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="extract_card_mentions",
            arguments={"strategy": "deterministic"},
        ),
    )
    assert split_observation.ok is True
    assert mention_observation.ok is True
    classified_state, classified = agent._execute_agent_tool(
        state,
        PlannerAction(
            tool="classify_rule_chapters",
            arguments={"query": "破坏代替 后续处理", "top_k": 1},
        ),
    )

    assert classified.ok is True
    assert [item["chapter_id"] for item in classified_state.rule_chapter_candidates] == [
        replacement_chapter
    ]
    prompt = build_agent_planner_prompt(classified_state)
    assert replacement_chapter in prompt
    assert chain_chapter in prompt
    assert "完整可选目录" in prompt

    _, searched = agent._execute_agent_tool(
        classified_state,
        PlannerAction(
            tool="search_rules",
            arguments={
                "query": "连锁 逆序 结算",
                "chapter_ids": [chain_chapter],
            },
        ),
    )
    assert searched.ok is True
    assert searched.evidence_ids == ("rule:chain",)


def test_planner_accepts_chapter_classification_and_filtered_rule_search_tools():
    classify = parse_agent_planner_response(
        json.dumps(
            {
                "tool": "classify_rule_chapters",
                "arguments": {"query": "代替破坏 后续处理", "top_k": 3},
            },
            ensure_ascii=False,
        )
    )
    search = parse_agent_planner_response(
        json.dumps(
            {
                "tool": "search_rules",
                "arguments": {
                    "query": "代替破坏 后续处理",
                    "chapter_ids": ["c03/resistance.xhtml"],
                },
            },
            ensure_ascii=False,
        )
    )

    assert classify.tool == "classify_rule_chapters"
    assert search.arguments["chapter_ids"] == ["c03/resistance.xhtml"]


def test_source_status_uses_actual_evidence_and_attempts_for_faq_decisive_case():
    agent = AdjudicationAgent([], settings=_settings())
    faq = EvidenceItem(
        evidence_id="ygocdb:faq:1",
        kind="faq",
        source="ygocdb",
        text="FAQ 明确给出本案裁定。",
        relationship="direct_card",
    )
    state = agent._replace(
        create_initial_agent_state("本案如何处理？"),
        scenario=ParsedScenario(
            original_text="本案如何处理？",
            questions=("本案如何处理？",),
        ),
        evidence=(faq,),
        evidence_retrieval_stages=(
            {"stage": "direct_faq", "returned_evidence_ids": [faq.evidence_id]},
            {"stage": "rules", "returned_evidence_ids": []},
        ),
        coverage=CoverageReport(sufficient=True),
    )

    payload = json.loads(agent._deterministic_answer(state))

    assert payload["confidence"] == "medium"
    assert payload["source_status"] == {
        "faq": "found",
        "rules": "not_found",
        "failure_reason": "已依据 FAQ 证据完成裁定，但未找到对应的规则文档机制。",
    }


def test_source_status_marks_low_confidence_when_faq_is_insufficient_and_rules_missing():
    agent = AdjudicationAgent([], settings=_settings())
    faq = EvidenceItem(
        evidence_id="ygocdb:faq:analogous",
        kind="faq",
        source="ygocdb",
        text="只有相似机制，不能完整覆盖本案。",
        relationship="analogous",
    )
    state = agent._replace(
        create_initial_agent_state("本案如何处理？"),
        scenario=ParsedScenario(
            original_text="本案如何处理？",
            questions=("本案如何处理？",),
        ),
        evidence=(faq,),
        evidence_retrieval_stages=(
            {"stage": "analogous_faq", "returned_evidence_ids": [faq.evidence_id]},
            {"stage": "rules", "returned_evidence_ids": []},
        ),
        coverage=CoverageReport(
            sufficient=False,
            missing_evidence=("缺少本案直接机制证据",),
            next_action="retry",
        ),
    )

    payload = json.loads(agent._deterministic_answer(state))

    assert payload["confidence"] == "low"
    assert payload["source_status"]["faq"] == "found"
    assert payload["source_status"]["rules"] == "not_found"
    assert "FAQ 证据不足" in payload["source_status"]["failure_reason"]


def test_non_llm_agent_card_effect_only_is_insufficient_after_empty_adjudication_searches():
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
        ruling_index=YgocdbRulingIndex.from_payloads([]),
        rule_index=RuleIndex([]),
    )

    response = agent.run("Can Alpha Dragon activate?")
    answer = json.loads(response.answer)

    assert any(item.kind == "card_effect" for item in response.evidence)
    assert not any(
        item.kind in {"faq", "card_supplement", "rule_passage"}
        for item in response.evidence
    )
    assert response.agent_coverage["sufficient"] is False
    assert response.confidence == "low"
    assert answer["confidence"] == "low"
    assert answer["source_status"]["faq"] == "not_found"
    assert answer["source_status"]["rules"] == "not_found"
    assert answer["source_status"]["failure_reason"] == (
        "未找到 FAQ 和规则文档证据，当前证据不足。"
    )


def test_non_llm_agent_preserves_decisive_faq_only_sufficiency():
    ruling_index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 1,
                "faqs": [
                    {
                        "fid": "alpha-activation",
                        "title": "Alpha Dragon 的发动",
                        "question": "这张卡可以发动吗？",
                        "answer": "可以发动。",
                        "refer": {},
                    }
                ],
            }
        ]
    )
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings(),
        ruling_index=ruling_index,
        rule_index=RuleIndex([]),
    )

    response = agent.run("Can Alpha Dragon activate?")
    answer = json.loads(response.answer)

    assert response.agent_coverage["sufficient"] is True
    assert response.confidence == "medium"
    assert answer["confidence"] == "medium"
    assert answer["source_status"] == {
        "faq": "found",
        "rules": "not_found",
        "failure_reason": "已依据 FAQ 证据完成裁定，但未找到对应的规则文档机制。",
    }


def test_agent_progress_events_are_concise_and_do_not_expose_private_planner_fields():
    private_text = "不要泄露的内部假设"
    answer = {
        "conclusion": "依据规则证据处理。",
        "answers": [
            {
                "question": "Alpha Dragon 可以发动吗？",
                "case_conclusion": "可以按规则证据处理。",
                "hypothetical_conclusion": "",
                "reasoning": [
                    {
                        "judgment": "规则段落支持该结论。",
                        "evidence_ids": ["rule:activation"],
                    }
                ],
            }
        ],
        "evidence_use": [
            {
                "evidence_id": "rule:activation",
                "role": "rule_document",
                "use": "发动规则。",
            }
        ],
        "uncertainties": [],
    }
    llm = FakeLlm(
        [
            *_deterministic_preparation_actions(),
            json.dumps(
                {
                    "tool": "resolve_cards",
                    "arguments": {"mentions": ["Alpha Dragon"]},
                    "rationale": private_text,
                    "notes": [private_text],
                    "hypotheses": [private_text],
                }
            ),
            json.dumps(
                {
                    "tool": "search_rules",
                    "arguments": {"query": "怪兽效果 发动 条件"},
                    "rationale": private_text,
                }
            ),
            json.dumps(
                {
                    "tool": "submit_ruling",
                    "arguments": {"answer": answer},
                    "rationale": private_text,
                }
            ),
        ]
    )
    events = []
    agent = AdjudicationAgent(
        [Card(1, "Alpha Dragon", "Destroy one card.")],
        settings=_settings({"RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "4"}),
        rule_index=RuleIndex(
            [
                RuleSection(
                    rule_id="rule:activation",
                    title="发动条件",
                    chapter_path=("规则", "发动条件"),
                    xhtml_path="rules.xhtml",
                    anchor="activation",
                    text="怪兽效果的发动条件。",
                )
            ]
        ),
        llm=llm,
        progress_callback=events.append,
    )

    agent.run("Alpha Dragon 可以发动吗？")

    stages = {event.stage for event in events}
    assert {
        "parse",
        "parse_tool",
        "card_resolution",
        "planner",
        "tool",
        "coverage",
        "final",
    }.issubset(stages)
    parse_messages = [event.message for event in events if event.stage == "parse_tool"]
    assert any(message.startswith("开始拆分裁定问题") for message in parse_messages)
    assert any(message.startswith("卡名抽取完成") for message in parse_messages)
    assert all(private_text not in event.message for event in events)
    assert all(set(event.to_dict()).isdisjoint({"rationale", "notes", "hypotheses"}) for event in events)


def test_sample_scenario_resolves_three_named_cards():
    cards = [
        Card(68625623, UNCHAINED_SOUL_OF_RAGE, "Destroy one card."),
        Card(94014327, UNCHAINED_ABOMINATION, "Destroy one monster."),
        Card(13203964, CYBER_DRAGON, "Lose 1000 ATK instead of being destroyed."),
    ]
    agent = AdjudicationAgent(cards, settings=_settings())

    response = agent.run(
        f"对方发动了手卡{UNCHAINED_SOUL_OF_RAGE}的效果，要将场上的"
        f"{UNCHAINED_ABOMINATION}破坏，此时对方选择适用{UNCHAINED_ABOMINATION}"
        f"的效果想要破坏我方场上的{CYBER_DRAGON}，{CYBER_DRAGON}可以作为"
        "被破坏的替代降低1000攻击力吗？如果适用降低1000攻击力，对方还能特殊召唤吗？"
    )

    assert {card.matches[0].card_id for card in response.resolved_cards} == {
        68625623,
        94014327,
        13203964,
    }


def test_adjudication_case_file_unchained_replacement_regression():
    case_dir = Path(__file__).resolve().parents[1] / "scripts" / "adjudication_cases"
    scenario = (case_dir / "1.txt").read_text(encoding="utf-8")
    expected = json.loads(
        (case_dir / "unchained_replacement.expected.json").read_text(encoding="utf-8")
    )
    cards = [
        Card(68625623, UNCHAINED_SOUL_OF_RAGE, "Destroy one card."),
        Card(94014327, UNCHAINED_ABOMINATION, "Destroy one monster."),
        Card(13203964, CYBER_DRAGON, "Lose 1000 ATK instead of being destroyed."),
    ]
    ruling_index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 68625623,
                "faqs": [],
                "supplement": {
                    "text": "Supplement text for the hand effect.",
                    "refer": {},
                    "date": "2026",
                },
            },
            {
                "id": 94014327,
                "faqs": [
                    {
                        "fid": "24336",
                        "title": "Destruction replacement",
                        "question": "If destruction is replaced?",
                        "answer": "If the card was not destroyed, later processing is not applied.",
                        "refer": {"68625623": True},
                    }
                ],
            },
        ]
    )
    agent = AdjudicationAgent(
        cards,
        settings=_settings(),
        ruling_index=ruling_index,
    )

    response = agent.run(scenario)

    assert {card.matches[0].card_id for card in response.resolved_cards} == set(
        expected["expected_card_ids"]
    )
    assert set(expected["expected_evidence_ids"]).issubset(
        {item.evidence_id for item in response.evidence}
    )
    assert response.agent_coverage["sufficient"] is expected["require_sufficient"]


def test_agent_prompt_includes_coverage_and_trace():
    state = create_initial_agent_state("If Alpha applies, can Beta summon?")
    state = AdjudicationAgent([], settings=_settings())._replace(
        state,
        scenario=ParsedScenario(
            original_text=state.scenario_text,
            questions=("If Alpha applies, can Beta summon?",),
        ),
        evidence=(
            EvidenceItem(
                evidence_id="ygocdb:faq:24345",
                kind="faq",
                source="ygocdb",
                text="If the target was not destroyed, the special summon is not applied.",
                relationship="analogous",
            ),
        ),
        coverage=CoverageReport(sufficient=True),
    )

    prompt = build_agent_adjudication_answer_prompt(state)

    assert "coverage_report" in prompt
    assert "tool_trace_summary" in prompt
    assert "card_resolution" in prompt
    assert "ygocdb:faq:24345" in prompt


def test_dedupe_evidence_keeps_stronger_relationship_and_stage_metadata():
    evidence = dedupe_evidence(
        [
            EvidenceItem(
                evidence_id="ygocdb:faq:same",
                kind="faq",
                source="ygocdb",
                text="Analogous text.",
                relationship="analogous",
                metadata={"retrieval_stage": "analogous_faq"},
            ),
            EvidenceItem(
                evidence_id="ygocdb:faq:same",
                kind="faq",
                source="ygocdb",
                text="Direct text.",
                relationship="direct_card",
                card_ids=(1,),
                metadata={"retrieval_stage": "direct_faq"},
            ),
        ]
    )

    assert len(evidence) == 1
    assert evidence[0].relationship == "direct_card,analogous"
    assert evidence[0].metadata["retrieval_stages"] == ["analogous_faq", "direct_faq"]


def test_agent_uses_persistent_ruling_retriever_before_runtime_indexes():
    class FakeRulingRetriever:
        def search(self, *args, **kwargs):
            return RulingRetrievalResult(
                candidates=(
                    RulingCandidate(
                        evidence_id="ygocdb:faq:24345",
                        score=1.0,
                        source="hybrid",
                        relationship="analogous",
                        document=RulingDocument(
                            evidence_id="ygocdb:faq:24345",
                            kind="faq",
                            source="ygocdb",
                            title="Later processing",
                            text="If the target was not destroyed, the special summon is not applied.",
                        ),
                    ),
                )
            )

    cards = [Card(1, "Alpha", "Destroy one card.")]
    agent = AdjudicationAgent(
        cards,
        settings=_settings(),
        ruling_retrievers=(FakeRulingRetriever(),),
    )

    response = agent.run("If Alpha is not destroyed, can it special summon?")

    assert any(item.evidence_id == "ygocdb:faq:24345" for item in response.evidence)
    assert response.agent_coverage is not None
    assert response.agent_coverage["missing_hypothetical_evidence"] == []
    retrieve_trace = [
        entry for entry in response.agent_trace if entry["node"] == "retrieve_evidence"
    ][0]
    assert retrieve_trace["metadata"]["used_persistent_ruling_retrieval"] is True


def _settings(extra: dict[str, str] | None = None) -> Settings:
    env = {
        "RAG_DATA_DIR": "data",
        "RAG_ADJUDICATION_AGENT_MAX_ITERATIONS": "1",
        "RAG_ADJUDICATION_CARD_CONFIDENCE_THRESHOLD": "0.75",
        "RAG_ADJUDICATION_CARD_AMBIGUITY_MARGIN": "0.08",
        "RAG_ADJUDICATION_LLM_SCENARIO_PARSING": "0",
    }
    if extra:
        env.update(extra)
    return Settings.from_env(
        env
    )
