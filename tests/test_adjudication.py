import json
import zipfile

import pytest

from rag_agent.adjudication import (
    EvidenceItem,
    ParsedScenario,
    RuleIndex,
    RuleSection,
    RulingAdjudicator,
    SimulationStatus,
    YgocdbCache,
    YgocdbRulingIndex,
    build_adjudication_answer_prompt,
    html_to_plain_text,
    normalize_adjudication_answer_source_status,
    parse_ocg_rules_epub,
    parse_scenario,
    resolve_card_mentions,
    _split_long_rule_paragraph,
    _extract_question_candidates,
    _minimal_question_clause,
    validate_adjudication_answer_json,
    validate_card_mention_extraction_payload,
    validate_evidence_citations,
)
from rag_agent.cards import Card


def test_card_mention_validation_drops_unquoted_short_nicknames_but_keeps_exact_cards():
    source = "对手发动大猫效果，再发动雷破；之后从卡组特殊召唤天杯龙。"
    mentions = validate_card_mention_extraction_payload(
        {"mentions": ["大猫", "雷破", "天杯龙"]},
        source,
        known_cards=(Card(1, "雷破", ""),),
    )

    assert mentions == ("雷破",)


def test_card_mention_validation_keeps_quoted_short_unknown_name_for_resolution():
    source = "对手发动「雷破」的效果。"

    mentions = validate_card_mention_extraction_payload(
        {"mentions": ["雷破"]},
        source,
        known_cards=(),
    )

    assert mentions == ("雷破",)


def test_ygocdb_faq_ingestion_dedupes_and_indexes_refer_cards():
    payloads = [
        {
            "id": 100,
            "faqs": [
                {
                    "fid": "f1",
                    "title": '<a data-refer="200">Alpha</a>',
                    "date": "2026-01-01",
                    "question": "Can it destroy?",
                    "answer": "Yes.",
                    "refer": {"200": True},
                }
            ],
        },
        {
            "id": 300,
            "faqs": [
                {
                    "fid": "f1",
                    "title": "Duplicate",
                    "date": "2026-01-01",
                    "question": "Can it destroy?",
                    "answer": "Yes.",
                    "refer": {"400": True},
                }
            ],
        },
    ]

    index = YgocdbRulingIndex.from_payloads(payloads, snapshot_at="snapshot")

    assert len(index.faq_by_fid) == 1
    faq = index.faq_by_fid["f1"]
    assert faq.primary_card_ids == (100, 300)
    assert faq.refer_card_ids == (200, 400)
    assert index.direct_faqs(100)[0].fid == "f1"
    assert index.related_faqs(200)[0].fid == "f1"
    assert index.related_faqs(200)[0].to_evidence("related_card").relationship == "related_card"


def test_ygocdb_supplement_is_lower_priority_evidence():
    index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 100,
                "supplement": {
                    "text": "<ul><li>This is a Dragon monster.</li></ul>",
                    "date": "2021-01-01",
                    "refer": {"200": True},
                },
            }
        ]
    )

    supplement = index.supplement(100)

    assert supplement is not None
    evidence = supplement.to_evidence()
    assert evidence.kind == "card_supplement"
    assert evidence.relationship == "card_supplement"
    assert "Dragon monster" in evidence.text
    assert 200 in evidence.card_ids


def test_ygocdb_faq_search_boosts_same_effect_category_cards():
    index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 100,
                "faqs": [
                    {
                        "fid": "destroy_like",
                        "title": "Unrelated title",
                        "question": "Unrelated question",
                        "answer": "Unrelated answer.",
                        "refer": {},
                    },
                ],
            }
        ]
    )

    results = index.search(
        "no matching text",
        effect_tags=("破坏",),
        card_effect_tags_by_id={100: ("破坏", "特殊召唤")},
    )

    assert [faq.fid for faq in results] == ["destroy_like"]


def test_html_to_plain_text_strips_links_but_keeps_text():
    assert html_to_plain_text('<a data-refer="1">Card</a><br>Answer') == "Card Answer"


def test_ygocdb_cache_loads_directory_payload(tmp_path):
    cache_dir = tmp_path / "ygocdb"
    cache_dir.mkdir()
    (cache_dir / "100.json").write_text(
        json.dumps({"id": 100, "faqs": []}),
        encoding="utf-8",
    )

    assert YgocdbCache(cache_dir).load_card_payload(100)["id"] == 100


def test_ocg_rules_epub_parser_uses_nav_sections(tmp_path):
    epub_path = tmp_path / "rules.epub"
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>
<li><a href="c03/rule.xhtml#replace">Replacement</a></li>
</ol></nav></body></html>"""
    rule = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<section id="replace"><h1>Replacement</h1><p>Destroy replacement applies instead.</p></section>
</body></html>"""
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("nav.xhtml", nav)
        archive.writestr("c03/rule.xhtml", rule)

    index = RuleIndex.from_epub(epub_path)

    assert index.sections
    section = index.sections[0]
    assert section.rule_id == "ocgrules:c03/rule.xhtml#replace"
    assert section.xhtml_path == "c03/rule.xhtml"
    assert "Destroy replacement" in section.text
    assert index.search("destroy replacement")[0].rule_id == section.rule_id


def test_ocg_rules_epub_anchor_extraction_stops_before_next_heading(tmp_path):
    epub_path = tmp_path / "rules.epub"
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>
<li><a href="rules.xhtml#first">First Rule</a></li>
<li><a href="rules.xhtml#second">Second Rule</a></li>
</ol></nav></body></html>"""
    rule = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 id="first">First Rule</h2><p>Destroy replacement applies here.</p>
<p>Keep the first example.</p>
<h2 id="second">Second Rule</h2><p>Special summon failure applies here.</p>
</body></html>"""
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("nav.xhtml", nav)
        archive.writestr("rules.xhtml", rule)

    index = RuleIndex.from_epub(epub_path)
    first = index.section_by_id["ocgrules:rules.xhtml#first"]
    second = index.section_by_id["ocgrules:rules.xhtml#second"]

    assert "Destroy replacement" in first.text
    assert "Keep the first example" in first.text
    assert "Special summon failure" not in first.text
    assert "Special summon failure" in second.text
    assert index.search("Second Rule", top_k=1)[0].rule_id == second.rule_id


def test_ocg_rules_epub_skips_page_top_when_anchor_entries_exist(tmp_path):
    epub_path = tmp_path / "rules.epub"
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>
<li><a href="rules.xhtml">Whole Page</a></li>
<li><a href="rules.xhtml#first">First Rule</a></li>
</ol></nav></body></html>"""
    rule = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Whole Page</h1>
<h2 id="first">First Rule</h2><p>Specific anchored text.</p>
</body></html>"""
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("nav.xhtml", nav)
        archive.writestr("rules.xhtml", rule)

    sections = parse_ocg_rules_epub(epub_path)

    assert [section.rule_id for section in sections] == ["ocgrules:rules.xhtml#first"]


def test_ocg_rules_epub_chunks_by_paragraphs(tmp_path):
    epub_path = tmp_path / "rules.epub"
    nav = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>
<li><a href="rules.xhtml#long">Long Rule</a></li>
</ol></nav></body></html>"""
    rule = """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 id="long">Long Rule</h2>
<p>Alpha paragraph about replacement.</p>
<p>Beta paragraph about special summon.</p>
<p>Gamma paragraph about resolution.</p>
</body></html>"""
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("nav.xhtml", nav)
        archive.writestr("rules.xhtml", rule)

    sections = parse_ocg_rules_epub(epub_path, max_section_chars=70)

    assert len(sections) >= 2
    assert all(len(section.text) <= 90 for section in sections)
    assert any("Alpha paragraph" in section.text for section in sections)
    assert any("Gamma paragraph" in section.text for section in sections)


def test_long_chinese_rule_paragraph_chunks_on_sentence_punctuation():
    paragraph = (
        "第一句说明破坏代替。"
        "第二句说明效果处理！"
        "第三句询问是否继续？"
        "第四句说明连锁结算。"
    )

    chunks = _split_long_rule_paragraph(paragraph, max_chars=16)

    assert chunks == [
        "第一句说明破坏代替。",
        "第二句说明效果处理！",
        "第三句询问是否继续？",
        "第四句说明连锁结算。",
    ]


def test_rule_index_classifies_and_filters_epub_chapters():
    replacement_chapter = "c03/resistance.xhtml"
    chain_chapter = "c02/chain.xhtml"
    index = RuleIndex(
        [
            RuleSection(
                rule_id="rule:replacement:chunk:1",
                title="代替破坏",
                chapter_path=("深入OCG规则", "卡片的抗性", "代替破坏"),
                xhtml_path=replacement_chapter,
                anchor="replace",
                text="破坏可以被其他处理代替。",
            ),
            RuleSection(
                rule_id="rule:replacement:chunk:2",
                title="代替破坏",
                chapter_path=("深入OCG规则", "卡片的抗性", "代替破坏"),
                xhtml_path=replacement_chapter,
                anchor="replace",
                text="代替适用后检查后续处理。",
            ),
            RuleSection(
                rule_id="rule:chain",
                title="如何结算连锁",
                chapter_path=("OCG规则基础", "连锁基础", "如何结算连锁"),
                xhtml_path=chain_chapter,
                anchor="resolve",
                text="连锁处理中也可能提到破坏。",
            ),
        ]
    )

    chapters = index.classify_chapters("代替破坏 后续处理", top_k=2)
    catalog = index.chapter_catalog()
    filtered = index.search("破坏 处理", chapter_ids=(chain_chapter,), top_k=5)

    assert [entry.chapter_id for entry in catalog] == [
        replacement_chapter,
        chain_chapter,
    ]
    assert catalog[1].title == "连锁基础"
    assert catalog[1].section_titles == ("如何结算连锁",)
    assert chapters[0].chapter_id == replacement_chapter
    assert chapters[0].chapter_path == ("深入OCG规则", "卡片的抗性")
    assert chapters[0].section_titles == ("代替破坏",)
    assert [section.rule_id for section in filtered] == ["rule:chain"]


def test_parse_scenario_and_resolve_cards():
    cards = [
        Card(1, "Alpha Dragon", "Destroy one card."),
        Card(2, "Beta Serpent", "If it would be destroyed, lose 1000 ATK instead."),
    ]

    scenario = parse_scenario(
        "Can Alpha Dragon destroy Beta Serpent, and can Beta Serpent apply instead?",
        known_cards=cards,
    )
    resolved = resolve_card_mentions(scenario.card_mentions, cards)

    assert scenario.card_mentions == ("Alpha Dragon", "Beta Serpent")
    assert "destroy" in scenario.original_text.lower()
    assert [item.matches[0].card_id for item in resolved] == [1, 2]


@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("「", "」"),
        ("『", "』"),
        ("“", "”"),
        ("‘", "’"),
        ("《", "》"),
        ("〈", "〉"),
        ("〔", "〕"),
        ("〖", "〗"),
        ('"', '"'),
        ("（", "）"),
        ("(", ")"),
        ("【", "】"),
        ("[", "]"),
    ],
)
def test_question_marks_inside_paired_card_effect_delimiters_are_content(opening, closing):
    text = f"卡片效果文本为{opening}支付1000 LP，可以发动吗？{closing}这个效果能否适用？"

    assert parse_scenario(text).questions == (text,)


def test_question_extractor_handles_nested_parentheses_in_card_effect_text():
    text = "若卡片效果文本为（支付1000 LP（这个效果可以发动吗？）后继续），这个效果是否适用？"

    assert parse_scenario(text).questions == ("这个效果是否适用？",)


def test_minimal_question_clause_preserves_outer_question_after_delimiter():
    text = "这个效果（『可以发动吗？』）？"

    assert _minimal_question_clause(text) == text


def test_unclosed_delimiter_keeps_internal_question_out_of_candidates():
    text = "卡片效果文本为「支付1000 LP，可以发动吗？"

    assert _extract_question_candidates(text) == ()


def test_ascii_apostrophe_does_not_open_question_delimiter():
    text = "Card's effect can activate?"

    assert _extract_question_candidates(text) == (text,)


def test_adjudicator_gathers_direct_related_supplement_and_rules(tmp_path):
    cards = [
        Card(100, "Alpha Dragon", "Destroy one monster."),
        Card(200, "Beta Serpent", "Lose 1000 ATK instead of being destroyed."),
    ]
    ruling_index = YgocdbRulingIndex.from_payloads(
        [
            {
                "id": 100,
                "faqs": [
                    {
                        "fid": "direct",
                        "title": "Direct ruling",
                        "question": "Destroy?",
                        "answer": "It destroys.",
                        "refer": {},
                    },
                    {
                        "fid": "related",
                        "title": "Related ruling",
                        "question": "Replacement?",
                        "answer": "Use instead.",
                        "refer": {"200": True},
                    },
                ],
                "supplement": {"text": "Supplement text.", "refer": {}, "date": "2020"},
            }
        ]
    )
    epub_path = tmp_path / "rules.epub"
    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr(
            "nav.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol>'
            '<li><a href="rules.xhtml#r">Destroy replacement</a></li>'
            "</ol></nav></body></html>",
        )
        archive.writestr(
            "rules.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            '<p>Destroy replacement and effect resolution.</p></body></html>',
        )
    adjudicator = RulingAdjudicator(
        cards,
        ruling_index=ruling_index,
        rule_index=RuleIndex.from_epub(epub_path),
    )

    response = adjudicator.adjudicate(
        "Can Alpha Dragon destroy Beta Serpent? Can Beta Serpent apply instead?"
    )
    relationships = {item.relationship for item in response.evidence}
    kinds = {item.kind for item in response.evidence}

    assert any("direct_card" in relationship for relationship in relationships)
    assert any("related_card" in relationship for relationship in relationships)
    assert "card_supplement" in kinds
    assert "rule_passage" in kinds
    assert response.simulation.status == "not_requested"


def test_adjudicator_reports_unavailable_simulation():
    response = RulingAdjudicator([Card(1, "Alpha", "Text")]).adjudicate(
        "Can Alpha activate?",
        simulate=True,
    )

    assert response.simulation.status == "not_available"
    assert any("Simulator" in warning for warning in response.warnings)


def test_adjudicator_uses_custom_simulation_verifier():
    class Verifier:
        def verify(self, scenario):
            return SimulationStatus(status="not_encoded", warnings=("not encoded",))

    response = RulingAdjudicator(
        [Card(1, "Alpha", "Text")],
        simulator=Verifier(),
    ).adjudicate("Can Alpha activate?", simulate=True)

    assert response.simulation.status == "not_encoded"
    assert "not encoded" in response.warnings


def test_validate_evidence_citations_rejects_unknown_ids():
    validate_evidence_citations("Use [card:1:effect].", {"card:1:effect"})

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_evidence_citations("Use [card:2:effect].", {"card:1:effect"})


def test_adjudication_answer_prompt_requires_structured_hypothetical_json():
    scenario = ParsedScenario(
        original_text="Can Alpha apply? If it applies, can Beta summon?",
        questions=("Can Alpha apply?", "If it applies, can Beta summon?"),
    )
    evidence = [
        EvidenceItem(
            evidence_id="ygocdb:faq:1",
            kind="faq",
            source="ygocdb",
            text="If the target was not destroyed, the later summon is not applied.",
            relationship="analogous",
            title="Analogous ruling",
        )
    ]

    prompt = build_adjudication_answer_prompt(scenario, evidence)

    assert "只返回严格 JSON" in prompt
    assert "在 conclusion 字段最先给出整体裁定结论" in prompt
    assert "每个问题都要分别回答" in prompt
    assert "hypothetical_conclusion" in prompt
    assert "source_status" in prompt
    assert "不能只说前提不成立就停止" in prompt
    assert "direct_card FAQ" in prompt
    assert "优先" in prompt
    assert "说明类比关系" in prompt
    assert "破坏了该效果要求破坏的对象/那些卡" in prompt
    assert "ygocdb:faq:1" in prompt

def test_validate_adjudication_answer_json_rejects_unknown_ids():
    answer = json.dumps(
        {
            "conclusion": "Test.",
            "source_status": {
                "faq": "not_searched",
                "rules": "not_searched",
                "failure_reason": "",
            },
            "answers": [
                {
                    "question": "Q",
                    "case_conclusion": "A",
                    "hypothetical_conclusion": "",
                    "reasoning": [
                        {
                            "judgment": "J",
                            "evidence_ids": ["card:1:effect"],
                        }
                    ],
                }
            ],
            "evidence_use": [
                {
                    "evidence_id": "card:1:effect",
                    "role": "card_effect",
                    "use": "U",
                }
            ],
            "uncertainties": [],
        },
        ensure_ascii=False,
    )
    validate_adjudication_answer_json(answer, {"card:1:effect"})

    with pytest.raises(ValueError, match="unknown evidence"):
        validate_adjudication_answer_json(answer, {"card:2:effect"})


def _valid_adjudication_answer_payload():
    return {
        "conclusion": "Test.",
        "source_status": {
            "faq": "not_searched",
            "rules": "not_searched",
            "failure_reason": "",
        },
        "answers": [
            {
                "question": "Q",
                "case_conclusion": "A",
                "hypothetical_conclusion": "",
                "reasoning": [{"judgment": "J", "evidence_ids": []}],
            }
        ],
        "evidence_use": [],
        "uncertainties": [],
        "confidence": "medium",
    }


def test_validate_adjudication_answer_json_accepts_compatible_confidence_field():
    payload = _valid_adjudication_answer_payload()

    validate_adjudication_answer_json(json.dumps(payload), set())


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda payload: payload.update(conclusion=""),
            "LLM adjudication answer field conclusion must be a non-empty string.",
        ),
        (
            lambda payload: payload["answers"][0].update(question=""),
            "LLM adjudication answer field answers[0].question must be a non-empty string.",
        ),
        (
            lambda payload: payload["answers"][0].update(case_conclusion=None),
            "LLM adjudication answer field answers[0].case_conclusion must be a non-empty string.",
        ),
        (
            lambda payload: payload["answers"][0].update(hypothetical_conclusion=None),
            "LLM adjudication answer field answers[0].hypothetical_conclusion must be a string.",
        ),
        (
            lambda payload: payload["answers"][0].update(reasoning={}),
            "LLM adjudication answer field answers[0].reasoning must be an array.",
        ),
        (
            lambda payload: payload["answers"][0]["reasoning"][0].update(judgment=""),
            "LLM adjudication answer field answers[0].reasoning[0].judgment must be a non-empty string.",
        ),
        (
            lambda payload: payload["answers"][0]["reasoning"][0].update(evidence_ids=[1]),
            "LLM adjudication answer field answers[0].reasoning[0].evidence_ids must be an array of strings.",
        ),
        (
            lambda payload: payload.update(evidence_use=[{"evidence_id": "id"}]),
            "LLM adjudication answer field evidence_use[0].role is required.",
        ),
        (
            lambda payload: payload.update(uncertainties=[1]),
            "LLM adjudication answer field uncertainties must be an array of strings.",
        ),
    ],
)
def test_validate_adjudication_answer_json_validates_submit_schema_fields(mutate, expected):
    payload = _valid_adjudication_answer_payload()
    mutate(payload)

    with pytest.raises(ValueError) as exc_info:
        validate_adjudication_answer_json(json.dumps(payload), set())

    assert str(exc_info.value) == expected


def test_validate_adjudication_answer_json_requires_valid_source_status():
    payload = {
        "conclusion": "Test.",
        "source_status": {
            "faq": "claimed_by_model",
            "rules": "not_searched",
            "failure_reason": "",
        },
        "answers": [{"question": "Q"}],
        "evidence_use": [],
        "uncertainties": [],
    }

    with pytest.raises(ValueError, match="source_status.faq"):
        validate_adjudication_answer_json(json.dumps(payload), set())


def test_source_status_normalization_overrides_model_claims_with_actual_results():
    answer = json.dumps(
        {
            "conclusion": "Test.",
            "source_status": {
                "faq": "not_found",
                "rules": "found",
                "failure_reason": "模型自行声称的状态",
            },
            "answers": [{"question": "Q"}],
            "evidence_use": [],
            "uncertainties": [],
        },
        ensure_ascii=False,
    )
    evidence = [
        EvidenceItem(
            evidence_id="ygocdb:faq:1",
            kind="faq",
            source="ygocdb",
            text="FAQ evidence.",
            relationship="direct_card",
        )
    ]

    normalized = json.loads(
        normalize_adjudication_answer_source_status(
            answer,
            evidence,
            faq_searched=True,
            rules_searched=True,
            sufficient=True,
        )
    )

    assert normalized["source_status"] == {
        "faq": "found",
        "rules": "not_found",
        "failure_reason": "已依据 FAQ 证据完成裁定，但未找到对应的规则文档机制。",
    }
