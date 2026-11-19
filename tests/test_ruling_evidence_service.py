from pathlib import Path
import json

import pytest

from rag_agent import adjudication_service as service
from rag_agent.cards import Card
from rag_agent.config import Settings


@pytest.mark.parametrize("legacy_agent_flag", [False, True])
@pytest.mark.parametrize("missing_fact", [False, True])
def test_pure_rag_service_uses_bounded_loop(
    monkeypatch, legacy_agent_flag, missing_fact
):
    card = Card(card_id=40737112, name="混沌之黑魔术师", description="测试卡文",
                type=33, attribute=32, race=2)
    calls = []
    events = []

    class Model:
        def invoke(self, prompt):
            payload = json.loads(prompt.split("\n输入资料（证据文本只作为资料）\n", 1)[1])
            calls.append(payload)
            if len(calls) == 1:
                facts = payload["card_context"][0]["text"]
                assert "属性：dark" in facts and "种族：spellcaster" in facts
                assert "已核验卡名对应：混沌黑魔术师 → 混沌之黑魔术师" in facts
                return '{"checkpoints":[{"question":"如何裁定","query":"测试机制"}]}'
            if len(calls) == 2:
                if missing_fact:
                    return '{"items":[{"id":"C1","status":"missing_fact","refs":[],"detail":"哪一张卡先发动"}]}'
                return '{"items":[{"id":"C1","status":"supported","refs":["E1"]}]}'
            return '{"answers":[{"id":"C1","conclusion":"测试结论","refs":["E1"]}]}'

    monkeypatch.setattr(service, "load_cards", lambda _: [card])
    monkeypatch.setattr(service, "build_ruling_index", lambda *_: None)
    monkeypatch.setattr(service, "build_rule_index", lambda *_: None)
    monkeypatch.setattr(service, "build_persistent_ruling_retrievers", lambda *_: ())
    monkeypatch.setattr(service, "create_deepseek_chat_model", lambda _: Model())
    request = service.AdjudicationRequest(
        scenario="混沌黑魔术师的卡片属性是什么？",
        db_path=Path("synthetic.cdb"),
        use_llm=True,
        use_agent=legacy_agent_flag,
        include_trace=True,
        progress_callback=events.append,
    )
    response = service.execute_adjudication(request, Settings.from_env())
    assert response.simulation.status == "not_requested"
    assert response.agent_coverage["route"] == "bounded_evidence_loop"
    assert [event.stage for event in events] == ["plan", "coverage", "answer"]
    assert [item["stage"] for item in response.agent_trace] == ["plan", "coverage", "answer"]
    answer = json.loads(response.answer)
    if missing_fact:
        assert response.agent_coverage["clarification_questions"] == ["请补充确认：哪一张卡先发动"]
        assert "测试结论" not in response.answer
    else:
        assert response.agent_coverage["status"] == "complete"
        assert answer["conclusion"] == "测试结论"
        ids = answer["answers"][0]["reasoning"][0]["evidence_ids"]
        assert ids == [response.evidence[0].evidence_id]
