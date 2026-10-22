"""Regression checks for indexed article evidence and summary-only degradation."""
import json
from types import SimpleNamespace
import pytest
from rag_agent import general_tools as general
from rag_agent import persona_tools as persona
from rag_agent.config import Settings

URL = "https://example.org/article"

def search(content=""):
    return {"results":[{"url":URL,"summary":"摘要","content":content}],
            "sources":[{"url":URL}]}

def model(actions):
    actions=iter(actions)
    return lambda messages:SimpleNamespace(content=json.dumps(next(actions),ensure_ascii=False))

def test_indexed_body_avoids_unnecessary_fetch(monkeypatch):
    monkeypatch.setattr(persona,"run_general_tool",lambda *args:search("索引正文"))
    invoke=model([{"tool":"web_search","arguments":{"query":"public topic"}},{"reply":"正文支持的回答","source_ids":[1]}])
    with persona.persona_deadline():
        reply,sources,used=persona.run_persona_turn(invoke,Settings.from_env({}),[])
    assert reply["reply"]=="正文支持的回答"
    assert used==["web_search"] and sources[0]["url"]==URL

def test_summary_only_preserves_limited_answer(monkeypatch):
    monkeypatch.setattr(persona,"run_general_tool",lambda *args:search())
    invoke=model([{"tool":"web_search","arguments":{"query":"topic"}},{"reply":"初步回答"},
                  {"reply":"该网站摘要提到有一份公告，具体细节待核实。"}])
    with persona.persona_deadline():
        reply,_,_=persona.run_persona_turn(invoke,Settings.from_env({}),[])
    assert reply["reply"].startswith("以下仅依据搜索摘要")
    assert "该网站摘要提到" in reply["reply"]

def test_failed_fetch_can_use_other_indexed_source(monkeypatch):
    def run(tool,*args):
        if tool=="read_webpage":raise ValueError("synthetic failure")
        return search("另一来源的正文")
    monkeypatch.setattr(persona,"run_general_tool",run)
    invoke=model([{"tool":"read_webpage","arguments":{"url":URL}},
                  {"tool":"web_search","arguments":{"query":"alternative"}},
                  {"reply":"根据另一来源可以确认的内容"}])
    with persona.persona_deadline():
        reply,_,used=persona.run_persona_turn(invoke,Settings.from_env({}),[])
    assert used==["read_webpage","web_search"]
    assert reply["reply"]=="根据另一来源可以确认的内容"

def test_search_body_budget_and_explicit_content_request(monkeypatch):
    calls=[]
    def api(settings,operation,payload):
        calls.append(payload)
        return {"results":[{"url":URL+str(i),"content":"x"*10000} for i in range(5)]}
    monkeypatch.setattr(general,"_api",api)
    result=general.run_general_tool("web_search",{"query":"topic","count":5},Settings.from_env({}))
    assert calls[0]["need_content"] is True
    assert sum(len(x["content"]) for x in result["results"])==12000
    assert all(len(x["content"])<=4000 for x in result["results"])
    assert result["results"][0]["content_source"]=="search_index"
    assert result["results"][0]["content_truncated"] is True

def test_compaction_bounds_indexed_content():
    messages=[{"role":"user","content":json.dumps({"tool_result_untrusted_data":{
        "evidence":{"results":[{"content":"x"*4000,"summary":"s"*1200}]}}})} for _ in range(4)]
    persona.compact_older_evidence(messages)
    oldest=json.loads(messages[0]["content"])["tool_result_untrusted_data"]["evidence"]["results"][0]
    assert len(oldest["content"])==1000 and oldest["content_truncated"]
    assert len(json.loads(messages[-1]["content"])["tool_result_untrusted_data"]["evidence"]["results"][0]["content"])==4000
