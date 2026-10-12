import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from rag_agent.assistant_agent import AssistantAgent,AssistantAgentRequest,AssistantAgentResponse,AssistantToolObservation,parse_assistant_planner_action
from rag_agent.sessions import SessionStore
from rag_agent.http_runtime import HttpRuntime

class ScriptLlm:
    def __init__(self,actions):self.actions=iter(actions);self.prompts=[]
    def invoke(self,prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(content=json.dumps(next(self.actions),ensure_ascii=False))
def action(tool,args):return {"tool":tool,"arguments":args}
def request(**kwargs):return AssistantAgentRequest(text="你好",db_path=Path("missing.cdb"),**kwargs)

def test_chat_does_not_call_tools_and_reads_history():
    llm=ScriptLlm([action("finish",{"answer":"你好呀"})])
    runner=Mock()
    result=AssistantAgent(llm,runner).run(replace(request(),history=({"user":"阿狸","assistant":"OGN-066"},),quoted_reply="上一条回复"))
    runner.run.assert_not_called()
    assert result.answer=="你好呀" and "OGN-066" in llm.prompts[0] and "上一条回复" in llm.prompts[0]

def test_rift_tool_and_specialist_answer_preserved():
    llm=ScriptLlm([action("rift_rules",{"query":"符文战场 OGN-131","top_k":6}),action("finish",{"answer":"改写答案"})])
    runner=Mock()
    runner.run.return_value=AssistantToolObservation(tool="rift_rules",ok=True,summary="完成",data={"answer":"原始裁定 [R-CR-1]","sources":[{"rule_id":"R-CR-1"}],"coverage":{"unresolved_mentions":["未知卡"]}})
    result=AssistantAgent(llm,runner).run(request())
    assert result.answer=="原始裁定 [R-CR-1]"
    assert "未知卡" in llm.prompts[1] and "R-CR-1" in llm.prompts[1]

@pytest.mark.parametrize("tool",["rift_rules","rift_search_cards"])
def test_rift_arguments_are_strict(tool):
    with pytest.raises(ValueError):parse_assistant_planner_action(json.dumps(action(tool,{"query":"卡","top_k":True})))
    with pytest.raises(ValueError):parse_assistant_planner_action(json.dumps(action(tool,{"query":"卡","unknown":1})))

def test_session_isolation_expiry_bounds_and_clear(tmp_path):
    store=SessionStore(tmp_path/"sessions.db",ttl=30,turns=2)
    response=AssistantAgentResponse(answer="阿狸 OGN-066",tool_results=(),warnings=(),trace=())
    a="qq:1:group:1:user:1";b="qq:1:group:1:user:2";c="qq:1:group:2:user:1"
    for n in range(4):store.save(a,str(n),response)
    assert [item["user"] for item in store.history(a)]==["2","3"]
    assert store.history(b)==[] and store.history(c)==[]
    with store.connect() as db:db.execute("update turns set time=?",(time.time()-31,))
    assert store.history(a)==[]
    store.save(a,"新消息",response);store.clear(a)
    assert store.history(a)==[]

def test_same_session_serialized_and_other_session_independent(tmp_path):
    store=SessionStore(tmp_path/"sessions.db")
    entered=threading.Event();release=threading.Event();seen=[]
    def first():
        with store.locked("a"):entered.set();release.wait(2);seen.append("first")
    def second():
        with store.locked("a"):seen.append("second")
    with ThreadPoolExecutor(3) as pool:
        a=pool.submit(first);assert entered.wait(1)
        b=pool.submit(second)
        with store.locked("b"):seen.append("other")
        assert seen==["other"]
        release.set();a.result();b.result()
    assert seen==["other","first","second"]

def test_slow_http_request_does_not_block_fast_request():
    entered=threading.Event();release=threading.Event()
    def slow(request):entered.set();release.wait(3);return "slow"
    async def run():
        runtime=HttpRuntime()
        task=asyncio.create_task(runtime.call(slow,None))
        try:
            assert await asyncio.to_thread(entered.wait,1)
            assert await asyncio.wait_for(runtime.call(lambda request:"chat",None),0.5)=="chat"
        finally:release.set()
        assert await task=="slow"
        runtime.pool.shutdown()
    asyncio.run(run())


def test_timed_out_job_keeps_capacity_until_worker_exits(monkeypatch):
    monkeypatch.setenv("AGENT_WORKERS","1")
    monkeypatch.setenv("AGENT_REQUEST_TIMEOUT","0.05")
    entered=threading.Event();release=threading.Event();exited=threading.Event()
    def slow(request):
        entered.set();release.wait(2);exited.set();return "done"
    async def run():
        runtime=HttpRuntime()
        try:
            with pytest.raises(TimeoutError):await runtime.call(slow,None)
            assert entered.is_set() and runtime.slots.locked()
        finally:release.set()
        assert await asyncio.to_thread(exited.wait,1)
        await asyncio.sleep(0.02)
        assert not runtime.slots.locked()
        runtime.pool.shutdown()
    asyncio.run(run())
