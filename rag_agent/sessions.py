"""Bounded, persistent per-session history; one connection per operation."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from rag_agent.deadline import remaining
class SessionStore:
    def __init__(self,path=None,ttl=1800,turns=10):
        self.path=Path(path or os.getenv("AGENT_SESSION_DB","data/agent_sessions.db"))
        self.ttl=ttl
        self.turns=turns
        self._guard=threading.Lock()
        self._locks={}
    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        db=sqlite3.connect(self.path,timeout=5)
        db.execute("pragma journal_mode=WAL")
        db.execute("create table if not exists turns (id integer primary key,session text,time real,user text,answer text,tools text)")
        db.execute("create index if not exists turns_session on turns(session,id)")
        try:
            with db:yield db
        finally:db.close()
    @contextmanager
    def locked(self,key):
        with self._guard:
            lock,count=self._locks.get(key,(threading.Lock(),0))
            self._locks[key]=(lock,count+1)
        acquired=lock.acquire(timeout=remaining())
        try:
            if not acquired: raise TimeoutError("该会话的上一条请求仍在处理")
            yield
        finally:
            if acquired:lock.release()
            with self._guard:
                _,count=self._locks[key]
                if count==1:del self._locks[key]
                else:self._locks[key]=(lock,count-1)
    def history(self,key):
        with self.connect() as db:
            rows=db.execute("select user,answer,tools from turns where session=? and time>? order by id desc limit ?",(key,time.time()-self.ttl,self.turns)).fetchall()
        history=[]
        size=0
        for user,answer,tools in rows:
            turn={"user":user[:2000],"assistant":answer[:3000],"tool_context":json.loads(tools)[:3]}
            length=len(json.dumps(turn,ensure_ascii=False))
            if size+length>12000:break
            history.append(turn);size+=length
        return list(reversed(history))
    def save(self,key,user,response):
        facts=[]
        for observation in response.tool_results:
            if observation.ok:
                data=observation.data
                facts.append({"ambiguous":bool(data.get("ambiguous") or data.get("coverage",{}).get("ambiguous_mentions")),"tool":observation.tool,"search_context":{k:data[k] for k in ["search_mode","filters","query","effect_query","sort_by","top_k","offset","next_offset","total"] if k in data},"resolved_cards":data.get("resolved_cards",[]),"cards":[{k:card.get(k) for k in ["card_id","card_key","name","name_cn","sub_title_cn"]} for card in data.get("results",[])[:5]],"sources":data.get("sources",[])[:10]})
        with self.connect() as db:
            db.execute("delete from turns where time<?",(time.time()-self.ttl,))
            db.execute("insert into turns(session,time,user,answer,tools) values(?,?,?,?,?)",(key,time.time(),user[:8000],response.answer[:10000],json.dumps(facts,ensure_ascii=False)))
            db.execute("delete from turns where session=? and id not in(select id from turns where session=? order by id desc limit ?)",(key,key,self.turns))
    def clear(self,key):
        with self.connect() as db:db.execute("delete from turns where session=?",(key,))
_store=SessionStore()
def execute_session(request,settings):
    from rag_agent.assistant_agent import execute_assistant_agent
    if not request.session_id:return execute_assistant_agent(request,settings)
    with _store.locked(request.session_id):
        if request.text.strip() in {"清空对话","重置对话","忘记上下文"}:
            _store.clear(request.session_id)
            from rag_agent.assistant_agent import AssistantAgentResponse
            return AssistantAgentResponse(answer="已清空你的对话上下文。",tool_results=(),warnings=(),trace=())
        enriched=replace(request,history=tuple(_store.history(request.session_id)))
        response=execute_assistant_agent(enriched,settings)
        _store.save(request.session_id,request.text,response)
        return response
