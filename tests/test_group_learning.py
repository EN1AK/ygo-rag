import asyncio,json
from types import SimpleNamespace

import pytest
from rag_agent import group_learning as learning
from rag_agent import group_chat as chat
from rag_agent.config import Settings
from rag_agent.web import create_app
from rag_agent.group_budget import GroupBudgetExceeded
from test_web import call_asgi_app


def payload():
    return dict(group_id=819151070,samples=[dict(source_id=f's{i}',text=t) for i,t in enumerate(
        ['抽卡又寄了','主打一个参与','我直接投降'],1)])


@pytest.mark.parametrize('key,value', [('group_id',123),('api_key','secret'),('samples',[]),('samples',[dict(source_id='s1',text='短句')]*3)])
def test_request_boundaries(key,value):
    data=payload();data[key]=value
    with pytest.raises(ValueError):learning.parse_learning_request(data)


def test_learning_rejects_hallucinations_and_cites_human_source(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    items=[dict(situation='自嘲抽卡',expression='又寄了',source_id='s1'),
           dict(situation='没发生的事',expression='我是群主',source_id='s1'),
           dict(situation='假来源',expression='投降',source_id='s9'),
           dict(situation='坏来源',expression='投降',source_id=[])]
    capture={}
    class Model:
        def bind(self,**kw):capture['opts']=kw;return self
        def invoke(self,messages):
            capture['messages']=messages
            return SimpleNamespace(content=json.dumps(dict(expressions=items)),usage_metadata=dict(input_tokens=100,output_tokens=100))
    monkeypatch.setattr(learning,'create_deepseek_chat_model',lambda settings:Model())
    result=learning.execute_learning(learning.parse_learning_request(payload()),Settings.from_env({}))
    assert result==dict(expressions=items[:1])
    assert capture['opts']['extra_body']['thinking']['type']=='disabled'
    assert 'tools' not in capture['opts']
    assert capture['messages'][0]['content']==learning.PROMPT


def test_combined_reference_cap_and_prompt_single_call(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    data=dict(group_id=819151070,messages=[dict(speaker='成员1',text='今天抽卡又寄了')],
        style_samples=['旧样本']*3,expressions=[dict(situation='自嘲抽卡运气',expression='又寄了')]*5)
    request=chat.parse_group_chat_request(data)
    calls=[]
    class Model:
        def bind(self,**kw):return self
        def invoke(self,messages):
            calls.append(messages)
            return SimpleNamespace(content='{"reply":""}')
    monkeypatch.setattr(chat,'create_deepseek_chat_model',lambda settings:Model())
    assert chat.execute_group_chat(request,Settings.from_env({}))==dict(reply='')
    assert len(calls)==1
    assert len(json.loads(calls[0][1]['content'])['situational_expressions'])==5
    data['style_samples'].append('多了一条')
    with pytest.raises(ValueError):chat.parse_group_chat_request(data)


def test_provider_excess_candidates_are_validated_and_bounded():
    items=[dict(situation='场景'+str(i),expression='主打一个参与',source_id='s1') for i in range(8)]
    items.insert(0,dict(situation='错误引用',expression='未出现的表达',source_id='s1'))
    assert learning.valid_expressions(items,{'s1':'主打一个参与'})==items[1:6]


def test_http_learning_status_and_budget_rejection(tmp_path,monkeypatch):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    app=create_app(group_learning_handler=lambda r:dict(expressions=[]))
    response=asyncio.run(call_asgi_app(app,'POST','/api/group-learning',payload()))
    assert response['status']==200 and json.loads(response['body'])==dict(expressions=[])
    response=asyncio.run(call_asgi_app(app,'GET','/api/group-chat/status'))
    assert response['status']==200 and json.loads(response['body'])['monthly_limit_cny']==10
    def exhausted(request):raise GroupBudgetExceeded()
    app=create_app(group_learning_handler=exhausted)
    assert asyncio.run(call_asgi_app(app,'POST','/api/group-learning',payload()))['status']==429
