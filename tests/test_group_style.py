import asyncio
import json
from types import SimpleNamespace

import pytest
from rag_agent import group_style as style
from rag_agent import group_chat as chat
from rag_agent.config import Settings
from rag_agent.web import create_app
from test_web import call_asgi_app


def payload():
    return dict(group_id=819151070,previous_summary='',style_samples=['短句一','短句二','短句三'],
                statistics=dict(message_count=200,participant_count=3,mean_length=8,short_message_ratio=1))


@pytest.mark.parametrize('field,value',[('group_id',123),('style_samples',['x']*9),('previous_summary','x'*601),('api_key','secret')])
def test_reject_invalid_or_out_of_scope_requests(field,value):
    data=payload(); data[field]=value
    with pytest.raises(ValueError): style.parse_style_request(data)


@pytest.mark.parametrize('key,value',[('message_count',199),('participant_count',2),('mean_length',float('nan')),('short_message_ratio',2)])
def test_reject_insufficient_or_invalid_aggregate_statistics(key,value):
    data=payload(); data['statistics'][key]=value
    with pytest.raises(ValueError): style.parse_style_request(data)


def test_old_profile_is_data_and_reply_uses_profile(monkeypatch):
    captured={}
    class Model:
        def bind(self,**kwargs): return self
        def invoke(self,messages):
            captured['messages']=messages
            return SimpleNamespace(content='{"summary":"本群偏好短句，语气随意，以自嘲和顺着上句接梗为主。"}')
    monkeypatch.setattr(style,'create_deepseek_chat_model',lambda settings:Model())
    result=style.execute_style_summary(style.parse_style_request(payload()),Settings.from_env({}))
    assert '短句' in result['summary']
    assert captured['messages'][0]['content']==style.PROMPT
    assert len(json.loads(captured['messages'][1]['content'])['anonymous_examples'])==3
    request=chat.parse_group_chat_request(dict(group_id=819151070,messages=[dict(speaker='成员1',text='今天又寄了')],style_profile='短句自嘲。'))
    assert request.style_profile=='短句自嘲。'
    fake=SimpleNamespace(invoke=lambda messages:(captured.update(reply_messages=messages) or SimpleNamespace(content='{"reply":"主打一个参与"}')))
    monkeypatch.setattr(chat,'create_deepseek_chat_model',lambda settings:SimpleNamespace(bind=lambda **kwargs:fake))
    chat.execute_group_chat(request,Settings.from_env({}))
    assert json.loads(captured['reply_messages'][1]['content'])['group_style_description']=='短句自嘲。'


@pytest.mark.parametrize('summary',['','@成员','账号123456789','https://example.com','x'*601])
def test_bad_summary_not_persistable(monkeypatch,summary):
    fake=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content=json.dumps({'summary':summary})))
    monkeypatch.setattr(style,'create_deepseek_chat_model',lambda settings:SimpleNamespace(bind=lambda **kwargs:fake))
    with pytest.raises(RuntimeError): style.execute_style_summary(style.parse_style_request(payload()),Settings.from_env({}))


def test_summary_endpoint_and_old_chat_clients():
    app=create_app(group_style_handler=lambda request:{'summary':'本群习惯短句口语。'})
    response=asyncio.run(call_asgi_app(app,'POST','/api/group-style',payload()))
    assert response['status']==200 and '短句' in json.loads(response['body'])['summary']
    data=payload(); data['group_id']=123
    assert asyncio.run(call_asgi_app(app,'POST','/api/group-style',data))['status']==400
    assert chat.parse_group_chat_request(dict(group_id=819151070,messages=[dict(speaker='成员1',text='你好')])).style_profile==''
