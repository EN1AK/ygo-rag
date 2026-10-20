import asyncio,json
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest
from rag_agent import mention_chat as mention
from rag_agent import sessions
from rag_agent.sessions import SessionStore
from rag_agent.config import Settings
from rag_agent.assistant_agent import AssistantAgentRequest
from rag_agent.web import create_app
from test_web import call_asgi_app

KEY='qq:12345:group:819151070:user:111'


@pytest.fixture(autouse=True)
def stores(tmp_path,monkeypatch):
    monkeypatch.setattr(mention,'_casual_store',SessionStore(tmp_path/'casual.db',turns=8))
    monkeypatch.setattr(mention,'load_private_persona',lambda:'测试爱弥斯人格')
    monkeypatch.setattr(sessions,'_store',SessionStore(tmp_path/'pro.db'))
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))


def request(text='今天又抽不到想要的卡',key=KEY,group=819151070,**extra):
    return mention.parse_mention_request(dict(session_id=key,group_id=group,text=text,**extra))


def invoke_fixture(monkeypatch,data,capture=None):
    def invoke(factory,settings,route,messages,**options):
        if capture is not None:capture.append(dict(route=route,messages=messages,options=options))
        return SimpleNamespace(content=json.dumps(data))
    monkeypatch.setattr(mention,'invoke_group_model',invoke)


@pytest.mark.parametrize('text',['翻译 hello','裁定 这张卡能连锁发动吗','符文战场规则：攻击怎么结算','查卡 黑魔术师'])
def test_explicit_specialist_never_loads_persona_or_classifier(monkeypatch,text):
    def forbidden(*args,**kwargs):raise AssertionError('Persona/classifier must not run')
    monkeypatch.setattr(mention,'load_persona',forbidden)
    monkeypatch.setattr(mention,'invoke_group_model',forbidden)
    assert mention.execute_route(mention.parse_route_request(dict(text=text)),Settings.from_env({}))==dict(route='professional')
    assert mention.execute_mention(request(text),Settings.from_env({}))==dict(route='professional')


@pytest.mark.parametrize('text,expected',[
    ('ck 黑魔术师','professional'),('/ck黑魔术师','professional'),('裁定 这张卡能连锁发动吗','professional'),
    ('翻译 hello','professional'),('translate hello','professional'),('查卡 青眼白龙','professional'),
    ('规则 攻击怎么结算','professional'),('符文战场规则：攻击怎么结算','professional'),
    ('rbpl test','professional'),('pl test','professional'),('卡价 黑魔术师','professional'),
    ('原神和鸣潮哪个nb','casual'),('这张卡能连锁发动吗','casual'),('我要发动魔法','casual'),
    ('https://example.com','casual'),('12345678','casual'),('cking','casual'),
    ('帮我翻译 hello','casual'),('比较两款游戏性能和付费成本','casual'),('那接下来呢','casual'),
])
def test_fixed_command_routes_without_model_or_history(monkeypatch,text,expected):
    def forbidden(*args,**kwargs): raise AssertionError('No classifier or persona in routing')
    monkeypatch.setattr(mention,'invoke_group_model',forbidden)
    monkeypatch.setattr(mention,'load_persona',forbidden)
    req=mention.parse_route_request(dict(text=text,previous_route='professional',quoted_route='professional'))
    assert mention.execute_route(req,Settings.from_env({}))==dict(route=expected)


def test_persona_reads_only_same_user_casual_history_and_preserves_original(monkeypatch):
    capture=[];original='原样人格 **\n完整保留。'
    monkeypatch.setattr(mention,'load_persona',lambda:original)
    invoke_fixture(monkeypatch,dict(reply='下把一定',meme_category=None),capture)
    neutral=SimpleNamespace(answer='专业机密上下文',tool_results=())
    sessions._store.save(KEY,'旧混合记录',neutral)
    sessions._store.save(KEY+':professional','专业问答',neutral)
    mention._casual_store.save(KEY+':persona','之前的闲聊',SimpleNamespace(answer='之前的人格回复',tool_results=()))
    mention._casual_store.save(KEY.replace('111','222')+':persona','别人的闲聊',neutral)
    assert mention.execute_mention(request(),Settings.from_env({}))['route']=='casual'
    messages=capture[0]['messages'];encoded=json.dumps(messages,ensure_ascii=False)
    assert original in messages[0]['content']
    assert '之前的闲聊' in encoded and '专业机密上下文' not in encoded and '别人的闲聊' not in encoded
    assert len(mention._casual_store.history(KEY+':persona'))==2
    assert len(sessions._store.history(KEY+':professional'))==1


def test_professional_execution_cannot_read_legacy_or_casual_turns(monkeypatch):
    from rag_agent import assistant_agent
    sessions._store.save(KEY,'旧混合人格',SimpleNamespace(answer='旧人格内容',tool_results=()))
    mention._casual_store.save(KEY+':persona','人格闲聊',SimpleNamespace(answer='我是小鲸鱼',tool_results=()))
    seen=[]
    def professional(req,settings):
        seen.append(req)
        return SimpleNamespace(answer='严谨专业结果 [来源1]',tool_results=())
    monkeypatch.setattr(assistant_agent,'execute_assistant_agent',professional)
    base=AssistantAgentRequest(text='专业问题',db_path=Path('unused.cdb'),session_id=KEY+':professional')
    sessions.execute_session(base,Settings.from_env({}))
    assert seen[0].history==()
    sessions.execute_session(replace(base,text='继续专业问题'),Settings.from_env({}))
    assert seen[1].history[0]['assistant']=='严谨专业结果 [来源1]'
    assert '小鲸鱼' not in json.dumps(seen[1].history,ensure_ascii=False)


def test_persona_cannot_override_fixed_route(monkeypatch):
    invoke_fixture(monkeypatch,dict(reply='人格中间草稿',needs_professional=True))
    assert mention.execute_mention(request(),Settings.from_env({}))['route']=='casual'
    assert len(mention._casual_store.history(KEY+':persona'))==1


def test_other_groups_cannot_read_allowed_group_style(monkeypatch):
    other=KEY.replace('819151070','67890')
    with pytest.raises(ValueError):request(key=other,group=67890,style_profile='其他群的风格')
    with pytest.raises(ValueError):request(key=other,group=67890,expressions=[dict(situation='场景',expression='说法')])
    invoke_fixture(monkeypatch,dict(reply='你好呀'))
    assert mention.execute_mention(request(key=other,group=67890),Settings.from_env({}))['route']=='casual'


@pytest.mark.parametrize('extra',[dict(session_id='invalid'),dict(group_id=123),dict(api_key='secret')])
def test_scoped_request_validation(extra):
    data=dict(session_id=KEY,group_id=819151070,text='你好');data.update(extra)
    with pytest.raises(ValueError):mention.parse_mention_request(data)


def test_clear_removes_all_three_histories_without_model(monkeypatch):
    answer=SimpleNamespace(answer='内容',tool_results=())
    for key in [KEY,KEY+':professional','other']:sessions._store.save(key,'内容',answer)
    for key in [KEY+':persona','other']:mention._casual_store.save(key,'内容',answer)
    monkeypatch.setattr(mention,'invoke_group_model',lambda *a,**k:(_ for _ in ()).throw(AssertionError()))
    assert mention.clear_mention(KEY)==dict(cleared=True)
    assert not sessions._store.history(KEY) and not sessions._store.history(KEY+':professional')
    assert not mention._casual_store.history(KEY+':persona')
    assert sessions._store.history('other') and mention._casual_store.history('other')


def test_clear_waits_for_inflight_persona_then_deletes_it(monkeypatch):
    started,release=threading.Event(),threading.Event()
    def invoke(*args,**kwargs):
        started.set();assert release.wait(5)
        return SimpleNamespace(content='{"reply":"旧回复"}')
    monkeypatch.setattr(mention,'invoke_group_model',invoke)
    with ThreadPoolExecutor(max_workers=2) as pool:
        generation=pool.submit(mention.execute_mention,request(),Settings.from_env({}))
        assert started.wait(5)
        clear=pool.submit(mention.clear_mention,KEY)
        assert not clear.done()
        release.set();generation.result();clear.result()
    assert mention._casual_store.history(KEY+':persona')==[]


def test_http_routes_and_credential_rejection():
    app=create_app(mention_route_handler=lambda req:dict(route='casual'),mention_chat_handler=lambda req:dict(route='casual',reply='你好'),mention_clear_handler=lambda key:dict(cleared=True))
    for path,payload in [('/api/mention-route',dict(text='你好')),('/api/mention-chat',dict(text='你好',session_id=KEY,group_id=819151070)),('/api/mention-clear',dict(session_id=KEY))]:
        assert asyncio.run(call_asgi_app(app,'POST',path,payload))['status']==200
        payload['api_key']='secret'
        assert asyncio.run(call_asgi_app(app,'POST',path,payload))['status']==400


@pytest.mark.parametrize('size',[101,800,2000])
def test_long_mention_reply_keeps_full_history_and_original_persona(monkeypatch,size):
    capture=[];original='完整原人格。'
    monkeypatch.setattr(mention,'load_persona',lambda:original)
    invoke_fixture(monkeypatch,dict(reply='长'*size,meme_category=None),capture)
    result=mention.execute_mention(request('请展开聊聊'),Settings.from_env({}))
    assert result['reply']=='长'*size
    assert mention._casual_store.history(KEY+':persona')[-1]['assistant']=='长'*size
    assert original in capture[0]['messages'][0]['content']
    assert '不强制一句话' in capture[0]['messages'][0]['content']
    assert capture[0]['options']['max_tokens']==1800


def test_oversize_mention_response_never_saved(monkeypatch):
    invoke_fixture(monkeypatch,dict(reply='长'*2001,meme_category=None))
    with pytest.raises(RuntimeError):mention.execute_mention(request(),Settings.from_env({}))
    assert mention._casual_store.history(KEY+':persona')==[]


PRIVATE_KEY='qq:12345:private:user:111'


def test_private_session_does_not_receive_group_or_other_user_history(monkeypatch):
    capture=[];invoke_fixture(monkeypatch,dict(reply='私聊回复'),capture)
    neutral=SimpleNamespace(answer='群聊内容',tool_results=())
    mention._casual_store.save(KEY+':persona','群聊问题',neutral)
    mention._casual_store.save(PRIVATE_KEY.replace('111','222')+':persona','其他用户',neutral)
    req=mention.parse_mention_request(dict(session_id=PRIVATE_KEY,text='你好',group_id=None))
    assert mention.execute_mention(req,Settings.from_env({}))['route']=='casual'
    data=json.loads(capture[0]['messages'][1]['content'])
    assert data['casual_history']==[] and data['group_style_description']=='' and data['situational_expressions']==[]


@pytest.mark.parametrize('extra',[dict(group_id=819151070),dict(style_profile='群风格'),dict(expressions=[dict(situation='场景',expression='表达')])])
def test_private_session_cannot_claim_group_learning(extra):
    with pytest.raises(ValueError):mention.parse_mention_request(dict(session_id=PRIVATE_KEY,text='你好',**extra))


def test_clear_private_session_preserves_group_sessions():
    response=SimpleNamespace(answer='回复',tool_results=())
    for key in [PRIVATE_KEY,KEY]:
        mention._casual_store.save(key+':persona','闲聊',response)
        sessions._store.save(key+':professional','专业',response)
    mention.clear_mention(mention.session_base(dict(session_id=PRIVATE_KEY)))
    assert mention._casual_store.history(PRIVATE_KEY+':persona')==[]
    assert sessions._store.history(PRIVATE_KEY+':professional')==[]
    assert mention._casual_store.history(KEY+':persona')
    assert sessions._store.history(KEY+':professional')



def test_private_persona_is_aemeath_without_group_persona_or_legacy_private_turns(monkeypatch):
    capture=[]
    def forbidden():raise AssertionError('Private prompt cannot load group persona')
    monkeypatch.setattr(mention,'load_persona',forbidden)
    original='完整小爱原文\n包含原始换行。'
    monkeypatch.setattr(mention,'load_private_persona',lambda:original)
    response=SimpleNamespace(answer='旧大肥鱼回答',tool_results=())
    mention._casual_store.save(PRIVATE_KEY+':persona','旧人格问题',response)
    invoke_fixture(monkeypatch,dict(reply='小爱来了'),capture)
    req=mention.parse_mention_request(dict(session_id=PRIVATE_KEY,text='你好'))
    assert mention.execute_mention(req,Settings.from_env({}))['reply']=='小爱来了'
    assert original in capture[0]['messages'][0]['content']
    assert json.loads(capture[0]['messages'][1]['content'])['casual_history']==[]
    assert mention._casual_store.history(PRIVATE_KEY+':persona:aemeath')[-1]['assistant']=='小爱来了'
    assert mention._casual_store.history(PRIVATE_KEY+':persona')[-1]['assistant']=='旧大肥鱼回答'


def test_group_prompt_never_loads_private_persona(monkeypatch):
    def forbidden():raise AssertionError('Group cannot load private persona')
    monkeypatch.setattr(mention,'load_private_persona',forbidden)
    monkeypatch.setattr(mention,'load_persona',lambda:'用户定制的大肥鱼原文')
    capture=[];invoke_fixture(monkeypatch,dict(reply='群聊回复'),capture)
    mention.execute_mention(request(),Settings.from_env({}))
    assert '用户定制的大肥鱼原文' in capture[0]['messages'][0]['content']


def test_clear_removes_new_and_legacy_private_personas_only():
    response=SimpleNamespace(answer='回复',tool_results=())
    for key in [PRIVATE_KEY+':persona',PRIVATE_KEY+':persona:aemeath',KEY+':persona']:
        mention._casual_store.save(key,'问题',response)
    mention.clear_mention(PRIVATE_KEY)
    assert mention._casual_store.history(PRIVATE_KEY+':persona')==[]
    assert mention._casual_store.history(PRIVATE_KEY+':persona:aemeath')==[]
    assert mention._casual_store.history(KEY+':persona')
