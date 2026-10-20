import json
from types import SimpleNamespace

import pytest
from rag_agent import group_chat as chat
from rag_agent import group_persona as persona
from rag_agent.config import Settings


def payload():
    return dict(group_id=819151070,messages=[dict(speaker='成员1',text='抽牌又寄了')],style_profile='短句轻吐槽',
        expressions=[dict(situation='抽卡不顺',expression='这下寄了')],meme_categories=[dict(category='sigh',description='无奈'),dict(category='happy',description='开心')])


@pytest.mark.parametrize('categories',[[dict(category='../bad',description='错误')],[dict(category='happy',description='开心')]*2,[dict(category='happy',description='x'*161)],[dict(category='happy',description='开心')]*33])
def test_category_request_boundaries(categories):
    data=payload();data['meme_categories']=categories
    with pytest.raises(ValueError):chat.parse_group_chat_request(data)


def fake_model(monkeypatch,tmp_path,response):
    monkeypatch.setenv('GROUP_CHAT_BUDGET_DB',str(tmp_path/'budget.db'))
    capture={}
    class Model:
        def bind(self,**kw):capture['options']=kw;return self
        def invoke(self,messages):
            capture['messages']=messages
            return SimpleNamespace(content=json.dumps(response),usage_metadata=dict(input_tokens=100,output_tokens=20))
    monkeypatch.setattr(chat,'create_deepseek_chat_model',lambda settings:Model())
    return capture


def test_persona_verbatim_and_learning_independent_in_one_call(tmp_path,monkeypatch):
    text='原文角色设定\n保留 ** 和 **** 占位符\n不要替换文字。\n'
    path=tmp_path/'persona.txt';path.write_text(text,encoding='utf-8')
    monkeypatch.setenv('GROUP_CHAT_PERSONA_FILE',str(path))
    captured=fake_model(monkeypatch,tmp_path,dict(reply='这下寄了',meme_category='sigh'))
    result=chat.execute_group_chat(chat.parse_group_chat_request(payload()),Settings.from_env({}))
    assert result==dict(reply='这下寄了',meme_category='sigh')
    assert text in captured['messages'][0]['content'] and path.read_text(encoding='utf-8')==text
    data=json.loads(captured['messages'][1]['content'])
    assert data['group_style_description']=='短句轻吐槽' and data['situational_expressions']
    assert captured['options']['extra_body']['thinking']['type']=='disabled'
    assert 'tools' not in captured['options']


@pytest.mark.parametrize('reply,category,expected',[('','sigh','sigh'),('短句','../sigh',None),('短句',['sigh'],None),('@成员','sigh',None),('https://bad.example','sigh',None)])
def test_output_category_allowlist_and_unsafe_suppression(tmp_path,monkeypatch,reply,category,expected):
    fake_model(monkeypatch,tmp_path,dict(reply=reply,meme_category=category))
    result=chat.execute_group_chat(chat.parse_group_chat_request(payload()),Settings.from_env({}))
    assert result['meme_category']==expected
    if reply.startswith(('@','http')):assert result['reply']==''


def test_legacy_missing_default_and_explicit_disable(tmp_path,monkeypatch):
    monkeypatch.delenv('GROUP_CHAT_PERSONA_FILE',raising=False)
    monkeypatch.setattr(persona,'DEFAULT_PERSONA',tmp_path/'missing')
    assert persona.load_persona()==''
    fake_model(monkeypatch,tmp_path,dict(reply='旧客户端',meme_category='happy'))
    data=payload();data.pop('meme_categories')
    assert chat.execute_group_chat(chat.parse_group_chat_request(data),Settings.from_env({}))==dict(reply='旧客户端')
    monkeypatch.setenv('GROUP_CHAT_PERSONA_FILE','')
    assert persona.load_persona()==''


def test_invalid_explicit_persona_and_size_limit(tmp_path,monkeypatch):
    path=tmp_path/'persona.txt';monkeypatch.setenv('GROUP_CHAT_PERSONA_FILE',str(path))
    with pytest.raises(RuntimeError):persona.load_persona()
    path.write_text('x'*20001,encoding='utf-8')
    with pytest.raises(RuntimeError):persona.load_persona()
