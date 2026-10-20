import pytest
from rag_agent import private_persona as persona


def test_private_persona_reads_unmodified_utf8(tmp_path,monkeypatch):
    path=tmp_path/'persona.txt';original='小爱原文\n\n保留换行。\n'
    path.write_text(original,encoding='utf-8')
    monkeypatch.setenv('PRIVATE_CHAT_PERSONA_FILE',str(path))
    assert persona.load_private_persona()==original


@pytest.mark.parametrize('value',['missing','empty','oversize','disabled'])
def test_invalid_private_persona_never_falls_back_to_group(tmp_path,monkeypatch,value):
    path=tmp_path/'persona.txt'
    if value=='empty':path.write_text(' ',encoding='utf-8')
    if value=='oversize':path.write_text('长'*20001,encoding='utf-8')
    monkeypatch.setenv('PRIVATE_CHAT_PERSONA_FILE','' if value=='disabled' else str(path))
    with pytest.raises(RuntimeError):persona.load_private_persona()
