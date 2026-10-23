import asyncio,json,logging
from types import SimpleNamespace
import pytest
from rag_agent import persona_tools as tools
from rag_agent.web import create_app
from rag_agent.config import Settings
from test_web import call_asgi_app

def run(actions,messages=None):
    actions=iter(actions)
    def invoke(messages):
        result=next(actions)
        return SimpleNamespace(content=result if isinstance(result,str) else json.dumps(result))
    with tools.persona_deadline():
        return tools.run_persona_turn(invoke,Settings.from_env({}),messages or [])

@pytest.mark.parametrize("bad",["not-json",[],{"reply":None},{"reply":""},{"reply":"","meme_category":["invalid"]}])
def test_bad_model_output_repaired_once(bad):
    result,_,_=run([bad,{"reply":"有效回复"}])
    assert result["reply"]=="有效回复"

def test_repeated_empty_reply_has_clear_fallback():
    with pytest.raises(RuntimeError, match="Invalid mention chat response"):
        run([{"reply":""},{"reply":""}])

def test_valid_meme_only_reply_does_not_require_text():
    messages=[{"role":"user","content":json.dumps({"available_meme_categories":[{"category":"laugh"}]})}]
    result,_,_=run([{"reply":"","meme_category":"laugh"}],messages)
    assert result["meme_category"]=="laugh"

def test_http_error_classification_never_logs_response_body(caplog):
    def handler(request): raise RuntimeError("No safe persona reply")
    app=create_app(mention_chat_handler=handler)
    with caplog.at_level(logging.WARNING):
        response=asyncio.run(call_asgi_app(app,method="POST",path="/api/mention-chat",
            payload={"session_id":"qq:123:group:456:user:111","group_id":456,"text":"synthetic"}))
    assert response["status"]==500 and json.loads(response["body"])["error_code"]=="chat_reply_rejected"
    assert "code=chat_reply_rejected" in caplog.text
