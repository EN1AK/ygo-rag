"""In-process adapters. No HTTP call or runtime mutation of global credentials."""
from functools import lru_cache
import threading
from rag_agent.deadline import remaining
from rag_agent.shared_embeddings import get_embeddings
_lock=threading.Lock()
_runtime=None
def runtime():
    global _runtime
    with _lock:
        if _runtime is None:
            from rift_rag.runtime import RiftConfig,RiftRuntime
            _runtime=RiftRuntime(RiftConfig.from_env(),get_embeddings,remaining)
        return _runtime
def query_rules(query,top_k=6,trace=False,mode=None):
    return runtime().query(query,top_k,trace,mode or "agent")
def search_cards(query,top_k=10):
    from rift_rag.cards import search_riftbound_cards,card_title
    cards=search_riftbound_cards(query,top_k)
    results=[]
    for card in cards:
        item={k:v for k,v in card.items() if k not in {"editions","image_path"}}
        item["title"]=card_title(card)
        item["editions"]=[c["card_key"] for c in card["editions"]]
        results.append(item)
    ambiguous=len(cards)>1 or bool(cards and cards[0]["match_kind"]=="fuzzy")
    return {"results":results,"ambiguous":ambiguous,"answer":"请从候选卡片中确认具体卡号。" if ambiguous else "查卡完成。" if cards else "未找到卡片。"}
