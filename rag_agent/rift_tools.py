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
def search_cards(query="",top_k=10,*,filters=None,effect_query="",offset=0,sort_by="card_key"):
    from rift_rag.card_search import search_catalog
    from rift_rag.cards import card_title
    remaining()
    data=search_catalog(query,top_k,filters=filters,effect_query=effect_query,offset=offset,sort_by=sort_by,embedding_provider=get_embeddings)
    results=[]
    for card in data["results"]:
        item={k:v for k,v in card.items() if k not in {"editions","image_path"}}
        item["title"]=card_title(card)
        item["editions"]=[c["card_key"] for c in card["editions"]]
        results.append(item)
    return dict(data,results=results)
