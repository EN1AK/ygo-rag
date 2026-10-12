"""Reuse compatible normalized BGE embeddings, serialize model inference."""
from functools import lru_cache
import threading
from rag_agent.retrieval.embeddings import BgeM3Embeddings
class SharedEmbeddings:
    def __init__(self,model,device):
        self.embedding=BgeM3Embeddings(model,device=device)
        self.lock=threading.Lock()
    def embed_documents(self,texts):
        with self.lock:return self.embedding.embed_documents(texts)
    def embed_query(self,text):return self.embed_documents([text])[0]
@lru_cache(maxsize=4)
def get_embeddings(model,device):return SharedEmbeddings(model,device)
