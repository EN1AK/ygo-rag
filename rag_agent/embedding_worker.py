from __future__ import annotations

import json
import os
import sys
import traceback

from rag_agent.retrieval.embeddings import BgeM3Embeddings


def main() -> int:
    embeddings = BgeM3Embeddings(
        os.environ["RAG_EMBEDDING_MODEL"],
        device=os.environ.get("RAG_EMBEDDING_DEVICE", "cuda"),
        backend=os.environ.get("RAG_EMBEDDING_BACKEND", "transformers"),
        show_progress=False,
    )
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            operation = payload.get("operation")
            if operation == "query":
                response = {
                    "ok": True,
                    "vector": embeddings.embed_query(str(payload["text"])),
                }
            elif operation == "documents":
                response = {
                    "ok": True,
                    "vectors": embeddings.embed_documents(
                        [str(text) for text in payload.get("texts", [])]
                    ),
                }
            else:
                raise ValueError(f"unsupported embedding operation: {operation}")
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
