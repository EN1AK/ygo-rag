from __future__ import annotations

import json
import sys

from rag_agent.retrieval.hybrid import Candidate
from rag_agent.retrieval.reranker import TransformersReranker


def main() -> int:
    payload = json.loads(sys.stdin.read())
    reranker = TransformersReranker(
        payload["model_name"],
        device=payload.get("device", "auto"),
    )
    candidates = [
        Candidate(
            card_id=int(item["card_id"]),
            score=float(item.get("score", 0.0)),
            source=str(item.get("source", "hybrid")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in payload.get("candidates", [])
    ]
    query = str(payload["query"])
    reranked = reranker.rerank(query, candidates)
    score_by_id = {candidate.card_id: candidate.score for candidate in reranked}
    scores = [score_by_id[candidate.card_id] for candidate in candidates]
    sys.stdout.write(json.dumps({"scores": scores}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
