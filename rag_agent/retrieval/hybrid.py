from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    card_id: int
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[Candidate]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[Candidate]:
    scores: dict[int, float] = {}
    tie_breakers: dict[int, float] = {}
    metadata_by_id: dict[int, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, candidate in enumerate(ranked_list, start=1):
            scores[candidate.card_id] = scores.get(candidate.card_id, 0.0) + 1.0 / (
                k + rank
            )
            tie_breakers[candidate.card_id] = max(
                tie_breakers.get(candidate.card_id, float("-inf")),
                math.log1p(max(candidate.score, 0.0)),
            )
            metadata_by_id.setdefault(candidate.card_id, candidate.metadata)

    fused = [
        Candidate(
            card_id=card_id,
            score=score,
            source="hybrid",
            metadata=metadata_by_id.get(card_id, {}),
        )
        for card_id, score in scores.items()
    ]
    fused.sort(
        key=lambda candidate: (
            -candidate.score,
            -tie_breakers.get(candidate.card_id, 0.0),
            candidate.card_id,
        )
    )
    if top_k is not None:
        return fused[:top_k]
    return fused
