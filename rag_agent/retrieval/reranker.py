from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
from typing import Callable, Sequence

from rag_agent.retrieval.hybrid import Candidate


def _normalize_rerank_text(value: object) -> str:
    text = str(value or "")
    return " ".join(text.replace("\x00", " ").split())


@dataclass(frozen=True)
class LlmJudgeScore:
    card_id: int
    score: float
    reason: str


def build_llm_rerank_prompt(query: str, candidates: Sequence[Candidate]) -> str:
    candidate_payload = []
    for candidate in candidates:
        metadata = candidate.metadata
        candidate_payload.append(
            {
                "card_id": candidate.card_id,
                "name": str(metadata.get("name", "")),
                "retrieval_score": f"{candidate.score:.4f}",
                "source_text": _normalize_rerank_text(
                    metadata.get("desc") or metadata.get("text") or ""
                ),
            }
        )
    return (
        "你是游戏王卡片效果相似度 judge。"
        "只允许评价候选列表中的卡，不要引入候选外卡片。\n"
        "请优先判断主要效果意图是否相似，例如保护对象、发动条件、作用对象、效果结果；"
        "不要只因为费用、数字或“无效并破坏”等表面关键词相同就给高分。\n"
        "请只输出 JSON，不要输出 Markdown。格式："
        '{"results":[{"card_id":123,"score":0.0到1.0之间的数字,"reason":"简短中文理由"}]}'
        "\n\n"
        f"用户问题：{query}\n\n"
        "候选卡 JSON：\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}"
    )


def parse_llm_rerank_response(
    content: object,
    *,
    allowed_ids: set[int],
) -> list[LlmJudgeScore]:
    text = _extract_response_text(content)
    try:
        payload = json.loads(_strip_json_fence(text))
    except Exception as exc:
        raise RuntimeError("Unable to parse LLM rerank response as JSON.") from exc

    if isinstance(payload, list):
        raw_results = payload
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        raw_results = payload["results"]
    else:
        raise RuntimeError("Unable to parse LLM rerank response: expected results list.")

    parsed: list[LlmJudgeScore] = []
    seen: set[int] = set()
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        try:
            card_id = int(item["card_id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if card_id not in allowed_ids or card_id in seen:
            continue
        reason = _normalize_rerank_text(item.get("reason", ""))
        parsed.append(
            LlmJudgeScore(
                card_id=card_id,
                score=score,
                reason=reason or "LLM judge did not provide a reason.",
            )
        )
        seen.add(card_id)
    return parsed


def _extract_response_text(content: object) -> str:
    value = getattr(content, "content", content)
    return str(value)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


@dataclass
class LlmReranker:
    llm: object
    max_candidates: int = 20
    warning_callback: Callable[[str], None] | None = None

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        limited = list(candidates[: max(self.max_candidates, 0)])
        overflow = list(candidates[len(limited) :])
        if not limited:
            return list(candidates)

        prompt = build_llm_rerank_prompt(query, limited)
        try:
            response = self.llm.invoke(prompt)
            scores = parse_llm_rerank_response(
                response,
                allowed_ids={candidate.card_id for candidate in limited},
            )
        except Exception as exc:
            self._warn(
                "LLM rerank failed; falling back to hybrid candidate order. "
                f"{exc}"
            )
            return list(candidates)
        score_by_id = {score.card_id: score for score in scores}
        original_position = {
            candidate.card_id: index for index, candidate in enumerate(candidates)
        }

        scored: list[Candidate] = []
        unscored: list[Candidate] = []
        for candidate in limited:
            judge_score = score_by_id.get(candidate.card_id)
            if judge_score is None:
                unscored.append(self._with_reason(candidate, candidate.score, "LLM judge did not score this candidate."))
                continue
            scored.append(
                self._with_reason(candidate, judge_score.score, judge_score.reason)
            )

        scored.sort(key=lambda candidate: (-candidate.score, candidate.card_id))
        unscored.extend(
            self._with_reason(candidate, candidate.score, "Candidate was not sent to LLM judge due to max candidate limit.")
            for candidate in overflow
        )
        unscored.sort(key=lambda candidate: original_position[candidate.card_id])
        return scored + unscored

    def _with_reason(self, candidate: Candidate, score: float, reason: str) -> Candidate:
        return Candidate(
            card_id=candidate.card_id,
            score=float(score),
            source="llm_reranker",
            metadata=candidate.metadata | {"llm_judge_reason": reason},
        )

    def _warn(self, message: str) -> None:
        if self.warning_callback is not None:
            self.warning_callback(message)


@dataclass
class BgeReranker:
    model_name: str
    device: str = "auto"
    _model: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "FlagEmbedding is required to load bge-reranker-v2-m3. "
                    "Install dependencies with `python -m pip install -r requirements.txt`."
                ) from exc
            kwargs = {}
            if self.device != "auto":
                kwargs["devices"] = self.device
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.device.startswith("cuda"),
                **kwargs,
            )
        return self._model

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        model = self._load_model()
        normalized_query = _normalize_rerank_text(query)
        pairs = [
            (
                normalized_query,
                _normalize_rerank_text(
                    candidate.metadata.get("desc") or candidate.metadata.get("text") or ""
                ),
            )
            for candidate in candidates
        ]
        scores = model.compute_score(pairs)
        if isinstance(scores, float):
            scores = [scores]
        reranked = [
            Candidate(
                card_id=candidate.card_id,
                score=float(score),
                source="reranker",
                metadata=candidate.metadata,
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda candidate: (-candidate.score, candidate.card_id))
        return reranked


@dataclass
class SubprocessReranker:
    model_name: str
    device: str = "auto"
    python_executable: str = sys.executable
    timeout_seconds: int = 300

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        payload = {
            "model_name": self.model_name,
            "device": self.device,
            "query": query,
            "candidates": [
                {
                    "card_id": candidate.card_id,
                    "score": candidate.score,
                    "source": candidate.source,
                    "metadata": candidate.metadata,
                    "text": str(
                        candidate.metadata.get("desc")
                        or candidate.metadata.get("text")
                        or ""
                    ),
                }
                for candidate in candidates
            ],
        }
        env = os.environ.copy()
        env["RAG_DEVICE"] = self.device
        env["RAG_RERANKER_DEVICE"] = self.device
        env["PYTHONIOENCODING"] = "utf-8"
        env.setdefault("USE_TF", "0")
        env.setdefault("USE_FLAX", "0")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        env.setdefault("TRANSFORMERS_NO_FLAX", "1")
        completed = subprocess.run(
            [self.python_executable, "-m", "rag_agent.rerank_worker"],
            input=json.dumps(payload, ensure_ascii=True),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "reranker subprocess failed: "
                + ((completed.stderr or "").strip() or (completed.stdout or "").strip())
            )
        result = json.loads(completed.stdout)
        scores = result["scores"]
        reranked = [
            Candidate(
                card_id=candidate.card_id,
                score=float(score),
                source="reranker",
                metadata=candidate.metadata,
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda candidate: (-candidate.score, candidate.card_id))
        return reranked


@dataclass
class TransformersReranker:
    model_name: str
    device: str = "auto"
    _model: object | None = None
    _tokenizer: object | None = None

    def _resolve_model_path(self) -> str:
        if os.path.exists(self.model_name):
            return self.model_name
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return self.model_name
        local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
        try:
            return snapshot_download(self.model_name, local_files_only=local_only)
        except Exception:
            return self.model_name

    def _load_model(self) -> object:
        if self._model is None:
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is required for subprocess reranking. "
                    "Install dependencies with `python -m pip install -r requirements.txt`."
                ) from exc
            model_path = self._resolve_model_path()
            local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_only,
                use_fast=False,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=local_only,
            )
            if self.device != "auto":
                self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def rerank(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for transformers reranking.") from exc

        model = self._load_model()
        assert self._tokenizer is not None
        normalized_query = _normalize_rerank_text(query)
        device = next(model.parameters()).device
        scores: list[float] = []
        with torch.no_grad():
            for candidate in candidates:
                passage = _normalize_rerank_text(
                    candidate.metadata.get("desc") or candidate.metadata.get("text") or ""
                )
                try:
                    inputs = self._tokenizer(
                        normalized_query,
                        passage,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                except Exception as exc:
                    raise TypeError(
                        "Failed to tokenize rerank pair "
                        f"card_id={candidate.card_id} "
                        f"query_type={type(normalized_query)!r} "
                        f"passage_type={type(passage)!r} "
                        f"query_preview={normalized_query[:80]!r} "
                        f"passage_preview={passage[:120]!r}"
                    ) from exc
                inputs = {key: value.to(device) for key, value in inputs.items()}
                score = model(**inputs).logits.view(-1).float().detach().cpu().item()
                scores.append(float(score))
        reranked = [
            Candidate(
                card_id=candidate.card_id,
                score=float(score),
                source="reranker",
                metadata=candidate.metadata,
            )
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda candidate: (-candidate.score, candidate.card_id))
        return reranked
