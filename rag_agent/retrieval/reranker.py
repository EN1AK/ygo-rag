from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
from typing import Sequence

from rag_agent.retrieval.hybrid import Candidate


def _normalize_rerank_text(value: object) -> str:
    text = str(value or "")
    return " ".join(text.replace("\x00", " ").split())


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
