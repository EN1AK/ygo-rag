from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class BgeM3Embeddings:
    model_name: str
    device: str = "auto"
    backend: str = "auto"
    show_progress: bool = True
    _model: object | None = None
    _tokenizer: object | None = None

    def _load_model(self) -> object:
        if self._model is None:
            os.environ.setdefault("USE_TF", "0")
            os.environ.setdefault("USE_FLAX", "0")
            os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
            os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
            if self._resolved_backend() == "transformers":
                self._load_transformers_model()
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to load bge-m3. "
                    "Install dependencies with `python -m pip install -r requirements.txt`."
                ) from exc
            kwargs = {}
            if self.device != "auto":
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def _resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if self.device.startswith("cuda"):
            return "transformers"
        return "sentence-transformers"

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

    def _load_transformers_model(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for CUDA embedding backend. "
                "Install dependencies with `python -m pip install -r requirements.txt`."
            ) from exc

        model_path = self._resolve_model_path()
        local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=local_only,
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            local_files_only=local_only,
        )
        if self.device != "auto":
            self._model = self._model.to(self.device)
        self._model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        if self._resolved_backend() == "transformers":
            return self._embed_documents_with_transformers(texts)
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_documents_with_transformers(self, texts: list[str]) -> list[list[float]]:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:
            raise RuntimeError("torch is required for transformers embedding backend.") from exc

        assert self._tokenizer is not None
        assert self._model is not None
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=8192,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            vectors = outputs.last_hidden_state[:, 0]
            vectors = F.normalize(vectors, p=2, dim=1)
        return vectors.detach().cpu().float().tolist()
