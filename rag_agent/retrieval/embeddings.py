from __future__ import annotations

from dataclasses import dataclass, field
import os
import json
import queue
import subprocess
import sys
import threading


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
        if os.name == "nt":
            # Import optional native dependencies serially before transformers lazily
            # reaches them through generation helpers. This avoids intermittent DLL
            # access violations observed during the first CUDA retrieval on Windows.
            try:
                import sklearn  # noqa: F401
            except ImportError:
                pass
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
        max_length = int(os.environ.get("RAG_EMBEDDING_MAX_LENGTH", "8192"))
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            vectors = outputs.last_hidden_state[:, 0]
            vectors = F.normalize(vectors, p=2, dim=1)
        return vectors.detach().cpu().float().tolist()


@dataclass
class SubprocessBgeM3Embeddings:
    """Keep CUDA/transformers native libraries outside the networked agent process."""

    model_name: str
    device: str = "cuda"
    backend: str = "transformers"
    show_progress: bool = False
    python_executable: str = sys.executable
    timeout_seconds: int = 300
    _process: subprocess.Popen[str] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        payload = self._request({"operation": "documents", "texts": texts})
        return [
            [float(value) for value in vector]
            for vector in payload["vectors"]
        ]

    def embed_query(self, text: str) -> list[float]:
        payload = self._request({"operation": "query", "text": text})
        return [float(value) for value in payload["vector"]]

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.close()
            process.wait(timeout=5)
        except Exception:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        with self._lock:
            process = self._ensure_process()
            assert process.stdin is not None
            assert process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            process.stdin.flush()
            response_queue: queue.Queue[str] = queue.Queue(maxsize=1)
            reader = threading.Thread(
                target=lambda: response_queue.put(process.stdout.readline()),
                daemon=True,
            )
            reader.start()
            try:
                raw_response = response_queue.get(timeout=self.timeout_seconds)
            except queue.Empty as exc:
                self.close()
                raise RuntimeError(
                    "embedding worker timed out after "
                    f"{self.timeout_seconds} seconds"
                ) from exc
            if not raw_response:
                return_code = process.poll()
                self.close()
                raise RuntimeError(
                    "embedding worker exited without a response"
                    + (
                        f" (return code {return_code})"
                        if return_code is not None
                        else ""
                    )
                )
            response = json.loads(raw_response)
            if not response.get("ok"):
                raise RuntimeError(
                    "embedding worker failed: "
                    + str(response.get("error") or "unknown error")
                )
            return response

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["RAG_EMBEDDING_MODEL"] = self.model_name
        env["RAG_EMBEDDING_DEVICE"] = self.device
        env["RAG_EMBEDDING_BACKEND"] = self.backend
        env.setdefault("USE_TF", "0")
        env.setdefault("USE_FLAX", "0")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        env.setdefault("TRANSFORMERS_NO_FLAX", "1")
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        self._process = subprocess.Popen(
            [self.python_executable, "-m", "rag_agent.embedding_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
            env=env,
            creationflags=creationflags,
        )
        return self._process

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
