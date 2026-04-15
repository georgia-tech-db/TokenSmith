import sqlite3
import hashlib
import multiprocessing
import multiprocessing.pool
import os
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from llama_cpp import Llama
from tqdm import tqdm

# Global variables for worker processes
_worker_model: Optional[Llama] = None
_worker_embedding_dim: int = 0


def _truthy_env(name: str) -> bool:
    """Return True if the environment variable *name* is set to a truthy value."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _candidate_contexts(n_ctx: int) -> List[int]:
    """Return a descending list of embedding context sizes to try."""
    candidates = []
    for candidate in [n_ctx, min(n_ctx, 2048), min(n_ctx, 1024), 512]:
        candidate = int(candidate)
        if candidate > 0 and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _load_embedding_model(
    model_path: str,
    n_ctx: int,
    n_threads: int = None,
    verbose: bool = False,
) -> tuple[Llama, int]:
    """Load a GGUF embedding model with GPU/CPU and context-size fallback.

    Args:
        model_path: Path to the ``.gguf`` model file.
        n_ctx: Context window size.
        n_threads: Number of CPU threads (None for auto-detect).
        verbose: Whether to enable verbose llama.cpp output.

    Returns:
        A tuple of ``(model, actual_n_ctx)``.
    """
    force_cpu = _truthy_env("TOKENSMITH_FORCE_CPU")
    last_error = None

    for candidate_n_ctx in _candidate_contexts(n_ctx):
        common_kwargs = {
            "model_path": model_path,
            "n_ctx": candidate_n_ctx,
            "n_threads": n_threads,
            "embedding": True,
            "verbose": verbose,
            "use_mmap": True,
        }

        if force_cpu:
            try:
                return Llama(**common_kwargs), candidate_n_ctx
            except Exception as exc:
                last_error = exc
                continue

        try:
            return Llama(**common_kwargs, n_gpu_layers=-1), candidate_n_ctx
        except Exception as gpu_exc:
            last_error = gpu_exc
            try:
                return Llama(**common_kwargs), candidate_n_ctx
            except Exception as cpu_exc:
                last_error = cpu_exc
                continue

    raise ValueError(
        f"Failed to create embedding llama_context for {model_path} "
        f"after trying n_ctx values {_candidate_contexts(n_ctx)}"
    ) from last_error

def _init_worker(model_path: str, n_ctx: int, n_threads: int):
    """Initializes the model inside a worker process."""
    global _worker_model, _worker_embedding_dim

    _worker_model, _ = _load_embedding_model(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )

    test_emb = _worker_model.create_embedding("test")['data'][0]['embedding']
    _worker_embedding_dim = len(test_emb)


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """Encodes a batch of text using the worker's local model instance."""
    global _worker_model, _worker_embedding_dim
    if _worker_model is None:
        return []

    embeddings = []
    for text in texts:
        try:
            emb = _worker_model.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        except Exception:
            raise

    return embeddings


class SentenceTransformer:
    """GGUF-backed text embedding model with single- and multi-process encoding.

    Wraps a llama.cpp ``Llama`` instance loaded in embedding mode and exposes an
    ``encode()`` interface compatible with sentence-transformers conventions.
    """

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        """
        Initialize with a local GGUF model file path.

        Args:
            model_path: Path to your local .gguf file
            n_ctx:      Context window size. Defaults to 4096.
            n_threads:  Number of threads (None = auto-detect)
        """
        self.model_path = model_path
        self.n_ctx = n_ctx

        self.model, self.n_ctx = _load_embedding_model(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=_truthy_env("TOKENSMITH_VERBOSE_EMBEDDER"),
        )
        self._embedding_dimension = None

        # Warm up — also caches embedding dimension
        _ = self.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def _encode_single(self, text: str) -> List[float]:
        """Encode a single text string and return its raw embedding vector."""
        return self.model.create_embedding(text)["data"][0]["embedding"]

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,  # Adjusted for 4B model
        normalize: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:

        """Encode texts to embeddings with batch processing.

        Args:
            texts: Single text or list of texts to encode.
            batch_size: Number of texts to process at once.
            normalize: Whether to L2-normalize embeddings.
            show_progress_bar: Whether to show a progress bar.

        Returns:
            numpy.ndarray: Float32 embeddings array of shape ``(len(texts), dim)``.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dimension)

        # Process in batches
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            try:
                # llama.cpp's Python wrapper has been unreliable for multi-string
                # embedding batches on the local GGUF backend, especially with Metal.
                # Default to deterministic one-by-one embedding and keep the batched
                # list API opt-in for experimentation only.
                if len(batch_texts) > 1 and _truthy_env("TOKENSMITH_ENABLE_BATCH_EMBEDDINGS"):
                    response = self.model.create_embedding(batch_texts)
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                else:
                    batch_embeddings = [self._encode_single(text) for text in batch_texts]

                embeddings.extend(batch_embeddings)

            except Exception as e:
                print(f"Error encoding batch: {e}")
                for text in batch_texts:
                    embeddings.append(self._encode_single(text))

        vecs = np.array(embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)

        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        """Compatibility method."""
        return self.embedding_dimension

    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        """
        Starts a pool of worker processes.
        """
        # Default to CPU count - 2 (leave room for OS/Main process)
        workers = num_workers or max(1, multiprocessing.cpu_count() - 2)

        print(f"Creating {workers} worker processes...")

        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, 1),
        )
        return pool

    def encode_multi_process(
        self,
        texts: List[str],
        pool: multiprocessing.pool.Pool,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Distributes encoding work across the worker pool."""
        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        chunks = [sorted_texts[i:i + batch_size] for i in range(0, len(sorted_texts), batch_size)]

        results = []
        print(f"Dispatching {len(chunks)} batches to pool...")
        for batch_result in tqdm(
            pool.imap(_encode_batch_worker, chunks),
            total=len(chunks),
            desc="Parallel Encoding",
        ):
            results.append(batch_result)

        flat_embeddings = [emb for batch in results for emb in batch]

        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered_embeddings = [flat_embeddings[i] for i in inverse_indices]

        return np.array(ordered_embeddings, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        """Gracefully shut down the worker pool and wait for processes to exit."""
        pool.close()
        pool.join()


class EmbeddingCache:
    """Persistent SQLite cache for embeddings."""

    def __init__(self, cache_dir: str = "index/cache"):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT,
                    model_hash TEXT,
                    query_text TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_hash, query_text)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")

    def get(self, model_path: str, query: str) -> Optional[np.ndarray]:
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                (model_hash, query),
            ).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set(self, model_path: str, query: str, embedding: np.ndarray):
        model_name = Path(model_path).stem
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        blob = embedding.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(model_name, model_hash, query_text, embedding) VALUES (?,?,?,?)",
                (model_name, model_hash, query, blob),
            )


class CachedEmbedder:
    """
    Wrapper around SentenceTransformer that caches query embeddings.
    Drop-in replacement for SentenceTransformer.
    """

    def __init__(self, model_path: str, **kwargs):
        self.embedder = SentenceTransformer(model_path, **kwargs)
        self.cache = EmbeddingCache()
        self.model_path = model_path

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        results = []
        to_compute = []
        to_compute_indices = []

        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_path, text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)

        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                self.cache.set(self.model_path, text, emb)
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    def __getattr__(self, name):
        return getattr(self.embedder, name)
