import sqlite3
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from llama_cpp import Llama
from tqdm import tqdm

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 40960, n_threads: int = None):
        """
        Initialize with a local GGUF model file path.
        
        Args:
            model_path: Path to your local .gguf file
            n_ctx: Context window size (increased to match Qwen3 training context)
            n_threads: Number of threads to use (None = auto-detect)
        """
        print(f"Loading model with n_ctx={n_ctx}, n_threads={n_threads}")
        
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=False,
            n_batch=512,
            use_mmap=True,
            logits_all=True
        )
        self._embedding_dimension = None
        
        _ = self.embedding_dimension
        print(f"Model loaded successfully. Embedding dimension: {self._embedding_dimension}")

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: int = 32,
               normalize: bool = False,
               device: str = None,
               show_progress_bar: bool = False,
               **kwargs) -> np.ndarray:
        """
        Encode texts to embeddings with batch processing.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Number of texts to process at once
            normalize: Whether to normalize embeddings
            device: Compatibility param (ignored, CPU only)
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy.ndarray: Float32 embeddings array
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        print(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            batch_embeddings = []
            for text in batch_texts:
                try:
                    embedding = self.model.create_embedding(text)['data'][0]['embedding']
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error encoding text: {e}")
                    batch_embeddings.append([0.0] * self.embedding_dimension)
			
            if len(batch_embeddings) != len(batch_texts):
                batch_embeddings.extend([[0.0] * self.embedding_dimension] * (len(batch_texts) - len(batch_embeddings)))
			
            embeddings.extend(batch_embeddings)
                
        vecs = np.array(embeddings, dtype=np.float32)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            vecs = vecs / norms
            
        return vecs

    def embed_one(self, text: str, normalize: bool = False) -> List[float]:
        """Encode single text and return as list."""
        return self.encode([text], normalize=normalize)[0].tolist()

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension


class EmbeddingCache:
    """Persistent SQLite cache for embeddings."""
    
    def __init__(self, cache_dir: str = "index/cache"):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
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
        """Retrieve cached embedding if it exists."""
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                (model_hash, query)
            ).fetchone()
            
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None
    
    def set(self, model_path: str, query: str, embedding: np.ndarray):
        """Store embedding in cache."""
        model_name = Path(model_path).stem
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        blob = embedding.astype(np.float32).tobytes()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (model_name, model_hash, query_text, embedding) VALUES (?,?,?,?)",
                (model_name, model_hash, query, blob)
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
        """Encode texts with caching support."""
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_path, text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)
        
        # Compute missing embeddings
        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                self.cache.set(self.model_path, text, emb)
                results.append((idx, emb))
        
        # Restore original order
        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])
    
    def __getattr__(self, name):
        """Delegate other methods to wrapped embedder."""
        return getattr(self.embedder, name)
