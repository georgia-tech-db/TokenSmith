import numpy as np
from typing import List, Union
from llama_cpp import Llama
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
import os


def _encode_worker(texts: List[str], model_path: str, batch_size: int, embedding_dim: int) -> np.ndarray:
    """
    Worker function that loads model and encodes texts.
    Each worker process gets its own model instance.
    """
    # Load model in worker process
    model = Llama(
        model_path=model_path,
        n_ctx=40960,
        embedding=True,
        verbose=False,
        n_batch=512,
        use_mmap=True,
        logits_all=True
    )
    
    embeddings = []
    for text in texts:
        try:
            emb = model.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        except Exception as e:
            print(f"Error encoding text in worker: {e}")
            embeddings.append([0.0] * embedding_dim)
    
    return np.array(embeddings, dtype=np.float32)


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
        
        self._model_path = model_path  # Store for parallel encoding
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
    
    def encode_parallel(self,
                       texts: List[str],
                       num_workers: int = None,
                       batch_size: int = 32,
                       show_progress_bar: bool = True) -> np.ndarray:
        """
        Encode texts in parallel using multiprocessing.
        Each worker gets its own model instance.
        
        Args:
            texts: List of texts to encode
            num_workers: Number of worker processes (None = auto-detect)
            batch_size: Batch size per worker
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy.ndarray: Float32 embeddings array
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        if num_workers is None:
            num_workers = max(1, min(cpu_count() - 1, len(texts) // 10))
            num_workers = max(1, num_workers)  # At least 1 worker
        
        # For small batches, use regular encoding
        if len(texts) < 100 or num_workers == 1:
            return self.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        
        # Split texts into worker chunks
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        
        # Get model path from the model object (we need to store it)
        model_path = getattr(self, '_model_path', None)
        if model_path is None:
            # Fallback to regular encoding if we can't get model path
            print("Warning: Cannot determine model path for parallel encoding. Using sequential encoding.")
            return self.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        
        print(f"Encoding {len(texts)} texts using {num_workers} workers...")
        
        # Create worker pool
        worker_args = [(chunk, model_path, batch_size, self.embedding_dimension) 
                      for chunk in text_chunks]
        
        with Pool(num_workers) as pool:
            results = pool.starmap(_encode_worker, worker_args)
        
        # Concatenate results
        all_embeddings = np.vstack(results)
        return all_embeddings

    def embed_one(self, text: str, normalize: bool = False) -> List[float]:
        """Encode single text and return as list."""
        return self.encode([text], normalize=normalize)[0].tolist()

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of embeddings (compatibility method)."""
        return self.embedding_dimension
