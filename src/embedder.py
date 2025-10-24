import numpy as np
from typing import List, Union
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
