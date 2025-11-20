import numpy as np
from typing import List, Union
import os
from tqdm import tqdm

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 8192, n_threads: int = None):
        """
        Initialize with a local GGUF model file path OR a HuggingFace model name.
        
        Args:
            model_path: Path to .gguf file OR HuggingFace model name (e.g. 'sentence-transformers/all-MiniLM-L6-v2')
            n_ctx: Context window size (only for GGUF)
            n_threads: Number of threads to use (only for GGUF)
        """
        self.model_path = model_path
        self.is_gguf = model_path.endswith(".gguf")
        self._embedding_dimension = None

        if self.is_gguf:
            from llama_cpp import Llama
            print(f"Loading GGUF model with n_ctx={n_ctx}, n_threads={n_threads}")
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                embedding=True,
                verbose=False,
                n_batch=512,
                use_mmap=True,
                logits_all=False
            )
            # Cache dim
            _ = self.embedding_dimension
            print(f"GGUF Model loaded. Embedding dimension: {self._embedding_dimension}")
        else:
            from sentence_transformers import SentenceTransformer as ST
            print(f"Loading HuggingFace model: {model_path}")
            self.model = ST(model_path)
            self._embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"HF Model loaded. Embedding dimension: {self._embedding_dimension}")

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (cached after first call)."""
        if self._embedding_dimension is None:
            if self.is_gguf:
                test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
                self._embedding_dimension = len(test_embedding)
            else:
                # Should be set in init for HF, but just in case
                self._embedding_dimension = self.model.get_sentence_embedding_dimension()
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
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        # HuggingFace path
        if not self.is_gguf:
            return self.model.encode(
                texts, 
                batch_size=batch_size, 
                normalize_embeddings=normalize, 
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )

        # GGUF path
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
                    # Return zero vector on failure
                    if self._embedding_dimension:
                         batch_embeddings.append([0.0] * self._embedding_dimension)
                    else:
                         # Fallback if dim unknown (unlikely)
                         batch_embeddings.append([])

            # Pad if needed (shouldn't happen with loop above but good safety)
            if len(batch_embeddings) != len(batch_texts):
                 dim = self.embedding_dimension
                 batch_embeddings.extend([[0.0] * dim] * (len(batch_texts) - len(batch_embeddings)))
            
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
