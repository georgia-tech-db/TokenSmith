"""
Simple embedder wrapper that handles both sentence-transformers and Qwen models.
"""
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class Embedder:
    """Simple embedder that works with both sentence-transformers and Qwen models."""
    
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._is_sentence_transformer = self._detect_type()
        self._load_model()
    
    def _detect_type(self) -> bool:
        """Detect if this is a sentence-transformers model."""
        return "sentence-transformers/" in self.model_name or "all-MiniLM" in self.model_name or "all-mpnet" in self.model_name
    
    def _load_model(self):
        """Load the appropriate model."""
        if self._is_sentence_transformer:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        else:
            # For Qwen and other HF models
            self._model = AutoModel.from_pretrained(self.model_name, device_map=self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def encode(self, texts: List[str], batch_size: int = 4, show_progress_bar: bool = True):
        """Encode texts to embeddings."""
        if self._is_sentence_transformer:
            return self._model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress_bar)
        else:
            return self._encode_hf(texts, batch_size)
    
    def _encode_hf(self, texts: List[str], batch_size: int):
        """Encode using HuggingFace models like Qwen."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self._tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Use mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(batch_embeddings)
        
        return torch.cat(embeddings, dim=0)
