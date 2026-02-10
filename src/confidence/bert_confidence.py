"""
BERT-based confidence scorer for hallucination detection.

Uses DeBERTa NLI model to calculate confidence scores by comparing
generated answers (hypothesis) against retrieved chunks (premise).

Also uses semantic similarity to penalize unrelated text pairs.
"""
import os
import warnings
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERTConfidenceScorer:
    """
    Uses DeBERTa NLI model to score answer confidence based on chunks.
    
    Compares: chunks (premise) vs answer (hypothesis)
    Returns: confidence score [0, 1] where:
    - 1.0 = answer fully entailed by chunks
    - 0.5 = answer neutral/partially supported
    - 0.0 = answer contradicts chunks
    """
    
    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", embed_model: str = None):
        """
        Initialize the BERT confidence scorer.
        
        Args:
            model_name: HuggingFace model name for NLI
            embed_model: Path to embedding model for semantic similarity check (optional)
        """
        self.model_name = model_name
        self.embed_model = embed_model
        self._tokenizer = None
        self._model = None
        self._embedder = None
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialization of model and tokenizer."""
        if self._initialized:
            return
        
        try:
            # Suppress CUDA warnings if running on CPU
            os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()  # Set to evaluation mode
            
            # Initialize embedder for semantic similarity if model path provided
            if self.embed_model:
                try:
                    from src.retriever import _get_embedder
                    self._embedder = _get_embedder(self.embed_model)
                except Exception as e:
                    self._embedder = None
            
            self._initialized = True
        except Exception as e:
            raise
    
    def calculate_confidence(self, answer: str, chunks: List[str]) -> float:
        """
        Calculate confidence that answer is supported by chunks.
        
        Args:
            answer: Generated answer text
            chunks: List of retrieved chunk texts
            
        Returns:
            Confidence score [0, 1] where:
            - 1.0 = answer fully entailed by chunks
            - 0.5 = answer neutral/partially supported
            - 0.0 = answer contradicts chunks or empty input
        """
        if not answer.strip() or not chunks:
            return 0.0
        
        # Lazy initialization
        if not self._initialized:
            self._initialize()
        
        try:
            # FIRST: Check semantic similarity to filter out completely unrelated pairs
            # This prevents NLI from giving high scores to unrelated text
            semantic_similarities = []
            if self._embedder:
                try:
                    answer_emb = self._embedder.encode([answer]).astype("float32")[0]
                    answer_emb_norm = answer_emb / (np.linalg.norm(answer_emb) + 1e-8)
                    
                    for chunk in chunks:
                        chunk_emb = self._embedder.encode([chunk]).astype("float32")[0]
                        chunk_emb_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-8)
                        # Cosine similarity
                        similarity = np.dot(answer_emb_norm, chunk_emb_norm)
                        semantic_similarities.append(float(similarity))
                except Exception as e:
                    semantic_similarities = [1.0] * len(chunks)  # Fallback: assume similar
            else:
                semantic_similarities = [1.0] * len(chunks)  # No embedder: skip semantic check
            
            # Compare answer against EACH chunk individually
            # Take the MAXIMUM confidence (if ANY chunk strongly supports it, we're confident)
            chunk_confidences = []
            max_length = 512
            
            for i, chunk in enumerate(chunks):
                # Truncate chunk if needed (prioritize keeping answer intact)
                inputs = self._tokenizer(
                    chunk,
                    answer,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Get predictions for this chunk
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits[0]
                    probs = torch.softmax(logits, dim=-1)
                
                # Interpret: [entailment, neutral, contradiction]
                entailment_prob = probs[0].item()
                neutral_prob = probs[1].item()
                contradiction_prob = probs[2].item()
                
                # Convert to confidence score for this chunk
                nli_confidence = (
                    entailment_prob * 1.0 +
                    neutral_prob * 0.5 +
                    contradiction_prob * 0.0
                )
                
                # Apply semantic similarity penalty: if chunks are semantically unrelated, reduce confidence
                semantic_sim = semantic_similarities[i] if i < len(semantic_similarities) else 0.5
                
                # If semantic similarity is very low (< 0.3), heavily penalize even if NLI says entailment
                if semantic_sim < 0.3:
                    chunk_confidence = nli_confidence * semantic_sim
                elif semantic_sim < 0.5:
                    chunk_confidence = nli_confidence * (0.5 + semantic_sim)
                else:
                    chunk_confidence = nli_confidence
                
                chunk_confidences.append({
                    "total": chunk_confidence,
                    "nli": nli_confidence,
                    "semantic": semantic_sim
                })
            
            # Take MAXIMUM confidence across all chunks
            if not chunk_confidences:
                return {"confidence": 0.0, "nli_score": 0.0, "semantic_score": 0.0}
            
            best_chunk_data = max(chunk_confidences, key=lambda x: x["total"])
            confidence = best_chunk_data["total"]
            best_nli = best_chunk_data["nli"]
            best_sem = best_chunk_data["semantic"]
            
            return {
                "confidence": confidence,
                "nli_score": best_nli,
                "semantic_score": best_sem
            }
            
        except Exception as e:
            return {"confidence": 0.0, "nli_score": 0.0, "semantic_score": 0.0}
    
    def should_return_answer(self, confidence: float, threshold: float = 0.5) -> bool:
        """
        Determine if answer should be returned based on confidence threshold.
        
        Args:
            confidence: Confidence score [0, 1]
            threshold: Minimum confidence to return answer
            
        Returns:
            True if confidence >= threshold, False otherwise
        """
        return confidence >= threshold
