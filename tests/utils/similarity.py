import os
import json
import difflib
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SimilarityScorer:
    def __init__(self):
        # Set env variable to force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            # Force CPU device because GPU doesn't support CUDA
            self.device = 'cpu'
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            warnings.filterwarnings("ignore", message=".*cuda.*", category=UserWarning)
            
            # Load model with explicit CPU device
            self.model = SentenceTransformer('all-MiniLM-L12-v2', device=self.device)
            self.util = util
            self.use_embeddings = True
            
            print(f"SimilarityScorer initialized with device: {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load sentence transformer, using text similarity only. Error: {e}")
            self.use_embeddings = False
    
    def text_similarity(self, text1, text2):
        """Calculate text similarity using SequenceMatcher."""
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using embeddings on CPU."""
        if not self.use_embeddings:
            return self.text_similarity(text1, text2)
        
        try:
            embeddings = self.model.encode(
                [text1, text2], 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=True
            )
            
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu()
            
            similarity = self.util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            print(f"Warning: Semantic similarity failed, falling back to text similarity. Error: {e}")
            return self.text_similarity(text1, text2)
    
    def keyword_match_score(self, text, keywords):
        """Calculate keyword matching score."""
        if not keywords:
            return 0
        
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched / len(keywords)
    
    def comprehensive_score(self, answer, expected, keywords):
        """Calculate comprehensive similarity score using CPU-only operations."""
        text_sim = self.text_similarity(answer, expected)
        semantic_sim = self.semantic_similarity(answer, expected) if self.use_embeddings else 0
        keyword_score = self.keyword_match_score(answer, keywords)
        
        # Weighted combination - arbitary for now
        if self.use_embeddings:
            final_score = 0.3 * text_sim + 0.5 * semantic_sim + 0.2 * keyword_score
        else:
            final_score = 0.7 * text_sim + 0.3 * keyword_score
        
        return {
            "text_similarity": text_sim,
            "semantic_similarity": semantic_sim,
            "keyword_score": keyword_score,
            "final_score": final_score,
            "keywords_matched": sum(1 for kw in keywords if kw.lower() in answer.lower())
        }
