import argparse
import json
import hashlib  
from typing import Dict, Optional, Any, List
import numpy as np
from src.embedder import SentenceTransformer
from src.config import RAGConfig
from src.retriever import filter_retrieved_chunks, BM25Retriever, FAISSRetriever, IndexKeywordRetriever, load_artifacts
from sentence_transformers import CrossEncoder

SEMANTIC_CACHE: Dict[str, List[Dict[str, Any]]] = {}
SEMANTIC_CACHE_THRESHOLD = 0.85
SEMANTIC_CACHE_MAX_ENTRIES = 50
QUESTION_EMBEDDERS: Dict[str, SentenceTransformer] = {}



# Add to your global variables
CROSS_ENCODER_MODEL: Optional[CrossEncoder] = None

def get_cross_encoder():
    global CROSS_ENCODER_MODEL
    if CROSS_ENCODER_MODEL is None:
        # A small, fast model ideal for caching verification
        CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return CROSS_ENCODER_MODEL


def normalize_question(q: str) -> str:
    return " ".join((q or "").strip().lower().split())


def make_cache_config_key(
    cfg: RAGConfig,
    args: argparse.Namespace,
    golden_chunks: Optional[list],
) -> str:
    payload = {
        "gen_model": args.model_path or cfg.gen_model,
        "embed_model": cfg.embed_model,
        "top_k": cfg.top_k,
        "system_prompt_mode": args.system_prompt_mode or cfg.system_prompt_mode,
        "ensemble_method": cfg.ensemble_method,
        "ranker_weights": cfg.ranker_weights,
        "use_hyde": cfg.use_hyde,
        "use_indexed_chunks": cfg.use_indexed_chunks,
        "disable_chunks": cfg.disable_chunks,
        "use_golden_chunks": bool(golden_chunks and cfg.use_golden_chunks),
        "index_prefix": getattr(args, "index_prefix", None),
    }
    if golden_chunks and cfg.use_golden_chunks:
        sig = hashlib.sha256("||".join(golden_chunks).encode("utf-8")).hexdigest()
        payload["golden_signature"] = sig
    return json.dumps(payload, sort_keys=True)


def semantic_cache_lookup(
    config_key: str, 
    query_embedding: np.ndarray, 
    current_question: str  # New parameter
):
    entries = SEMANTIC_CACHE.get(config_key) or []
    if not entries or query_embedding is None:
        return None
    
    candidates = []
    for entry in entries:
        cached_vec = entry.get("embedding")
        if cached_vec is None:
            continue
        
        # Fast Bi-Encoder filter (Cosine Similarity)
        sim = float(np.dot(cached_vec, query_embedding))
        
        # Shortlist candidates that are "vaguely" similar
        if sim > 0.40: 
            candidates.append(entry)

    if not candidates:
        return None

    # Verification Step: Cross-Encoder
    ce_model = get_cross_encoder()
    
    # Pair the current user question with every candidate's original question
    pairs = [[current_question, c["question"]] for c in candidates]
    
    # Get scores (higher is more similar)
    ce_scores = ce_model.predict(pairs)
    
    best_idx = np.argmax(ce_scores)
    
    # A score of 0.7-0.8 on most Cross-Encoders indicates strong semantic equivalence
    if ce_scores[best_idx] > 0.75:
        return candidates[best_idx]["payload"]
        
    return None

def semantic_cache_store(
    config_key: str,
    normalized_question: str,
    question_embedding: Optional[np.ndarray],
    payload: Dict[str, Any],
) -> None:
    if question_embedding is None:
        return
    entries = SEMANTIC_CACHE.setdefault(config_key, [])
    entries.append(
        {
            "question": normalized_question,
            "embedding": question_embedding.astype(np.float32),
            "payload": payload,
        }
    )
    if len(entries) > SEMANTIC_CACHE_MAX_ENTRIES:
        entries.pop(0)


def get_question_embedder(
    retrievers: List[Any], cfg: RAGConfig
) -> Optional[SentenceTransformer]:
    for retriever in retrievers or []:
        if isinstance(retriever, FAISSRetriever):
            return retriever.embedder
    model_path = cfg.embed_model
    if not model_path:
        return None
    embedder = QUESTION_EMBEDDERS.get(model_path)
    if embedder is None:
        embedder = SentenceTransformer(model_path)
        QUESTION_EMBEDDERS[model_path] = embedder
    return embedder


def compute_question_embedding(
    question: str,
    retrievers: List[Any],
    cfg: RAGConfig,
) -> Optional[np.ndarray]:
    embedder = get_question_embedder(retrievers, cfg)
    if not embedder:
        return None
    vec = embedder.encode(
        [question],
        batch_size=1,
        normalize=True,
        show_progress_bar=False,
    )
    if vec.size == 0:
        return None
    return vec[0]
