from typing import List, Tuple, Dict
import os, pickle, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os, pathlib, fitz, re
from typing import List
from src.chunking import make_chunk_strategy

def build_tfidf_tags(
    texts: List[str],
    *,
    ngram_range: Tuple[int, int] = (1, 3),
    max_features: int = 25000,
    min_df: int = 2,
    max_df: float = 0.6,
    top_k_per_chunk: int = 10,
) -> Tuple[TfidfVectorizer, List[List[str]]]:
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
        norm="l2",
        lowercase=True,
    )
    X = vec.fit_transform(texts)
    fn = np.array(vec.get_feature_names_out())

    tags_per_chunk: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            tags_per_chunk.append([]); continue
        idx, data = row.indices, row.data
        top = np.argsort(-data)[:top_k_per_chunk]
        tags_per_chunk.append(fn[idx[top]].tolist())
    return vec, tags_per_chunk

def query_top_tags(q: str, vectorizer: TfidfVectorizer, top_q: int = 8) -> List[str]:
    Xq = vectorizer.transform([q])
    if Xq.nnz == 0: return []
    fn = np.array(vectorizer.get_feature_names_out())
    idx, data = Xq.nonzero()[1], Xq.data
    top = np.argsort(-data)[:top_q]
    return fn[idx[top]].tolist()

def tag_affinity_score(
    chunk_tags: List[str], query_tags: List[str],
    *, mode: str = "weighted", tag_weights: Dict[str, float] = None
) -> float:
    if not chunk_tags or not query_tags: return 0.0
    s_chunk, s_query = set(chunk_tags), set(query_tags)
    inter = s_chunk & s_query
    if mode == "jaccard":
        return len(inter) / max(1, len(s_chunk | s_query))
    if not tag_weights: return float(len(inter))
    return float(sum(tag_weights.get(t, 1.0) for t in inter))


# ===================== EXP

_TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

def _extract_tables(text: str):
    tables = _TABLE_RE.findall(text)
    for i, t in enumerate(tables):
        text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
    return text, tables

def _restore_tables(chunk: str, tables: List[str]) -> str:
    for i, t in enumerate(tables):
        ph = f"[TABLE_PLACEHOLDER_{i}]"
        if ph in chunk:
            chunk = chunk.replace(ph, t)
    return chunk

def _pdf_to_text(pdf_path: str) -> str:
    with fitz.open(pdf_path) as doc:
        return "".join(page.get_text() for page in doc)
