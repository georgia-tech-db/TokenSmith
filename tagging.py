from typing import List, Tuple, Dict
import os, pickle, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os, pathlib, fitz, re
from typing import List
from chunking import make_chunk_strategy

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


# ---------- FAFO entrypoint: chunk + tag a single PDF ----------
def run_single_pdf_tagging(
    pdf_path: str = "chapters/31.pdf",
    out_prefix: str = "adhoc_31",
    *,
    chunk_mode: str = "sliding-tokens",    # chars | tokens | sliding-tokens
    chunk_tokens: int = 310,               # used by tokens/sliding-tokens
    chunk_overlap_tokens: int = 128,       # only for sliding-tokens
    chunk_size_char: int = 20_000,         # used by char mode
    tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    keep_tables: bool = True,
    top_k_per_chunk: int = 10,             # tags per chunk
) -> None:
    pdf_path = str(pdf_path)
    print(f"📄  Reading {pdf_path} …")
    text = _pdf_to_text(pdf_path)
    if not text:
        raise RuntimeError(f"No text extracted from {pdf_path}")

    # Table-safe chunking with your strategy factory (no dependency on preprocess.py)
    strategy = make_chunk_strategy(
        mode=chunk_mode,
        chunk_size_char=chunk_size_char,
        chunk_tokens=chunk_tokens,
        tokenizer_name=tokenizer_name,
    )

    work = text
    tables = []
    if keep_tables:
        work, tables = _extract_tables(work)

    chunks: List[str]
    if chunk_mode == "sliding-tokens":
        # the factory already set max_tokens from chunk_tokens; we also want overlap
        # Sneak in overlap if present on the strategy
        try:
            strategy.overlap_tokens = int(chunk_overlap_tokens)
        except Exception:
            pass
        chunks = strategy.chunk(work)
    else:
        chunks = strategy.chunk(work)

    if keep_tables and tables:
        chunks = [_restore_tables(c, tables) for c in chunks]

    print(f"🧩  Produced {len(chunks)} chunks with strategy={strategy.name()}")

    # Build TF-IDF tags and persist to meta/
    print("🏷️  Building TF-IDF tags …")
    vec, chunk_tags = build_tfidf_tags(
        chunks,
        ngram_range=(1, 1),
        max_features=25000,
        min_df=2,
        max_df=0.6,
        top_k_per_chunk=top_k_per_chunk,
    )

    os.makedirs("meta", exist_ok=True)
    # Save artifacts for FAFO inspection/re-use
    # pickle_save(chunks,      os.path.join("meta", f"{out_prefix}_chunks.pkl"))
    # pickle_save(chunk_tags,  os.path.join("meta", f"{out_prefix}_tags.pkl"))
    # pickle_save(vec,         os.path.join("meta", f"{out_prefix}_tfidf.pkl"))

    # Tiny console preview
    print(f"✅  Saved meta/{out_prefix}_chunks.pkl, _tags.pkl, _tfidf.pkl")
    preview_n = min(12, len(chunks))
    for i in range(preview_n):
        snippet = (chunks[i][:140].replace("\n", " ") + "…") if len(chunks[i]) > 140 else chunks[i]
        print(f"[chunk {i:02d}] tags={chunk_tags[i][:8]} | {snippet}")


# ---------- script runner ----------
if __name__ == "__main__":
    # Defaults are intentionally hard-coded (no CLI) so you can just:
    #   python tagging.py
    # Tweak parameters here as you FAFO.
    run_single_pdf_tagging()