import logging

import faiss
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from src.retriever import Retriever
from src.knowledge_graph.io import RUNS_DIR, load_graph_and_chunks
from src.knowledge_graph.section_tree import SectionTree
from src.knowledge_graph.summary_tree import SummaryEntry
from src.knowledge_graph.utils import KW_PATTERN, Normalizer, extract_ngrams


logger = logging.getLogger(__name__)

# Shared normalizer instance, spaCy model is expensive to load
_normalizer = Normalizer()


class CanonicalLookup:
    """Resolves a normalized keyword to its canonical form at query time.

    Uses a pre-built synonym table (dict lookup, O(1)) for known keywords.
    For unknown keywords, falls back to embedding-based nearest-neighbor search
    against canonical keyword embeddings, gated by a similarity threshold.

    Args:
        synonym_table: Mapping of normalized keyword → canonical form (synonyms only,
            no identity entries).
        canonical_keywords: Ordered list of canonical forms (aligned with embeddings).
        canonical_embeddings: Embedding matrix for canonical keywords (shape N × D).
        embedding_model: Sentence-transformer model name (must match the model used
            during offline canonicalization).
        fallback_threshold: Minimum cosine similarity for the embedding fallback to
            accept a canonical match (default 0.85).
    """

    def __init__(
        self,
        synonym_table: dict[str, str],
        canonical_keywords: list[str],
        canonical_embeddings: np.ndarray,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        fallback_threshold: float = 0.85,
    ):
        self.synonym_table = synonym_table
        self.canonical_keywords = canonical_keywords
        self.canonical_embeddings = canonical_embeddings
        self.fallback_threshold = fallback_threshold
        self._model_name = embedding_model
        self._model = None  # lazy-load

    def resolve(self, keyword: str) -> str:
        """Return the canonical form for *keyword*.

        1. Dictionary lookup in synonym_table.
        2. Embedding nearest-neighbour fallback (if threshold met).
        3. Return *keyword* unchanged if no mapping found.
        """
        if keyword in self.synonym_table:
            return self.synonym_table[keyword]

        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

        emb = self._model.encode([keyword])
        sims = cos_sim(emb, self.canonical_embeddings)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.fallback_threshold:
            synonym = self.canonical_keywords[best_idx]
            # print(f"Embedding fallback: '{keyword}' → '{synonym}' (sim={sims[best_idx]:.4f})")
            return synonym

        return keyword


def _tokens_subsumed(short: str, long: str) -> bool:
    """Return True if the tokens of *short* appear contiguously inside *long*."""
    ws, wl = short.split(), long.split()
    n = len(ws)
    return any(wl[i : i + n] == ws for i in range(len(wl) - n + 1))


def extract_query_nodes(
    query: str,
    graph: nx.Graph,
    canonical_lookup: CanonicalLookup | None = None,
) -> list[str]:
    """Match query terms against graph node labels.

    Generates unigrams, bigrams, and trigrams from *query*, normalises them,
    optionally maps each to its canonical form via *canonical_lookup*, and
    returns any that are present as nodes in *graph*. Shorter nodes that are
    token-level substrings of a longer matched node are dropped.

    Args:
        query: Natural-language query string.
        graph: The knowledge graph to match against.
        canonical_lookup: Optional lookup object for mapping normalized keywords
            to canonical forms. When provided, enables synonym-aware matching
            and an embedding-based fallback for out-of-vocabulary terms.

    Returns:
        List of matched node label strings (may be empty).
    """
    terms = extract_ngrams(query, KW_PATTERN)
    normalized_terms = _normalizer.normalize(terms)

    if canonical_lookup is not None:
        resolved = {canonical_lookup.resolve(t) for t in normalized_terms}
    else:
        resolved = set(normalized_terms)

    matched = [t for t in resolved if graph.has_node(t)]
    filtered = [
        n for n in matched
        if not any(n != m and _tokens_subsumed(n, m) for m in matched)
    ]
    return filtered


class KGNodeRetriever(Retriever):
    """Knowledge-graph retriever that scores chunks via BFS node matching.

    Scores are derived purely from graph topology: direct query-node matches
    score +1.0, and neighbors at hop *k* contribute
    ``neighbor_weight**k * (edge_weight / max_edge_weight)``.
    All scores are normalized to [0, 1].

    Plugs into ``EnsembleRanker`` via the standard ``Retriever`` interface.
    Combine with ``SectionTreeRetriever`` (and others) in the ensemble to
    blend complementary signals.
    """

    name = "kg_node"

    def __init__(
        self,
        graph: nx.Graph,
        kg_chunks: dict[int, str],
        neighbor_weight: float = 0.5,
        num_hops: int = 1,
        canonical_lookup: CanonicalLookup | None = None,
    ):
        self.graph = graph
        self.kg_chunks = kg_chunks
        self.neighbor_weight = neighbor_weight
        self.num_hops = num_hops
        self.canonical_lookup = canonical_lookup

    def get_scores(self, query: str, pool_size: int, chunks: list) -> dict[int, float]:
        """Return BFS-based relevance scores keyed by global chunk index.

        Args:
            query:     Natural-language query string.
            pool_size: Maximum number of chunks to return scores for.
            chunks:    The RAG pipeline's chunk list (used only for length).

        Returns:
            ``Dict[chunk_id, score]`` normalized to [0, 1].
            Returns an empty dict if no query nodes match the graph.
        """
        query_nodes = extract_query_nodes(query, self.graph, self.canonical_lookup)
        logger.debug("Query: %r", query)
        logger.debug("Matched query nodes (%d): %s", len(query_nodes), query_nodes)
        if not query_nodes:
            logger.debug("No query nodes matched — returning empty.")
            return {}

        max_edge_weight = max(
            (data["weight"] for _, _, data in self.graph.edges(data=True)),
            default=1,
        )
        max_edge_weight = max(max_edge_weight, 1)
        logger.debug("Max edge weight in graph: %s", max_edge_weight)

        scores: dict[int, float] = {}

        # Hop 0: directly matched query nodes
        for node in query_nodes:
            for chunk_id in self.graph.nodes[node].get("chunk_ids", []):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0

        # BFS over hops 1..num_hops; each node is visited only at its closest hop
        visited: set[str] = set(query_nodes)
        frontier: set[str] = set(query_nodes)

        for hop in range(1, self.num_hops + 1):
            decay = self.neighbor_weight ** hop
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    next_frontier.add(neighbor)
                    edge_weight = self.graph[node][neighbor].get("weight", 1)
                    contribution = decay * (edge_weight / max_edge_weight)
                    for chunk_id in self.graph.nodes[neighbor].get("chunk_ids", []):
                        scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            visited |= next_frontier
            frontier = next_frontier
            logger.debug("Hop %d: %d new node(s) explored.", hop, len(next_frontier))
            if not frontier:
                break

        if not scores:
            logger.debug("No chunks scored — returning empty.")
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            logger.debug("Max score is %s — returning empty.", max_score)
            return {}

        normalized = {cid: s / max_score for cid, s in scores.items()}
        logger.debug(
            "Normalized scores: %s",
            dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True)),
        )
        return normalized


class SectionTreeRetriever(Retriever):
    """Retriever that scores chunks based on section-heading relevance.

    Uses ``SectionTree.get_chunk_scores`` which blends:
    - Heading keyword overlap (structural signal).
    - KG keyword overlap aggregated from the graph (lexical signal).
    - Top-down score inheritance from parent sections to children.

    Plugs into ``EnsembleRanker`` via the standard ``Retriever`` interface.
    Combine with ``KGNodeRetriever`` (and others) in the ensemble to blend
    complementary signals.
    """

    name = "section_tree"

    def __init__(
        self,
        section_tree: SectionTree,
        graph: nx.Graph,
        canonical_lookup: CanonicalLookup | None = None,
        heading_alpha: float = 0.5,
        inheritance_decay: float = 0.5,
    ):
        self.section_tree = section_tree
        self.graph = graph
        self.canonical_lookup = canonical_lookup
        self.heading_alpha = heading_alpha
        self.inheritance_decay = inheritance_decay

    def get_scores(self, query: str, pool_size: int, chunks: list) -> dict[int, float]:
        """Return section-relevance scores keyed by global chunk index.

        Args:
            query:     Natural-language query string.
            pool_size: Maximum number of chunks to return scores for.
            chunks:    The RAG pipeline's chunk list (unused; present for interface compat).

        Returns:
            ``Dict[chunk_id, score]`` normalized to [0, 1].
        """
        query_keywords = set(extract_query_nodes(query, self.graph, self.canonical_lookup))
        return self.section_tree.get_chunk_scores(
            query_keywords,
            query=query,
            heading_alpha=self.heading_alpha,
            inheritance_decay=self.inheritance_decay,
        )


class SectionSummaryRetriever(Retriever):
    """Retriever that scores chunks via semantic similarity to LLM-generated section summaries.

    At query time the query is embedded and compared against all stored summary
    embeddings (chunk-group, section, and chapter level) using cosine similarity.
    Each FAISS hit distributes its similarity score to every chunk it covers;
    a chunk's final score is the maximum across all hits that include it.

    Plugs into ``EnsembleRanker`` via the standard ``Retriever`` interface.
    Combine with ``KGNodeRetriever`` and ``SectionTreeRetriever`` in the
    ensemble for a richer retrieval signal.
    """

    name = "section_summary"

    def __init__(
        self,
        index: faiss.Index,
        entries: list[SummaryEntry],
        embed_model: str = "all-MiniLM-L6-v2",
        top_sections: int = 20,
    ):
        self.index = index
        self.entries = entries
        self.embed_model_name = embed_model
        self.top_sections = top_sections
        self._model = None  # lazy-load

    def get_scores(self, query: str, pool_size: int, chunks: list) -> dict[int, float]:
        """Return summary-similarity scores keyed by global chunk index.

        Args:
            query:       Natural-language query string.
            pool_size:   Maximum number of chunks to return scores for (unused;
                         present for interface compatibility).
            chunks:      The RAG pipeline's chunk list (unused).

        Returns:
            ``Dict[chunk_id, score]`` where score is the maximum cosine
            similarity across all summary entries that cover that chunk.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embed_model_name)

        q_emb = self._model.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)

        k = min(self.top_sections, self.index.ntotal)
        similarities, indices = self.index.search(q_emb, k)

        chunk_scores: dict[int, float] = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or sim <= 0:
                continue
            for chunk_id in self.entries[idx].chunk_ids:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), float(sim))

        return chunk_scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the KG node retriever.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=RUNS_DIR,
        help="Run directory or runs/ parent (default: latest run).",
    )
    parser.add_argument("--query", default="What is SQL?")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--neighbor_weight", type=float, default=0.5)
    parser.add_argument("--num_hops", type=int, default=1)
    args = parser.parse_args()

    _graph, _chunks = load_graph_and_chunks(args.output_dir)
    _retriever = KGNodeRetriever(
        _graph, _chunks,
        neighbor_weight=args.neighbor_weight,
        num_hops=args.num_hops,
    )
    _scores = _retriever.get_scores(
        args.query, args.top_k, list(_chunks.values()))
    _results = sorted(
        [(cid, _chunks[cid], score)
         for cid, score in _scores.items() if cid in _chunks],
        key=lambda x: x[2], reverse=True,
    )[:args.top_k]

    print(f"\nTop {len(_results)} results for query: {args.query!r}\n")
    for i, (chunk_id, chunk_text, score) in enumerate(_results, 1):
        print(f"{i}. Chunk ID: {chunk_id}, Score: {score:.4f}")
        print(f"   Text: {chunk_text[:200]}...\n")
