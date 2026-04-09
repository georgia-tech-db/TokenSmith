import logging

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

from src.retriever import Retriever
from src.knowledge_graph.io import RUNS_DIR, load_graph_and_chunks
from src.knowledge_graph.section_tree import SectionTree
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
        embedding_model: Path to the GGUF embedding model (must match the model used
            during offline canonicalization).
        fallback_threshold: Minimum cosine similarity for the embedding fallback to
            accept a canonical match (default 0.85).
    """

    def __init__(
        self,
        synonym_table: dict[str, str],
        canonical_keywords: list[str],
        canonical_embeddings: np.ndarray,
        embedding_model: str = "models/Qwen3-Embedding-4B-Q5_K_M.gguf",
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
            from src.embedder import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

        emb = self._model.encode([keyword])
        sims = cos_sim(emb, self.canonical_embeddings)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= self.fallback_threshold:
            return self.canonical_keywords[best_idx]

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
    return [
        n for n in matched
        if not any(n != m and _tokens_subsumed(n, m) for m in matched)
    ]


class KGRetriever(Retriever):
    """Knowledge-graph retriever compatible with the RAG ``EnsembleRanker``.

    Implements the duck-typed interface (``name`` attribute + ``get_scores``
    method) so it can be slotted into the retrievers list without changes to
    the ranking logic.

    When a ``section_tree`` is provided, the final chunk score is a weighted
    blend of the local node-match score and the global section-level score::

        combined = beta * section_score + (1 - beta) * node_score

    Set ``beta = 0.0`` to disable section scoring (pure node-match).
    """

    name = "kg"

    def __init__(
        self,
        graph: nx.Graph,
        kg_chunks: dict[int, str],
        neighbor_weight: float = 0.5,
        num_hops: int = 1,
        section_tree: SectionTree | None = None,
        beta: float = 0.5,
        heading_alpha: float = 0.5,
        inheritance_decay: float = 0.5,
        canonical_lookup: CanonicalLookup | None = None,
    ):
        self.graph = graph
        self.kg_chunks = kg_chunks
        self.neighbor_weight = neighbor_weight
        self.num_hops = num_hops
        self.section_tree = section_tree
        self.beta = beta
        self.heading_alpha = heading_alpha
        self.inheritance_decay = inheritance_decay
        self.canonical_lookup = canonical_lookup

    def get_scores(self, query: str, pool_size: int, chunks: list) -> dict[int, float]:
        """Return KG-based relevance scores keyed by global chunk index.

        If a section tree was provided at construction time, blends local
        node-match scores with global section-level scores.

        Args:
            query:     Natural-language query string.
            pool_size: Maximum number of chunks to return scores for.
            chunks:    The RAG pipeline's chunk list (used only for length).

        Returns:
            ``Dict[chunk_id, score]`` with scores normalized to [0, 1].
            Returns an empty dict if no query nodes match the graph.
        """
        results = self.retrieve_from_kg(
            query,
            top_k=pool_size
        )
        node_scores: dict[int, float] = {
            cid: score for cid, _, score in results}

        if self.section_tree is None or self.beta == 0.0:
            return node_scores

        query_keywords = set(extract_query_nodes(
            query, self.graph, self.canonical_lookup))

        section_scores = self.section_tree.get_chunk_scores(
            query_keywords,
            query=query,
            heading_alpha=self.heading_alpha,
            inheritance_decay=self.inheritance_decay,
        )

        if not section_scores:
            return node_scores

        all_ids = set(node_scores) | set(section_scores)
        combined: dict[int, float] = {
            cid: self.beta * section_scores.get(cid, 0.0)
            + (1 - self.beta) * node_scores.get(cid, 0.0)
            for cid in all_ids
        }

        max_score = max(combined.values(), default=0.0)
        if max_score > 0:
            combined = {cid: v / max_score for cid, v in combined.items()}

        heading_mode = "hybrid" if query is not None else "kg-only"
        logger.debug(
            "Section blending (%s): beta=%s, %d section-scored, %d node-scored → %d combined",
            heading_mode, self.beta, len(section_scores), len(
                node_scores), len(combined),
        )
        return combined

    def retrieve_from_kg(self, query: str, top_k: int = 10) -> list[tuple[int, str, float]]:
        """Retrieve and rank chunks relevant to *query* via the knowledge graph.

        Scoring:
        - Each chunk referenced by a directly-matched query node receives +1.0.
        - Each chunk referenced by a node at hop *k* contributes
        ``neighbor_weight**k * (edge_weight / max_edge_weight)``.
        - Each node is scored only once, at the shortest hop distance from any
        matched query node (BFS order), so ``neighbor_weight`` acts as a
        geometric decay per hop.
        - All scores are normalized to [0, 1] before ranking.

        Args:
            query:           Natural-language query string.
            graph:           Knowledge graph produced by the KG pipeline.
            chunks:          Mapping of chunk ID to chunk text.
            top_k:           Maximum number of results to return.
            neighbor_weight: Per-hop decay factor (0–1) for neighbor contributions.
            num_hops:        Number of hops to traverse from matched query nodes.

        Returns:
            List of ``(chunk_id, chunk_text, score)`` tuples sorted descending.
            Returns an empty list if no query nodes are matched.
        """
        query_nodes = extract_query_nodes(
            query, self.graph, self.canonical_lookup)
        logger.debug("Query: %r", query)
        logger.debug("Matched query nodes (%d): %s",
                     len(query_nodes), query_nodes)
        if not query_nodes:
            logger.debug("No query nodes matched — returning empty.")
            return []

        max_edge_weight = max(
            (data["weight"] for _, _, data in self.graph.edges(data=True)),
            default=1,
        )
        max_edge_weight = max(max_edge_weight, 1)
        logger.debug("Max edge weight in graph: %s", max_edge_weight)

        scores: dict[int, float] = {}

        # Hop 0: directly matched query nodes
        for node in query_nodes:
            node_data = self.graph.nodes[node]
            direct_chunks = node_data.get("chunk_ids", [])
            logger.debug("  Node %r (hop=0): chunk_ids=%s",
                         node, direct_chunks)
            for chunk_id in direct_chunks:
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
                    neighbor_chunks = self.graph.nodes[neighbor].get(
                        "chunk_ids", [])
                    logger.debug(
                        "    Neighbor %r (hop=%d): edge_weight=%s, contribution=%.4f, chunk_ids=%s",
                        neighbor, hop, edge_weight, contribution, neighbor_chunks,
                    )
                    for chunk_id in neighbor_chunks:
                        scores[chunk_id] = scores.get(
                            chunk_id, 0.0) + contribution
            visited |= next_frontier
            frontier = next_frontier
            logger.debug("  Hop %d: %d new node(s) explored.",
                         hop, len(next_frontier))
            if not frontier:
                break

        logger.debug(
            "Raw scores (%d chunks): %s",
            len(scores),
            dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
        )

        if not scores:
            logger.debug("No chunks scored — returning empty.")
            return []

        max_score = max(scores.values())
        if max_score <= 0:
            logger.debug("Max score is %s — returning empty.", max_score)
            return []

        normalized = {cid: s / max_score for cid, s in scores.items()}
        logger.debug(
            "Normalized scores: %s",
            dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True)),
        )

        results = [
            (chunk_id, self.kg_chunks[chunk_id], score)
            for chunk_id, score in normalized.items()
            if chunk_id in self.kg_chunks
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        logger.debug("Returning top %d of %d scored chunks.",
                     min(top_k, len(results)), len(results))
        return results[:top_k]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the KG retriever.")
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
    _retriever = KGRetriever(
        _graph, _chunks,
        neighbor_weight=args.neighbor_weight,
        num_hops=args.num_hops,
    )
    _results = _retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nTop {len(_results)} results for query: {args.query!r}\n")
    for i, (chunk_id, chunk_text, score) in enumerate(_results, 1):
        print(f"{i}. Chunk ID: {chunk_id}, Score: {score:.4f}")
        print(f"   Text: {chunk_text[:200]}...\n")
