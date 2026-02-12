"""
Main RAG pipeline: ties indexing, retrieval, reranking, and generation together.

Usage:
    from src.llamaindex.pipeline import RAGPipeline

    pipe = RAGPipeline()          # uses default config
    pipe.build_index()            # index documents
    answer = pipe.query("What is normalization?")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole

from llama_index.core.callbacks import CallbackManager, CBEventType, LlamaDebugHandler
from llama_index.core.callbacks.base_handler import BaseCallbackHandler

from .config import LlamaIndexConfig
from .models import build_llm, build_embed_model
from .indexer import build_index, load_index, get_or_build_index
from .retriever import build_retriever, build_reranker


class PromptCapture(BaseCallbackHandler):
    """Lightweight callback that captures the last LLM prompt."""

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self.last_prompt: str = ""

    def on_event_start(self, event_type, payload=None, event_id="", **kwargs):
        if event_type == CBEventType.LLM and payload:
            # The prompt is in payload under various keys depending on version
            prompt = payload.get("formatted_prompt") or payload.get("messages") or ""
            print('===full prompt===', prompt)
            if prompt:
                self.last_prompt = str(prompt)

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        pass

    def start_trace(self, trace_id=None):
        pass

    def end_trace(self, trace_id=None, trace_map=None):
        pass


class RAGPipeline:
    """
    End-to-end RAG pipeline powered by LlamaIndex.

    Designed to be a competitive baseline to the hand-rolled TokenSmith pipeline.
    """

    def __init__(
        self,
        config: Optional[LlamaIndexConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config is not None:
            self.cfg = config
        elif config_path is not None:
            self.cfg = LlamaIndexConfig.from_yaml(config_path)
        else:
            self.cfg = LlamaIndexConfig()

        # Initialize global LlamaIndex settings with our models
        self._llm = build_llm(self.cfg)
        self._embed_model = build_embed_model(self.cfg)
        Settings.llm = self._llm
        Settings.embed_model = self._embed_model
        Settings.chunk_size = self.cfg.chunk_size
        Settings.chunk_overlap = self.cfg.chunk_overlap

        self._index = None
        self._query_engine = None

    # ── Index management ────────────────────────────────────────────────

    def index(self, force_rebuild: bool = False) -> None:
        """Build (or rebuild) the document index."""
        if force_rebuild:
            self._index = build_index(self.cfg)
        else:
            self._index = get_or_build_index(self.cfg, force_rebuild=force_rebuild)
        self._query_engine = None  # invalidate cached engine

    def load(self) -> None:
        """Load a previously built index from disk."""
        self._index = load_index(self.cfg)
        self._query_engine = None

    # ── Query engine construction ───────────────────────────────────────

    def _get_query_engine(self) -> RetrieverQueryEngine:
        """Lazily build the query engine with hybrid retrieval + reranking."""
        if self._query_engine is not None:
            return self._query_engine

        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call pipeline.index() or pipeline.load() first."
            )

        retriever = build_retriever(self._index, self.cfg)
        reranker = build_reranker(self.cfg)

        node_postprocessors = []
        if reranker is not None:
            node_postprocessors.append(reranker)

        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode="compact",  # efficient: stuffs as much context as fits
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )
        return self._query_engine

    # ── Querying ────────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """
        Run a query through the full RAG pipeline.

        Returns the generated answer string.
        """
        engine = self._get_query_engine()
        response = engine.query(question)
        return str(response)

    def query_with_sources(self, question: str) -> dict:
        """
        Query and return the answer, source chunks, and the final prompt.

        Returns:
            {
                "answer": str,
                "sources": [{"text": str, "score": float, "metadata": dict}, ...],
                "prompt": str,   # the final prompt sent to the LLM
            }
        """
        engine = self._get_query_engine()

        # Capture the final prompt via a callback
        prompt_capture = PromptCapture()
        engine.callback_manager.add_handler(prompt_capture)

        response = engine.query(question)

        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "text": node.text,
                    "score": node.score,
                    "metadata": node.metadata,
                }
            )

        return {
            "answer": str(response),
            "sources": sources,
            "prompt": prompt_capture.last_prompt,
        }

    def query_verbose(self, question: str) -> dict:
        """
        Query with full retrieval diagnostics.

        Runs retrieval + reranking explicitly so we can inspect the chunks
        before they reach the LLM, then generates an answer.

        Returns:
            {
                "answer": str,
                "chunks": [{"rank": int, "score": float, "text": str, "metadata": dict}, ...],
            }
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call pipeline.index() or pipeline.load() first."
            )

        from llama_index.core import QueryBundle

        retriever = build_retriever(self._index, self.cfg)
        reranker = build_reranker(self.cfg)

        # Step 1: Retrieve
        query_bundle = QueryBundle(query_str=question)
        nodes = retriever.retrieve(query_bundle)

        # Step 2: Rerank
        if reranker is not None:
            nodes = reranker.postprocess_nodes(nodes, query_bundle)

        # Collect chunk diagnostics
        chunks_info = []
        for rank, node in enumerate(nodes, 1):
            chunks_info.append(
                {
                    "rank": rank,
                    "score": node.score,
                    "text": node.text,
                    "metadata": node.metadata,
                }
            )

        # Step 3: Generate using the same compact synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode="compact",
        )
        response = response_synthesizer.synthesize(question, nodes)

        return {
            "answer": str(response),
            "chunks": chunks_info,
        }

    def stream_query(self, question: str):
        """
        Stream the response token-by-token.

        Returns (source_nodes, response_gen) so the caller can print chunks
        before streaming the answer.
        """
        if self._index is None:
            raise RuntimeError(
                "No index loaded. Call pipeline.index() or pipeline.load() first."
            )

        retriever = build_retriever(self._index, self.cfg)
        reranker = build_reranker(self.cfg)

        node_postprocessors = []
        if reranker is not None:
            node_postprocessors.append(reranker)

        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode="compact",
            streaming=True,
        )

        streaming_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=node_postprocessors,
        )

        streaming_response = streaming_engine.query(question)
        return streaming_response.source_nodes, streaming_response.response_gen
