"""
Test adapter for the LlamaRetriever pipeline.

Exposes a `get_llamaretriever_answer` function with the same
(answer, chunks_info, hyde_query) return convention used by the
benchmark harness so that `test_benchmarks.py` can run against this
backend with `--backend llamaretriever`.

Chunk IDs are dummy placeholders for now because the LlamaIndex
pipeline does not produce the integer chunk IDs used by the original
TokenSmith retriever.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import LlamaIndexConfig


# ── Module-level cache so models are loaded only once per session ─────────

_cached_artifacts: Optional[Dict] = None
_cached_logger = None
_cached_cfg: Optional[LlamaIndexConfig] = None


def _init_once(config: dict) -> Tuple[LlamaIndexConfig, Dict, "QueryLogger"]:
    """
    Build LlamaIndexConfig and initialise heavy artifacts (LLM, embeddings,
    index, retriever, reranker) exactly once.  Subsequent calls return the
    cached objects.
    """
    global _cached_artifacts, _cached_logger, _cached_cfg

    if _cached_artifacts is not None:
        return _cached_cfg, _cached_artifacts, _cached_logger

    from .main import init_artifacts

    cfg = LlamaIndexConfig.from_yaml("config/config.yaml")

    # Allow the test harness to override a handful of knobs
    if config.get("gen_model"):
        cfg.gen_model = config["gen_model"]
    if config.get("embed_model"):
        cfg.embed_model = config["embed_model"]
    if config.get("max_gen_tokens"):
        cfg.max_gen_tokens = config["max_gen_tokens"]

    artifacts, logger = init_artifacts(cfg)

    _cached_cfg = cfg
    _cached_artifacts = artifacts
    _cached_logger = logger

    return cfg, artifacts, logger


# ── Public API consumed by test_benchmarks.py ─────────────────────────────


def get_llamaretriever_answer(
    question: str,
    config: dict,
    golden_chunks: Optional[list] = None,
) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Run *question* through the LlamaRetriever pipeline and return the
    same triple the benchmark harness expects:

        (generated_answer, chunks_info, hyde_query)

    * ``chunks_info`` — list of dicts with at least a ``chunk_id`` key.
      For now every entry gets ``chunk_id = -1`` (dummy) because the
      LlamaIndex pipeline does not produce matching integer IDs.
    * ``hyde_query`` — always ``None``; HyDE is not implemented here.
    """
    from .main import get_answer

    cfg, artifacts, logger = _init_once(config)

    print(f"  🔍 [llamaretriever] Retrieving + curating evidence …")

    answer = get_answer(
        question=question,
        cfg=cfg,
        artifacts=artifacts,
        logger=logger,
    )

    # ── Build dummy chunks_info ───────────────────────────────────────────
    # The benchmark scorer expects a list[dict] where each dict has at
    # least a "chunk_id" key.  We emit one dummy entry per reference that
    # the agent selected so the rest of the scoring pipeline doesn't break.
    chunks_info: List[Dict] = [
        {"chunk_id": -1, "content": "(llamaretriever – dummy chunk ID)"}
    ]

    return answer, chunks_info, None
