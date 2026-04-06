"""
Test adapter for the BookRAG pipeline.

Exposes a `get_llamaretriever_answer` function with the same
(answer, chunks_info, hyde_query) return convention used by the
benchmark harness so that `test_benchmarks.py` can run against this
backend with `--backend llamaretriever`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import LlamaIndexConfig

_cached_artifacts: Optional[Dict] = None
_cached_logger = None
_cached_cfg: Optional[LlamaIndexConfig] = None


def _init_once(config: dict) -> Tuple[LlamaIndexConfig, Dict, "QueryLogger"]:
    global _cached_artifacts, _cached_logger, _cached_cfg

    if _cached_artifacts is not None:
        return _cached_cfg, _cached_artifacts, _cached_logger

    from .main import init_artifacts

    cfg = LlamaIndexConfig.from_yaml("config/config.yaml")

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


def get_llamaretriever_answer(
    question: str,
    config: dict,
    golden_chunks: Optional[list] = None,
) -> Tuple[str, List[Dict], Optional[str]]:
    """
    Run *question* through the BookRAG pipeline and return the
    same triple the benchmark harness expects:

        (generated_answer, chunks_info, hyde_query)
    """
    from .main import get_answer

    cfg, artifacts, logger = _init_once(config)

    print(f"  [llamaretriever/bookrag] Retrieving + synthesizing ...")

    answer, references = get_answer(
        question=question,
        cfg=cfg,
        artifacts=artifacts,
        logger=logger,
        return_references=True,
    )

    if references:
        evidence_lines = ["\n\nEvidence:"]
        for r in references:
            evidence_lines.append(
                f"\n[{r.id}] {r.source}"
                f"\nPath: {r.header_path}"
                f"\n{r.passage}"
            )
        answer = answer + "".join(evidence_lines)

    chunks_info: List[Dict] = [
        {"chunk_id": -1, "content": "(llamaretriever/bookrag – dummy chunk ID)"}
    ]

    return answer, chunks_info, None
