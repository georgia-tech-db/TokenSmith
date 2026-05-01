import json
import logging
from pathlib import Path
from typing import Optional
import argparse
import requests

from src.config import RAGConfig
from src.cache import get_cache

logger = logging.getLogger(__name__)

def run_scheduler_job(cfg: RAGConfig, args: argparse.Namespace, remote_url: str = "http://localhost:8000") -> None:
    """
    Triggered when the user enters '@server' in the local chat.
    Reads the latest log, extracts the query, sends it to the API server, and caches the result locally.
    """
    logs = Path("logs").glob("chat_*.json")

    # Find the most recently modified log file
    latest_log = sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    try:
        with open(latest_log, "r", encoding="utf-8") as f:
            log_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read log file {latest_log.name}: {e}")
        return

    query = log_data.get("query", "")
    logger.info(f"Triggering scheduler for query from {latest_log.name}")

    url = f"{remote_url}/scheduler/run"
    try:
        # We send the query and the log_data for context
        response = requests.post(
            url,
            json={"queries": [query], "log_data": log_data},
            timeout=300
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except requests.RequestException as e:
        logger.error(f"Failed to reach API server at {url}: {e}")
        return

    # Store the returned answers in the local semantic cache
    cache = get_cache(cfg)
    
    for result in results:
        ans = result.get("answer", "")
        q = result.get("query", "")
        new_score = result.get("score", 0.0)

        existing_score = 0.0
        if "retrieved_chunks" in log_data and len(log_data["retrieved_chunks"]) > 0:
            existing_score = log_data["retrieved_chunks"][0].get("score", 0.0)
        elif "ordered_scores" in log_data and len(log_data["ordered_scores"]) > 0:
            existing_score = log_data["ordered_scores"][0]

        if ans and new_score > existing_score:
            # Normalize and compute the embedding for the query
            normalized_q = cache.normalize_question(q)
            config_cache_key = cache.make_config_key(cfg, args, None)
            
            # Since retrievers aren't easily accessible here, we pass []
            # compute_embedding will instantiate a SentenceTransformer if needed
            embed_model = cfg.embed_model
            question_embedding = cache.compute_embedding(normalized_q, [], embed_model)
            
            cache_payload = {
                "answer": ans,
                "chunks_info": None,
                "hyde_query": None,
                "chunk_indices": [],
            }
            cache.store(config_cache_key, normalized_q, question_embedding, cache_payload)
            logger.info(f"Stored generated answer for '{q}' in local semantic cache (new_score: {new_score:.4f} > existing_score: {existing_score:.4f}).")
        elif ans:
            logger.info(f"Skipping cache update for '{q}' (new_score: {new_score:.4f} <= existing_score: {existing_score:.4f}).")
