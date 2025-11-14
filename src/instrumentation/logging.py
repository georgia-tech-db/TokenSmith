import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.config import QueryPlanConfig

INSTANCE: Optional["RunLogger"]= None

def get_logger() -> "RunLogger":
    global INSTANCE
    if INSTANCE is None:
        raise ValueError("get_logger called before init_logger!")
    return INSTANCE

class RunLogger:
    """
    Comprehensive logging for RAG pipeline runs.
    Creates one log file per session, capturing query analysis, retrieval, ranking, and generation.
    """
    def __init__(self, config: QueryPlanConfig):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        # Create session log file
        self.log_file = self.logs_dir / f"run_{self.session_id}.jsonl"

        # Initialize session
        self.current_query_data = {}
        self.query_count = 0
        self.init_logger()

    def init_logger(self):
        session_info = {
            "event": "session_start",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "config": {
                "chunk_config": self.config.chunk_config.to_string(),
                "top_k": self.config.top_k,
                "pool_size": self.config.pool_size,
                "embed_model": self.config.embed_model,
                "ensemble_method": self.config.ensemble_method,
                "ranker_weights": self.config.ranker_weights,
                "rrf_k": self.config.rrf_k,
                "rerank_mode": self.config.rerank_mode,
                "max_gen_tokens": self.config.max_gen_tokens
            }
        }
        self._write_log(session_info)

    def log_query_start(self, query: str, query_metadata: Optional[Dict] = None):
        """Start logging a new chat response related data."""
        self.query_count += 1
        self.current_query_data = {
            "event": "query",
            "timestamp": datetime.now().isoformat(),
            "query_id": self.query_count,
            "query": query,
            "query_length": len(query),
            "query_word_count": len(query.split()),
            "query_metadata": query_metadata or {}
        }

    def log_planner(self, planner_name: str, base_cfg: Dict[str, Any], new_cfg: Dict[str, Any]):
        """
        Log planner decision: planner name + config diff.
        """
        # Compute diffs
        diff = {}
        for k, base_val in base_cfg.items():
            new_val = new_cfg.get(k, base_val)
            if new_val != base_val:
                diff[k] = {"old": base_val, "new": new_val}

        self.current_query_data["planner"] = {
            "planner_name": planner_name,
            "config_diff": diff,
        }

    def log_retrieval(self, candidates: List[int], faiss_distances: Dict[int, float],
                      pool_size: int, embed_model: str):
        if not self.current_query_data:
            return

        retrieval_data = {
            "pool_size_requested": pool_size,
            "candidates_returned": len(candidates),
            "embed_model": embed_model,
            "candidate_indices": candidates,
            "faiss_distances": faiss_distances,
            "faiss_stats": {
                "min_distance": min(faiss_distances.values()) if faiss_distances else None,
                "max_distance": max(faiss_distances.values()) if faiss_distances else None,
                "avg_distance": sum(faiss_distances.values()) / len(faiss_distances) if faiss_distances else None
            }
        }
        self.current_query_data["retrieval"] = retrieval_data

    def log_ranking_scores(self, ranker_name: str, scores: Dict[int, float],
                           candidates: List[int]):
        """Log individual ranker scores."""
        if not self.current_query_data:
            return

        if "ranking" not in self.current_query_data:
            self.current_query_data["ranking"] = {}

        # Convert scores to ranks (1-based)
        sorted_candidates = sorted(candidates, key=lambda i: scores.get(i, 0.0), reverse=True)
        ranks = {idx: rank + 1 for rank, idx in enumerate(sorted_candidates)}

        ranking_data = {
            "scores": scores,
            "ranks": ranks,
            "stats": {
                "min_score": min(scores.values()) if scores else None,
                "max_score": max(scores.values()) if scores else None,
                "avg_score": sum(scores.values()) / len(scores) if scores else None,
                "nonzero_scores": len([s for s in scores.values() if s > 0]) if scores else 0
            }
        }
        self.current_query_data["ranking"][ranker_name] = ranking_data

    def log_ensemble_result(self, final_ranking: List[int], ensemble_method: str,
                            weights: Dict[str, float]):
        """Log ensemble fusion results."""
        if not self.current_query_data:
            return

        ensemble_data = {
            "method": ensemble_method,
            "weights": weights,
            "final_ranking": final_ranking,
            "top_5_indices": final_ranking[:5]
        }
        self.current_query_data["ensemble"] = ensemble_data

    def log_chunks_used(self, chunk_indices: List[int], chunks: List[str],
                        sources: List[str]):
        """Log details about chunks selected for generation."""
        if not self.current_query_data:
            return

        chunks_data = []
        for i, idx in enumerate(chunk_indices):
            chunk_info = {
                "rank": i + 1,
                "global_index": idx,
                "source": sources[idx] if idx < len(sources) else "unknown",
                "char_length": len(chunks[idx]) if idx < len(chunks) else 0,
                "word_count": len(chunks[idx].split()) if idx < len(chunks) else 0,
                "has_table": "<table>" in chunks[idx].lower() if idx < len(chunks) else False,
                "preview": (chunks[idx][:200] + "...") if idx < len(chunks) and len(chunks[idx]) > 200 else chunks[
                    idx] if idx < len(chunks) else ""
            }
            chunks_data.append(chunk_info)

        self.current_query_data["chunks_used"] = chunks_data

    def log_generation(self, response: str, generation_params: Dict[str, Any],
                       prompt_length_estimate: Optional[int] = None):
        """Log generation parameters and results."""
        if not self.current_query_data:
            return

        generation_data = {
            "parameters": generation_params,
            "prompt_length_estimate": prompt_length_estimate,
            "response_char_length": len(response),
            "response_word_count": len(response.split()),
            "response_preview": (response[:300] + "...") if len(response) > 300 else response,
            "response_full": response  # Store full response for analysis
        }
        self.current_query_data["generation"] = generation_data

    def log_query_complete(self, total_time_seconds: Optional[float] = None):
        """Finalize and write the current query log."""
        if not self.current_query_data:
            return

        self.current_query_data["completed_at"] = datetime.now().isoformat()
        if total_time_seconds:
            self.current_query_data["total_time_seconds"] = total_time_seconds

        self._write_log(self.current_query_data)
        self.current_query_data = {}  # Reset for next query

    def log_error(self, error: Exception, context: str = ""):
        """Log errors during processing."""
        error_data = {
            "event": "error",
            "timestamp": datetime.now().isoformat(),
            "query_id": self.query_count,
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "current_query": self.current_query_data.get("query", "")
        }
        self._write_log(error_data)

    def _write_log(self, data: Dict[str, Any]):
        """Write a log entry to the JSONL file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        if not self.log_file.exists():
            return {}

        queries_processed = 0
        total_chunks_retrieved = 0
        avg_response_length = 0
        ranker_usage = {}

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("event") == "query":
                        queries_processed += 1

                        # Accumulate stats
                        if "chunks_used" in entry:
                            total_chunks_retrieved += len(entry["chunks_used"])

                        if "generation" in entry:
                            avg_response_length += entry["generation"]["response_char_length"]

                        if "ranking" in entry:
                            for ranker in entry["ranking"]:
                                ranker_usage[ranker] = ranker_usage.get(ranker, 0) + 1

                except json.JSONDecodeError:
                    continue

        return {
            "session_id": self.session_id,
            "queries_processed": queries_processed,
            "avg_chunks_per_query": total_chunks_retrieved / max(1, queries_processed),
            "avg_response_length": avg_response_length / max(1, queries_processed),
            "ranker_usage": ranker_usage,
            "log_file": str(self.log_file)
        }



def init_logger(cfg: QueryPlanConfig):
    global INSTANCE
    INSTANCE = RunLogger(cfg)

def load_session_logs(session_id: str) -> List[Dict[str, Any]]:
    """Load all log entries for a session."""
    log_file = Path("logs") / f"run_{session_id}.jsonl"
    if not log_file.exists():
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return logs


def analyze_logs(session_id: str) -> Dict[str, Any]:
    """Analyze logs to provide insights about retrieval and ranking performance."""
    logs = load_session_logs(session_id)
    queries = [log for log in logs if log.get("event") == "query"]

    if not queries:
        return {"error": "No query logs found"}

    analysis = {
        "total_queries": len(queries),
        "ranker_performance": {},
        "retrieval_stats": {},
        "generation_stats": {}
    }

    # Analyze ranker consistency and performance
    for query_log in queries:
        if "ranking" in query_log:
            for ranker, data in query_log["ranking"].items():
                if ranker not in analysis["ranker_performance"]:
                    analysis["ranker_performance"][ranker] = {
                        "avg_score": 0,
                        "score_variance": [],
                        "usage_count": 0
                    }

                scores = list(data["scores"].values())
                if scores:
                    analysis["ranker_performance"][ranker]["avg_score"] += sum(scores) / len(scores)
                    analysis["ranker_performance"][ranker]["score_variance"].extend(scores)
                analysis["ranker_performance"][ranker]["usage_count"] += 1

    # Finalize ranker analysis
    for ranker in analysis["ranker_performance"]:
        count = analysis["ranker_performance"][ranker]["usage_count"]
        if count > 0:
            analysis["ranker_performance"][ranker]["avg_score"] /= count

        variance_data = analysis["ranker_performance"][ranker]["score_variance"]
        if variance_data:
            import statistics
            analysis["ranker_performance"][ranker]["score_std"] = statistics.stdev(variance_data)
            analysis["ranker_performance"][ranker]["score_range"] = [min(variance_data), max(variance_data)]

        del analysis["ranker_performance"][ranker]["score_variance"]  # Clean up

    return analysis