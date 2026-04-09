import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class RunLogger:
    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

    def save_chat_log(
        self,
        query: str,
        chat_request_params: Optional[Dict[str, Any]],
        config_state: Dict[str, Any],
        top_idxs: List[int],
        chunks: List[str],
        sources: List[str],
        page_map: Dict[int, List[int]],
        full_response: str,
        top_k: int,
        ordered_scores: List[float],
        chunk_diagnostics: Optional[Dict[int, Dict[str, Any]]] = None,
        additional_log_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a unique JSON log file for this query.

        Args:
            query:             The user query.
            chat_request_params: Params used in the chat request.
            config_state:      Snapshot of RAGConfig at query time.
            top_idxs:          Ordered list of retrieved chunk indices (post-reranking).
            chunks:            Full chunk list (all chunks, not just top-k).
            sources:           Source filepath per chunk.
            page_map:          Mapping of chunk_idx -> list of page numbers.
            full_response:     The generated answer.
            top_k:             Number of chunks used.
            ordered_scores:    Post-fusion RRF scores, aligned with top_idxs.
            chunk_diagnostics: Optional dict keyed by chunk_idx with per-retriever
                               diagnostic fields:
                               {
                                 idx: {
                                   "faiss_score": float,
                                   "faiss_rank": int,
                                   "bm25_score": float,
                                   "bm25_rank": int,
                                   "post_fusion_rank": int,
                                   "post_reranking_rank": int,
                                   "cross_encoder_score": float,
                                 }
                               }
            additional_log_info: Any extra fields to merge into the top-level log.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_id = f"chat_{timestamp_str}"

        page_numbers_list = [page_map.get(i, [1]) for i in top_idxs]

        lengths_match = (
            len(chunks) == len(top_idxs) == len(sources)
            == len(ordered_scores) == len(page_numbers_list)
        )

        if not lengths_match:
            print("Warning: Lengths of chunks, top_idxs, sources, ordered_scores, "
                  "and page_numbers do not match. Defaulting to long-form logging.")
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "chat_request_params": chat_request_params,
                "config_state": config_state,
                "top_k": top_k,
                "ordered_scores": ordered_scores[:len(top_idxs)],
                "top_idxs": top_idxs,
                "chunks": chunks[:len(top_idxs)],
                "sources": sources[:len(top_idxs)],
                "page_numbers": [page_map.get(i, [1]) for i in top_idxs],
                "full_response": full_response,
            }
        else:
            retrieved_chunks = []
            for rank, (chunk_text, idx, source, score, page_numbers) in enumerate(
                zip(chunks, top_idxs, sources, ordered_scores, page_numbers_list), start=1
            ):
                entry = {
                    "rank": rank,
                    "idx": idx,
                    "chunk": chunk_text,
                    "source": source,
                    "page_number": page_numbers,
                    # Post-fusion RRF score — always present
                    "post_fusion_score": score,
                }

                # Merge per-chunk diagnostics if provided
                if chunk_diagnostics and idx in chunk_diagnostics:
                    diag = chunk_diagnostics[idx]
                    entry["faiss_score"]           = diag.get("faiss_score", None)
                    entry["faiss_rank"]            = diag.get("faiss_rank", None)
                    entry["bm25_score"]            = diag.get("bm25_score", None)
                    entry["bm25_rank"]             = diag.get("bm25_rank", None)
                    entry["post_fusion_rank"]      = diag.get("post_fusion_rank", None)
                    entry["post_reranking_rank"]   = diag.get("post_reranking_rank", None)
                    entry["cross_encoder_score"]   = diag.get("cross_encoder_score", None)

                retrieved_chunks.append(entry)

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "chat_request_params": chat_request_params,
                "config_state": config_state,
                "top_k": top_k,
                "retrieved_chunks": retrieved_chunks,
                "full_response": full_response,
            }

        if additional_log_info:
            for key, value in additional_log_info.items():
                if key in log_data:
                    print(f"Warning: Key '{key}' in additional_log_info conflicts "
                          f"with existing log key. Skipping.")
                else:
                    log_data[key] = value

        log_file = self.logs_dir / f"{log_id}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4, cls=NpEncoder)


# Global instance
_INSTANCE = None

def get_logger():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RunLogger()
    return _INSTANCE