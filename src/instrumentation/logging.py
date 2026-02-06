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

    def save_chat_log(self, 
                      query: str, 
                      chat_request_params : Optional[Dict[str, Any]],
                      ordered_scores: List[Union[float, int]],
                      config_state: Dict[str, Any],
                      top_idxs: List[int], 
                      chunks: List[str], 
                      sources: List[str], 
                      page_map: Dict[int, int], 
                      full_response: str,
                      top_k: int):
        """Creates a unique JSON file for this specific chat request."""
        
        # timestamp for filename: 20240520_143005 (Sorts newest to bottom, 
        # but if you want newest at top, most OS sort by name DESC)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_id = f"chat_{timestamp_str}"
        
        ### make list of dicts for retrieval 

        # make page_numbers list
        page_numbers_list = [page_map.get(i, 1) for i in top_idxs]
        # make sure chunks, top_idxs, sources, and ordered_scores are the same length
        if not (len(chunks) == len(top_idxs) == len(sources) == len(ordered_scores) == len(page_numbers_list)):
            print("Warning: Lengths of chunks, top_idxs, sources, ordered_scores, and page_numbers do not match.")
            print("Defaulting to long form logging ")
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "chat_request_params": chat_request_params,
                "config_state" : config_state,
                "top_k": top_k,
                "ordered_scores": ordered_scores,
                "top_idxs": top_idxs,
                "chunks": chunks,
                "sources": sources,
                "page_numbers": [page_map.get(i, 1) for i in top_idxs],
                "full_response": full_response
            }
        else:
            retrieved_chunks = []
            for i, (chunk, idx, source, score, page_number) in enumerate(zip(chunks, top_idxs, sources, ordered_scores, page_numbers_list)):
                retrieved_chunks.append({
                    "rank": i + 1,
                    "idx": idx,
                    "chunk": chunk,
                    "source": source,
                    "score": score,
                    "page_number": page_number
                })

            log_data = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "chat_request_params": chat_request_params,
                "config_state" : config_state,
                "top_k": top_k,
                "retrieved_chunks": retrieved_chunks,
                "full_response": full_response
            }

        log_file = self.logs_dir / f"{log_id}.json"
        
        # Write as a single pretty-printed JSON file
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4, cls=NpEncoder)

# Global Instance logic
_INSTANCE = None
def get_logger():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = RunLogger()
    return _INSTANCE




