from typing import Dict, List, Optional, Any
from src.agent.types import ObservationMetadata

class ContextBudgetExceeded(Exception):
    """Raised when adding an observation would exceed the max context budget."""
    pass

class ContextRegistry:
    """
    Keyed registry for agent observations. 
    Enforces a rough token budget (approx 3.5 chars per token).
    """

    def __init__(self, max_tokens: int = 8000):
        self._observations: Dict[str, str] = {}
        self._metadata: Dict[str, ObservationMetadata] = {}
        self._counter: int = 0
        self._max_tokens = max_tokens
        self._current_chars = 0

    @property
    def current_tokens(self) -> int:
        return int(self._current_chars / 3.5)

    @property
    def status(self) -> Dict[str, Any]:
        used = self.current_tokens
        return {
            "used": used,
            "total": self._max_tokens,
            "usage_percent": (used / self._max_tokens) * 100 if self._max_tokens > 0 else 0.0,
            "count": len(self._observations)
        }

    def _check_budget(self, text: str):
        new_tokens = int(len(text) / 3.5)
        if (self.current_tokens + new_tokens) > self._max_tokens:
            raise ContextBudgetExceeded(
                f"Cannot add observation ({new_tokens} tokens). "
                f"Registry full: {self.current_tokens}/{self._max_tokens} tokens used."
            )

    def add(self, text: str, step: Optional[int] = None) -> str:
        self._check_budget(text)
        self._counter += 1
        ref_id = f"obs_{self._counter}"
        self._observations[ref_id] = text
        self._metadata[ref_id] = ObservationMetadata(added_in_step=step)
        self._current_chars += len(text)
        return ref_id

    def remove(self, ref_id: str, step: Optional[int] = None) -> bool:
        if ref_id in self._observations:
            text = self._observations.pop(ref_id)
            if ref_id in self._metadata:
                self._metadata[ref_id].removed_in_step = step
            else:
                self._metadata[ref_id] = ObservationMetadata(removed_in_step=step)
            self._current_chars -= len(text)
            return True
        return False

    def replace(self, ref_id: str, new_text: str, step: Optional[int] = None) -> None:
        if ref_id not in self._observations:
            raise KeyError(f"Observation {ref_id} not found.")

        old_text = self._observations[ref_id]
        diff_chars = len(new_text) - len(old_text)
        new_total_tokens = int((self._current_chars + diff_chars) / 3.5)
        
        if new_total_tokens > self._max_tokens:
            raise ContextBudgetExceeded(
                f"Replacement exceeds budget. Total would be {new_total_tokens}/{self._max_tokens}."
            )

        self._observations[ref_id] = new_text
        if ref_id in self._metadata:
            self._metadata[ref_id].replaced_in_step = step
            self._metadata[ref_id].replaced_with = ref_id
        self._current_chars += diff_chars

    def get(self, ref_id: str) -> Optional[str]:
        return self._observations.get(ref_id)

    def list_ids(self) -> List[str]:
        return list(self._observations.keys())

    def clear(self) -> None:
        self._observations.clear()
        self._metadata.clear()
        self._counter = 0
        self._current_chars = 0

    def __len__(self) -> int:
        return len(self._observations)

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        result = {}
        all_ref_ids = set(self._observations.keys()) | set(self._metadata.keys())
        
        for ref_id in all_ref_ids:
            meta = self._metadata.get(ref_id, ObservationMetadata())
            lifecycle = []
            if meta.added_in_step is not None: lifecycle.append(f"added-in-step-{meta.added_in_step}")
            if meta.removed_in_step is not None: lifecycle.append(f"removed-in-step-{meta.removed_in_step}")
            if meta.replaced_in_step is not None: lifecycle.append(f"replaced-in-step-{meta.replaced_in_step}")
            if meta.kept_in_final: lifecycle.append("kept-in-final-content")
            
            result[ref_id] = {
                "content": self._observations.get(ref_id),
                "lifecycle": "; ".join(lifecycle) if lifecycle else "no-events",
                "added_in_step": meta.added_in_step,
                "removed_in_step": meta.removed_in_step,
                "replaced_in_step": meta.replaced_in_step,
                "kept_in_final": meta.kept_in_final,
            }
        return result
