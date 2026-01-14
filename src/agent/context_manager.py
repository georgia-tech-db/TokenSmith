"""
Context registry for managing observations during agent investigation.
"""

from typing import Dict, List, Optional


class ContextRegistry:
    """Keyed registry for agent observations. Each tool execution returns a ref_id."""

    def __init__(self):
        self._observations: Dict[str, str] = {}
        self._counter: int = 0

    def add_observation(self, text: str) -> str:
        """Add an observation and return its ref_id."""
        self._counter += 1
        ref_id = f"obs_{self._counter}"
        self._observations[ref_id] = text
        return ref_id

    def get(self, ref_id: str) -> Optional[str]:
        """Get a single observation by ref_id."""
        return self._observations.get(ref_id)

    def get_context(self, keep_ids: List[str]) -> str:
        """Return concatenated context for the specified ref_ids."""
        parts = []
        for ref_id in keep_ids:
            if ref_id in self._observations:
                parts.append(f"[{ref_id}]\n{self._observations[ref_id]}")
        return "\n\n".join(parts)

    def prune(self, discard_ids: List[str]) -> None:
        """Remove observations by ref_id."""
        for ref_id in discard_ids:
            self._observations.pop(ref_id, None)

    def list_ids(self) -> List[str]:
        """Return all current observation ref_ids."""
        return list(self._observations.keys())

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()
        self._counter = 0

    def __len__(self) -> int:
        return len(self._observations)

