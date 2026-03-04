from abc import ABC, abstractmethod
from typing import List, Any

from src.knowledge_graph.models import Chunk


class BaseDivider(ABC):
    """Split raw text into a list of Chunks."""

    def __init__(self):
        self.metadata: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of this divider."""
        return {
            "class": self.__class__.__name__,
        }

    @abstractmethod
    def divide(self, text: str) -> List[Chunk]:
        """Split *text* into semantically coherent chunks.

        Args:
            text: The full raw text of the corpus.

        Returns:
            A list of :class:`Chunk` instances with unique, sequential IDs.
        """
        ...
