from abc import ABC, abstractmethod
from typing import List

from src.knowledge_graph.models import Chunk


class BaseDivider(ABC):
    """Split raw text into a list of Chunks."""

    @abstractmethod
    def divide(self, text: str) -> List[Chunk]:
        """Split *text* into semantically coherent chunks.

        Args:
            text: The full raw text of the corpus.

        Returns:
            A list of :class:`Chunk` instances with unique, sequential IDs.
        """
        ...
