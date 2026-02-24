"""Shared data models for the knowledge-graph pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    """A single text chunk produced by a Divider.

    Attributes:
        id: Unique index within the corpus.
        text: Raw text content.
        metadata: Optional metadata (e.g. page number, source document).
    """

    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Extraction output for a single chunk.

    Attributes:
        chunk_id: References ``Chunk.id``.
        nodes: Normalized node labels extracted from this chunk.
    """

    chunk_id: int
    nodes: List[str] = field(default_factory=list)
