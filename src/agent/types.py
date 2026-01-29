from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class AgentConfig:
    reasoning_limit: int = 5
    tool_limit: int = 20
    max_reasoning_tokens: int = 500
    max_generation_tokens: int = 400
    max_context_tokens: int = 8000
    model_path: str = ""

@dataclass
class AgentStep:
    thought: str
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    context_action: Dict[str, Any]
    signal: str

@dataclass
class ChunkMetadata:
    chunk_id: int
    score: float
    source: str
    full_text: str
    
@dataclass
class ObservationMetadata:
    """Metadata tracking lifecycle of an observation."""
    added_in_step: Optional[int] = None
    removed_in_step: Optional[int] = None
    replaced_in_step: Optional[int] = None
    replaced_with: Optional[str] = None
    kept_in_final: bool = False
