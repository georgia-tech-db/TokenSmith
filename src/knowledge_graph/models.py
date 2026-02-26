from dataclasses import dataclass, field
from typing import Any
from enum import Enum


@dataclass
class Chunk:
    id: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    chunk_id: int
    nodes: list[str] = field(default_factory=list)


@dataclass
class QueryFeatures:
    E_q: int = 0
    C: int = 0
    L_max: int = 0
    L_avg: float = 0.0
    D_avg: float = 0.0
    D_max: int = 0
    N_sub: int = 0
    M_sub: int = 0
    Doc_count: int = 0

    def to_dict(self) -> dict:
        return {
            "E_q": self.E_q,
            "C": self.C,
            "L_max": self.L_max,
            "L_avg": self.L_avg,
            "D_avg": self.D_avg,
            "D_max": self.D_max,
            "N_sub": self.N_sub,
            "M_sub": self.M_sub,
            "Doc_count": self.Doc_count,
        }


class DifficultyCategory(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyComponents:
    s1_multihop: int
    s2_fragmentation: int
    s3_subgraph_size: int
    s4_branching: int
    s5_dispersion: int

    def to_dict(self) -> dict:
        return {
            "s1_multihop": self.s1_multihop,
            "s2_fragmentation": self.s2_fragmentation,
            "s3_subgraph_size": self.s3_subgraph_size,
            "s4_branching": self.s4_branching,
            "s5_dispersion": self.s5_dispersion,
        }


@dataclass
class DifficultyScore:
    score: int
    category: DifficultyCategory
    components: DifficultyComponents

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "category": self.category.value,
            "components": self.components.__dict__,
        }


@dataclass
class QueryAnalysisResult:
    query: str
    features: QueryFeatures
    difficulty: DifficultyScore

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "features": self.features.to_dict(),
            "difficulty": self.difficulty.to_dict(),
        }
