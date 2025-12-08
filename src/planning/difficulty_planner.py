import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import QueryPlanConfig
from src.preprocessing.chunking import SectionRecursiveConfig
from copy import deepcopy
import re
from typing import Dict, Any

from src.planning.planner import QueryPlanner
from src.planning.heuristics import HeuristicQueryPlanner


class QueryDifficultyPlanner(QueryPlanner):
    """
    Two-Pipeline Query Planner
    --------------------------
    Routes queries to either "easy" or "hard" pipelines based on complexity indicators.
    
    Easy Pipeline: Simple, direct queries that benefit from fast, keyword-based retrieval
    Hard Pipeline: Complex queries that need comprehensive semantic understanding
    """
    
    @property
    def name(self) -> str:
        return "DifficultyBasedPlanner"

    def __init__(self, base_cfg: QueryPlanConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)
        self.heuristic_planner = HeuristicQueryPlanner(base_cfg)

    def classify_difficulty(self, query: str) -> str:
        """
        Classify query as 'easy' or 'hard' based on comprehensive complexity indicators.
        
        Easy queries (Level 1-3):
        - Simple definitions and fact recall
        - Basic concept explanations
        - Questions requiring some reasoning
        - Combining multiple concepts
        - Moderate complexity explanations
        
        Hard queries (Level 4-5):
        - Multi-step reasoning and problem-solving
        - Complex design and architecture questions
        - Advanced analysis and trade-offs
        - Open-ended questions requiring deep understanding
        """
        q = query.lower().strip()
        
        # Length-based indicators
        word_count = len(q.split())
        char_count = len(q)
        
        # VERY EASY indicators (Level 1)
        very_easy_indicators = [
            r'\b(?:what is|what does|what are)\b.*\?$',  # Simple definition questions
            r'\b(?:is|are|was|were|do|does|did|can|could|will|would)\b.*\?$',  # Yes/no questions
            r'\b(?:define|definition|meaning|stands for)\b',  # Direct definition requests
            r'\b(?:capital|name|who|when|where)\b.*\?$',  # Simple fact recall
        ]
        
        # EASY indicators (Level 2)
        easy_indicators = [
            r'\b(?:how do you|how to|how can you)\b',  # Simple procedural questions
            r'\b(?:create|make|build|write|declare)\b',  # Simple action requests
            r'\b(?:example|examples|instance)\b',  # Example requests
            r'\b(?:basic|simple|fundamental)\b',  # Basic concept indicators
            r'\b(?:chapter|section|ch\.|sec\.)\b',  # Location-based queries
        ]
        
        # MEDIUM indicators (Level 3)
        medium_indicators = [
            r'\b(?:advantages|disadvantages|pros|cons|benefits|drawbacks)\b',  # Trade-off analysis
            r'\b(?:implement|implementation|algorithm|method|approach)\b',  # Implementation questions
            r'\b(?:difference|differences|distinguish|distinction)\b',  # Comparison questions
            r'\b(?:explain|explanation|describe|description)\b',  # Explanatory questions
            r'\b(?:why|because|reason|reasons|cause|causes)\b',  # Reasoning questions
            r'\b(?:memory|performance|optimization|efficiency)\b',  # Technical analysis
            r'\b(?:debug|debugging|troubleshoot|error|exception)\b',  # Problem-solving
            r'\b(?:binary search|sorting|recursion|recursive)\b',  # Specific algorithms
            r'\b(?:data structure|data structures)\b',  # Data structure questions
            r'\b(?:__init__|constructor|initialization)\b',  # Object-oriented concepts
        ]
        
        # HARD indicators (Level 4-5)
        hard_indicators = [
            # Complex design and architecture
            r'\b(?:design|architecture|system|framework|platform)\b',
            r'\b(?:scalable|distributed|microservices|monolithic)\b',
            r'\b(?:concurrent|parallel|threading|asynchronous)\b',
            r'\b(?:security|authentication|authorization|encryption)\b',
            
            # Multi-step reasoning
            r'\b(?:compare and contrast|analyze|evaluate|assess)\b',
            r'\b(?:trade-offs|tradeoffs|trade offs)\b',
            r'\b(?:migrate|migration|refactor|refactoring)\b',
            r'\b(?:best practices|guidelines|standards|patterns)\b',
            
            # Complex procedural
            r'\b(?:process|procedure|workflow|pipeline)\b',
            r'\b(?:steps|stages|phases|iterations)\b',
            r'\b(?:integration|interaction|combination)\b',
            
            # Advanced concepts
            r'\b(?:paradigm|paradigms|methodology|methodologies)\b',
            r'\b(?:constraints|limitations|requirements|specifications)\b',
            r'\b(?:deployment|production|environment|infrastructure)\b',
        ]
        
        # VERY HARD indicators (Level 5)
        very_hard_indicators = [
            r'\b(?:research|research-level|advanced|cutting-edge)\b',
            r'\b(?:open-ended|open ended|exploratory|investigative)\b',
            r'\b(?:consensus|distributed consensus|consensus algorithm)\b',
            r'\b(?:fault-tolerant|fault tolerant|resilient|robust)\b',
            r'\b(?:real-time|real time|streaming|event-driven)\b',
            r'\b(?:machine learning|ml|ai|artificial intelligence)\b',
            r'\b(?:quantum|blockchain|cryptocurrency|decentralized)\b',
        ]
        
        # Count indicators for each difficulty level
        very_easy_score = sum(1 for pattern in very_easy_indicators if re.search(pattern, q))
        easy_score = sum(1 for pattern in easy_indicators if re.search(pattern, q))
        medium_score = sum(1 for pattern in medium_indicators if re.search(pattern, q))
        hard_score = sum(1 for pattern in hard_indicators if re.search(pattern, q))
        very_hard_score = sum(1 for pattern in very_hard_indicators if re.search(pattern, q))
        
        # Length-based scoring
        if word_count <= 5 and char_count <= 30:
            very_easy_score += 1
        elif word_count <= 10 and char_count <= 60:
            easy_score += 1
        elif word_count <= 20 and char_count <= 120:
            medium_score += 1
        elif word_count > 20 or char_count > 120:
            hard_score += 1
        
        # Complexity indicators
        if q.count('?') > 1 or ';' in q or ':' in q:
            medium_score += 1
        
        # Multiple clauses
        clause_indicators = ['what', 'how', 'why', 'when', 'where', 'which']
        clause_count = sum(1 for indicator in clause_indicators if indicator in q)
        if clause_count > 1:
            medium_score += 1
        
        # Multiple concepts (indicated by multiple technical terms)
        technical_terms = ['algorithm', 'data structure', 'function', 'class', 'object', 'method', 'variable', 'loop', 'condition', 'exception', 'database', 'api', 'framework', 'library']
        tech_term_count = sum(1 for term in technical_terms if term in q)
        if tech_term_count > 2:
            hard_score += 1
        elif tech_term_count > 1:
            medium_score += 1
        
        # Decision logic with weighted scoring - binary classification
        total_easy = very_easy_score + easy_score + medium_score  # Combine levels 1-3 into easy
        total_hard = hard_score + very_hard_score  # Combine levels 4-5 into hard
        
        # Determine difficulty based on highest score
        scores = {
            'easy': total_easy,
            'hard': total_hard
        }
        
        # Special handling for specific patterns - binary classification
        if re.search(r'\b(?:advantages|disadvantages)\b', q):
            scores['easy'] += 1  # Trade-off analysis is easy
        if re.search(r'\b(?:binary search|recursion|algorithm)\b', q):
            scores['easy'] += 1  # Algorithm questions are easy
        if re.search(r'\b(?:difference|differences)\b', q):
            scores['easy'] += 1  # Comparison questions are easy
        if re.search(r'\b(?:design|architecture|scalable|microservices)\b', q):
            scores['hard'] += 2  # Strong hard indicator
        if re.search(r'\b(?:compare and contrast)\b', q):
            scores['hard'] += 2  # Strong hard indicator
        if re.search(r'\b(?:implement.*algorithm|algorithm.*implement)\b', q):
            scores['easy'] += 1  # Implementation of algorithms is easy
        if re.search(r'\b(?:difference.*between|between.*difference)\b', q):
            scores['easy'] += 1  # "difference between" is easy
        if re.search(r'\b(?:rest|graphql|api|apis)\b', q):
            scores['easy'] += 1  # API questions are easy
        
        # If no clear indicators, use length as tiebreaker
        if max(scores.values()) == 0:
            if word_count <= 15:
                return "easy"
            else:
                return "hard"
        
        # Return the difficulty level with the highest score
        return max(scores, key=scores.get)

    def get_easy_pipeline_config(self, query: str, base_cfg: QueryPlanConfig) -> QueryPlanConfig:
        """
        Easy Pipeline Configuration:
        - Optimized for speed and direct keyword matching
        - BM25-heavy ranking for exact matches
        - Reduced pool size for efficiency
        Note: Chunk config is not changed as artifacts are pre-loaded
        """
        cfg = deepcopy(base_cfg)
        
        # Easy pipeline optimizations
        cfg.pool_size = min(cfg.pool_size, 30)  # Reduced pool for speed
        cfg.top_k = min(cfg.top_k, 3)  # Fewer results needed
        
        # BM25-heavy ranking for keyword matching
        # Preserve existing ranker weights structure, just adjust faiss/bm25
        new_weights = {"faiss": 0.2, "bm25": 0.8}
        # Preserve other ranker weights if they exist (e.g., index_keywords)
        for key in cfg.ranker_weights:
            if key not in new_weights:
                new_weights[key] = cfg.ranker_weights[key]
        cfg.ranker_weights = new_weights
        
        return cfg

    def get_hard_pipeline_config(self, query: str, base_cfg: QueryPlanConfig) -> QueryPlanConfig:
        """
        Hard Pipeline Configuration:
        - Optimized for comprehensive understanding
        - FAISS-heavy ranking for semantic similarity
        - Larger pool size for thorough search
        Note: Chunk config is not changed as artifacts are pre-loaded
        """
        cfg = deepcopy(base_cfg)
        
        # Hard pipeline optimizations
        cfg.pool_size = max(cfg.pool_size, 80)  # Larger pool for comprehensive search
        cfg.top_k = max(cfg.top_k, 7)  # More results for complex queries
        
        # FAISS-heavy ranking for semantic understanding
        # Preserve existing ranker weights structure, just adjust faiss/bm25
        new_weights = {"faiss": 0.7, "bm25": 0.3}
        # Preserve other ranker weights if they exist (e.g., index_keywords)
        for key in cfg.ranker_weights:
            if key not in new_weights:
                new_weights[key] = cfg.ranker_weights[key]
        cfg.ranker_weights = new_weights
        
        return cfg

    def plan(self, query: str) -> QueryPlanConfig:
        """
        Main planning method that routes to appropriate pipeline.
        """
        difficulty = self.classify_difficulty(query)
        
        if difficulty == "easy":
            cfg = self.get_easy_pipeline_config(query, self.base_cfg)
            cfg._pipeline_type = "easy"
        else:
            cfg = self.get_hard_pipeline_config(query, self.base_cfg)
            cfg._pipeline_type = "hard"
        
        # Log the decision
        self._log_decision(cfg, extra_info={"difficulty": difficulty, "pipeline": cfg._pipeline_type})
        
        return cfg

    def _log_decision(self, cfg: QueryPlanConfig, extra_info: Dict[str, Any] = None):
        """Log the planning decision with pipeline information."""
        info = {
            "planner": self.name,
            "pipeline_type": getattr(cfg, '_pipeline_type', 'unknown'),
            "chunk_config": cfg.chunk_config.to_string(),
            "ranker_weights": cfg.ranker_weights,
            "pool_size": cfg.pool_size,
            "top_k": cfg.top_k,
        }
        
        if extra_info:
            info.update(extra_info)
        
        print(f"[PLANNER] {info}")
