import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import QueryPlanConfig
from chunking import CharChunkConfig, TokenChunkConfig, SlidingTokenConfig, SectionChunkConfig
from copy import deepcopy
import re
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime

from planning.planner import QueryPlanner
from planning.heuristics import HeuristicQueryPlanner
from planning.difficulty_planner import QueryDifficultyPlanner
from generator import run_llama_cpp


class ComparisonPlanner(QueryPlanner):
    """
    Comparison Query Planner
    ------------------------
    Compares regex-based vs Qwen model-based difficulty classification.
    
    This planner runs both classification methods and logs the results
    for analysis and comparison.
    """
    
    @property
    def name(self) -> str:
        return "ComparisonPlanner"

    def __init__(self, base_cfg: QueryPlanConfig):
        super().__init__(base_cfg)
        self.base_cfg = deepcopy(base_cfg)
        self.difficulty_planner = QueryDifficultyPlanner(base_cfg)
        self.comparison_results = []
        self.model_path = base_cfg.model_path

    def classify_difficulty_regex(self, query: str) -> str:
        """
        Use the existing regex-based classification from QueryDifficultyPlanner.
        """
        return self.difficulty_planner.classify_difficulty(query)

    def classify_difficulty_qwen(self, query: str) -> str:
        """
        Use Qwen model to classify question difficulty as easy or hard.
        
        Returns 'easy' or 'hard' based on the model's binary classification.
        """
        prompt = self._create_difficulty_prompt(query)
        
        try:
            response = run_llama_cpp(
                prompt=prompt,
                model_path=self.model_path,
                max_tokens=50,  # Short response expected
                temperature=0.1,  # Low temperature for consistent classification
                threads=4
            )
            
            # Parse the response to extract difficulty classification
            difficulty_classification = self._parse_difficulty_response(response)
            
            # Return the classification directly (should be "easy" or "hard")
            return difficulty_classification
                
        except Exception as e:
            print(f"Error in Qwen classification: {e}")
            # Fallback to regex classification
            return self.classify_difficulty_regex(query)

    def _create_difficulty_prompt(self, query: str) -> str:
        """
        Create a prompt for Qwen to classify question difficulty.
        """
        return f"""<|im_start|>system
You are an expert at analyzing question difficulty. Classify the following question as either EASY or HARD.

EASY questions are:
- Simple definitions or fact recall (e.g., "What is a database?")
- Basic concept explanations (e.g., "What is a primary key?")
- Straightforward comparisons (e.g., "What is the difference between X and Y?")
- Questions that can be answered with a single concept or definition
- Questions that require minimal reasoning or explanation

HARD questions are:
- Complex system design questions (e.g., "Design a distributed database system...")
- Questions requiring multi-step reasoning and problem-solving
- Questions involving multiple constraints, trade-offs, or conflicting requirements
- Questions about implementing complex algorithms or architectures
- Questions that require deep technical knowledge and analysis
- Questions with phrases like "design", "implement", "architect", "optimize", "handle", "maintain" combined with complex requirements
- Questions mentioning distributed systems, scalability, fault tolerance, consensus algorithms, Byzantine failures, network partitions, or similar advanced concepts

Look for keywords and complexity indicators:
- HARD indicators: "design", "implement", "architect", "distributed", "scalable", "fault-tolerant", "consensus", "Byzantine", "network partitions", "multi-step", "optimize", "handle", "maintain", "guarantee", "ensure"
- EASY indicators: "what is", "what does", "explain", "define", "difference between", simple comparisons

Respond with ONLY "EASY" or "HARD" followed by a brief explanation.
<|im_end|>
<|im_start|>user
Question: {query}
<|im_end|>
<|im_start|>assistant
"""

    def _parse_difficulty_response(self, response: str) -> str:
        """
        Parse the Qwen response to extract the difficulty classification.
        """
        # Extract the assistant's response (after "assistant" marker)
        # Look for the pattern "assistant\n" or "assistant " followed by the response
        assistant_match = re.search(r'assistant\s*\n\s*(.*?)(?:\[end of text\]|$)', response, re.DOTALL | re.IGNORECASE)
        if assistant_match:
            assistant_response = assistant_match.group(1).strip()
        else:
            # If we can't find the assistant marker, use the whole response
            assistant_response = response
        
        # Look for "HARD" first (more specific), then "EASY"
        if re.search(r'\bHARD\b', assistant_response, re.IGNORECASE):
            return "hard"
        elif re.search(r'\bEASY\b', assistant_response, re.IGNORECASE):
            return "easy"
        
        # Fallback: look for the pattern "X [end of text]" for old format
        match = re.search(r'(\d)\s*\[end of text\]', response)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 3:
                return "easy"
            elif 4 <= level <= 5:
                return "hard"
        
        # Fallback: look for any number 1-5 in the entire response
        match = re.search(r'\b([1-5])\b', response)
        if match:
            level = int(match.group(1))
            if 1 <= level <= 3:
                return "easy"
            elif 4 <= level <= 5:
                return "hard"
        
        # If no valid classification found, return "easy" as default
        print(f"Warning: Could not parse difficulty from response. Assistant response: {assistant_response[:200]}...")
        return "easy"

    def compare_classifications(self, query: str) -> Dict[str, Any]:
        """
        Compare regex vs Qwen classifications for a single query.
        """
        start_time = time.time()
        
        # Get regex classification
        regex_result = self.classify_difficulty_regex(query)
        regex_time = time.time() - start_time
        
        # Get Qwen classification
        qwen_start = time.time()
        qwen_result = self.classify_difficulty_qwen(query)
        qwen_time = time.time() - qwen_start
        
        # Determine agreement
        agreement = regex_result == qwen_result
        
        result = {
            "query": query,
            "regex_classification": regex_result,
            "qwen_classification": qwen_result,
            "agreement": agreement,
            "regex_time_ms": round(regex_time * 1000, 2),
            "qwen_time_ms": round(qwen_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "query_length": len(query),
            "word_count": len(query.split())
        }
        
        self.comparison_results.append(result)
        return result

    def plan(self, query: str) -> QueryPlanConfig:
        """
        Main planning method that compares both approaches and uses regex result for actual planning.
        """
        # Compare classifications
        comparison = self.compare_classifications(query)
        
        # Use regex result for actual planning (as it's faster and more reliable)
        difficulty = comparison["regex_classification"]
        
        if difficulty == "easy":
            cfg = self.difficulty_planner.get_easy_pipeline_config(query, self.base_cfg)
            cfg._pipeline_type = "easy"
        else:
            cfg = self.difficulty_planner.get_hard_pipeline_config(query, self.base_cfg)
            cfg._pipeline_type = "hard"
        
        # Log the decision with comparison data
        self._log_decision(cfg, extra_info={
            "difficulty": difficulty,
            "pipeline": cfg._pipeline_type,
            "comparison": comparison
        })
        
        return cfg

    def get_comparison_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all comparison results.
        """
        if not self.comparison_results:
            return {"message": "No comparisons performed yet"}
        
        total_queries = len(self.comparison_results)
        agreements = sum(1 for r in self.comparison_results if r["agreement"])
        agreement_rate = agreements / total_queries if total_queries > 0 else 0
        
        # Count classifications
        regex_counts = {"easy": 0, "medium": 0, "hard": 0}
        qwen_counts = {"easy": 0, "medium": 0, "hard": 0}
        
        for result in self.comparison_results:
            regex_counts[result["regex_classification"]] += 1
            qwen_counts[result["qwen_classification"]] += 1
        
        # Calculate average times
        avg_regex_time = sum(r["regex_time_ms"] for r in self.comparison_results) / total_queries
        avg_qwen_time = sum(r["qwen_time_ms"] for r in self.comparison_results) / total_queries
        
        # Find disagreements
        disagreements = [r for r in self.comparison_results if not r["agreement"]]
        
        return {
            "total_queries": total_queries,
            "agreement_rate": round(agreement_rate, 3),
            "agreements": agreements,
            "disagreements": len(disagreements),
            "regex_classifications": regex_counts,
            "qwen_classifications": qwen_counts,
            "avg_regex_time_ms": round(avg_regex_time, 2),
            "avg_qwen_time_ms": round(avg_qwen_time, 2),
            "speed_ratio": round(avg_qwen_time / avg_regex_time, 2) if avg_regex_time > 0 else 0,
            "disagreement_examples": disagreements[:5]  # Show first 5 disagreements
        }

    def save_comparison_results(self, filename: str = None) -> str:
        """
        Save comparison results to a JSON file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_results_{timestamp}.json"
        
        data = {
            "summary": self.get_comparison_summary(),
            "detailed_results": self.comparison_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename

    def _log_decision(self, cfg: QueryPlanConfig, extra_info: Dict[str, Any] = None):
        """Log the planning decision with comparison information."""
        info = {
            "planner": self.name,
            "pipeline_type": getattr(cfg, '_pipeline_type', 'unknown'),
            "chunk_mode": cfg.chunk_mode,
            "ranker_weights": cfg.ranker_weights,
            "pool_size": cfg.pool_size,
            "top_k": cfg.top_k,
        }
        
        if extra_info:
            info.update(extra_info)
        
        if cfg.location_hint:
            info["location_hint"] = cfg.location_hint
        
        print(f"[COMPARISON_PLANNER] {info}")


def run_comparison_test(queries: List[str], config_path: str = "config/config.yaml") -> ComparisonPlanner:
    """
    Run a comparison test with a list of queries.
    
    Args:
        queries: List of query strings to test
        config_path: Path to configuration file
    
    Returns:
        ComparisonPlanner instance with results
    """
    from config import QueryPlanConfig
    
    # Load configuration
    cfg = QueryPlanConfig.from_yaml(config_path)
    
    # Create comparison planner
    planner = ComparisonPlanner(cfg)
    
    print(f"Running comparison test with {len(queries)} queries...")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}: {query}")
        result = planner.compare_classifications(query)
        print(f"  Regex: {result['regex_classification']} ({result['regex_time_ms']}ms)")
        print(f"  Qwen:  {result['qwen_classification']} ({result['qwen_time_ms']}ms)")
        print(f"  Agreement: {'✓' if result['agreement'] else '✗'}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    summary = planner.get_comparison_summary()
    print(f"Total queries: {summary['total_queries']}")
    print(f"Agreement rate: {summary['agreement_rate']:.1%}")
    print(f"Average regex time: {summary['avg_regex_time_ms']}ms")
    print(f"Average Qwen time: {summary['avg_qwen_time_ms']}ms")
    print(f"Speed ratio (Qwen/Regex): {summary['speed_ratio']:.1f}x")
    
    print(f"\nRegex classifications: {summary['regex_classifications']}")
    print(f"Qwen classifications: {summary['qwen_classifications']}")
    
    if summary['disagreements'] > 0:
        print(f"\nDisagreements ({summary['disagreements']}):")
        for i, d in enumerate(summary['disagreement_examples'], 1):
            print(f"  {i}. \"{d['query'][:60]}...\"")
            print(f"     Regex: {d['regex_classification']}, Qwen: {d['qwen_classification']}")
    
    # Save results
    filename = planner.save_comparison_results()
    print(f"\nResults saved to: {filename}")
    
    return planner
