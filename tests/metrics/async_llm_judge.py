from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
import json
from tests.metrics.base import MetricBase


class AsyncLLMJudgeMetric(MetricBase):
    """
    Async LLM Judge metric that logs Q&A pairs during test run
    and processes them in a single batch at the end.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the Async LLM Judge metric.
        
        Args:
            log_dir: Directory to store test history. If None, uses logs/<timestamp>
        """
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs") / timestamp
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "test_history.txt"
        self.results_file = self.log_dir / "async_llm_results.json"
        
        # Clear previous history
        if self.history_file.exists():
            self.history_file.unlink()
    
    @property
    def name(self) -> str:
        return "async_llm_judge"
    
    @property
    def weight(self) -> float:
        # Weight is 0 because we don't include this in immediate scoring
        return 0.0
    
    def is_available(self) -> bool:
        """Always available since it just logs to a file."""
        return True
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """
        Log the Q&A pair to file and return 0 (not included in immediate scoring).
        
        Args:
            answer: The generated answer
            expected: The question (passed as 'expected' in scorer for llm_judge)
            keywords: Optional keywords (not used)
            
        Returns:
            float: Always 0.0 since this doesn't contribute to immediate scoring
        """
        # Log to file
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": expected,
            "answer": answer,
        }
        
        # Append to history file (JSONL format)
        with open(self.history_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
        
        # Return 0 - doesn't contribute to immediate score
        return 0.0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', log_dir='{self.log_dir}')"


def batch_grade_all(log_dir: Optional[Path] = None, doc_url: Optional[str] = None) -> Dict[str, Dict]:
    """
    Process all logged Q&A pairs in a single batch.
    
    Args:
        log_dir: Directory containing test_history.txt
        doc_url: URL to reference document (PDF)
        
    Returns:
        dict: Mapping of question to grading result
    """
    # Find the most recent log directory if not specified
    if log_dir is None:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("No logs directory found")
            return {}
        
        # Get most recent timestamped directory
        subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("No log directories found")
            return {}
        
        log_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        print(f"Using most recent log directory: {log_dir}")
    
    log_dir = Path(log_dir)
    history_file = log_dir / "test_history.txt"
    results_file = log_dir / "async_llm_results.json"
    
    if not history_file.exists():
        print(f"No test history found at {history_file}")
        return {}
    
    # Load all Q&A pairs
    qa_pairs = []
    with open(history_file) as f:
        for line in f:
            if line.strip():
                qa_pairs.append(json.loads(line))
    
    if not qa_pairs:
        print("No Q&A pairs found in history")
        return {}
    
    print(f"\n{'='*60}")
    print(f"Batch grading {len(qa_pairs)} Q&A pairs...")
    print(f"{'='*60}\n")
    
    # Import LLM judge components
    try:
        from google import genai
        from google.genai import types
        import httpx
        from tests.metrics.llm_judge import GradingResult
    except ImportError as e:
        print(f"Cannot import LLM judge dependencies: {e}")
        return {}
    
    # Initialize client and load document
    doc_url = doc_url or "https://my.uopeople.edu/pluginfile.php/57436/mod_book/chapter/37620/Database%20System%20Concepts%204th%20Edition%20By%20Silberschatz-Korth-Sudarshan.pdf"
    
    try:
        client = genai.Client()
        print("Downloading reference document...")
        doc_data = httpx.get(doc_url).content
        print("Document loaded.\n")
    except Exception as e:
        print(f"Failed to initialize LLM judge: {e}")
        return {}
    
    # Grade all Q&A pairs sequentially
    results = {}
    
    for i, entry in enumerate(qa_pairs, 1):
        question = entry["question"]
        answer = entry["answer"]
        
        print(f"[{i}/{len(qa_pairs)}] Grading: {question[:60]}...")
        
        try:
            prompt = _build_grading_prompt(question, answer)
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=doc_data,
                        mime_type='application/pdf',
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GradingResult,
                )
            )
            
            # Parse structured output
            grading = GradingResult.model_validate_json(response.text)
            normalized_score = (grading.score - 1) / 4.0
            
            results[question] = {
                "score": grading.score,
                "normalized_score": normalized_score,
                "accuracy": grading.accuracy,
                "completeness": grading.completeness,
                "clarity": grading.clarity,
                "overall_reasoning": grading.overall_reasoning,
                "answer": answer,
                "timestamp": entry["timestamp"]
            }
            
            print(f"  ✓ Score: {grading.score}/5 ({normalized_score:.3f})")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[question] = {
                "error": str(e),
                "answer": answer,
                "timestamp": entry["timestamp"]
            }
    
    # Save results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch grading complete!")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}\n")
    
    # Print summary
    _print_summary(results)
    
    return results


def _build_grading_prompt(question: str, answer: str) -> str:
    """Build the grading prompt."""
    return f"""You are an expert evaluator for a database textbook Q&A system. Your task is to grade the quality of answers generated by an LLM pipeline.

**Reference Material:** The attached PDF is the authoritative source textbook on database systems.

**Evaluation Task:**
Question: {question}

Generated Answer: {answer}

**Grading Criteria:**
Evaluate the answer on the following dimensions:

1. **Accuracy (40%)**: Are the facts, concepts, and technical details correct according to the textbook?
2. **Completeness (30%)**: Does the answer fully address all aspects of the question?
3. **Clarity (20%)**: Is the answer well-organized, coherent, and easy to understand?
4. **Relevance (10%)**: Does the answer stay focused on the question without unnecessary tangents?

**Rating Scale:**
- 5 (Excellent): Highly accurate, complete, and clear; demonstrates deep understanding
- 4 (Good): Mostly accurate and complete with minor gaps or clarity issues
- 3 (Satisfactory): Correct core concepts but missing important details or has clarity problems
- 2 (Poor): Contains significant errors, omissions, or confusion
- 1 (Unacceptable): Fundamentally incorrect, irrelevant, or fails to address the question

**Instructions:**
- Base your evaluation ONLY on the attached textbook content
- Provide specific, actionable feedback
- Be fair but rigorous in your assessment"""


def _print_summary(results: Dict[str, Dict]):
    """Print summary of grading results."""
    successful = [r for r in results.values() if "error" not in r]
    failed = [r for r in results.values() if "error" in r]
    
    if successful:
        scores = [r["score"] for r in successful]
        avg_score = sum(scores) / len(scores)
        
        print(f"Successfully graded: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"\nAverage score: {avg_score:.2f}/5")
        print(f"Score distribution:")
        for score in [5, 4, 3, 2, 1]:
            count = sum(1 for s in scores if s == score)
            bar = "█" * count
            print(f"  {score}/5: {bar} ({count})")


def print_results(log_dir: Optional[Path] = None):
    """
    Print formatted results from async LLM grading.
    
    Args:
        log_dir: Directory containing async_llm_results.json
    """
    # Find the most recent log directory if not specified
    if log_dir is None:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("No logs directory found")
            return
        
        subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        if not subdirs:
            print("No log directories found")
            return
        
        log_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    log_dir = Path(log_dir)
    results_file = log_dir / "async_llm_results.json"
    
    if not results_file.exists():
        print(f"No results found at {results_file}")
        print("Run batch_grade_all() first to generate results.")
        return
    
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    if not results:
        print("No results found")
        return
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("ASYNC LLM JUDGE RESULTS")
    print(f"{'='*80}\n")
    
    for i, (question, result) in enumerate(results.items(), 1):
        print(f"[{i}] Question: {question}")
        print(f"{'─'*80}")
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  Score: {result['score']}/5 ({result['normalized_score']:.3f})")
            print(f"  Accuracy: {result['accuracy']}")
            print(f"  Completeness: {result['completeness']}")
            print(f"  Clarity: {result['clarity']}")
            print(f"  Overall: {result['overall_reasoning']}")
        
        print()
    
    # Print summary
    print(f"{'='*80}")
    _print_summary(results)
    print(f"{'='*80}\n")

