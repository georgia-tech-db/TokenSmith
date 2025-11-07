from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types
import httpx
from tests.metrics.base import MetricBase
from tests.metrics.llm_judge import GradingResult

# Shared state for async grading
_results_lock = threading.Lock()
_grading_results: Dict[str, Dict] = {}
_executor: Optional[ThreadPoolExecutor] = None
_client = None
_doc_data = None
_initialized = False

# Rate limiting
_rate_limit_lock = threading.Lock()
_last_request_time = 0.0
_min_request_interval = 1.0  # Minimum 1 second between requests


class AsyncLLMJudgeMetric(MetricBase):
    """
    Async LLM Judge that spawns threads to grade answers in background.
    Results accumulate in shared dict and are included in final scoring.
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs") / timestamp
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.log_dir / "async_llm_results.json"
        
        # Initialize client once
        if not _initialized:
            _lazy_init()
    
    @property
    def name(self) -> str:
        return "async_llm_judge"
    
    @property
    def weight(self) -> float:
        return 0.0
    
    def is_available(self) -> bool:
        return True
    
    def calculate(self, answer: str, expected: str, keywords: Optional[List[str]] = None) -> float:
        """
        Submit grading task to thread pool. Return current score if available, else 0.0.
        
        Args:
            answer: Generated answer
            expected: Question
            keywords: Not used
            
        Returns:
            Current score from shared dict, or 0.0 if still grading
        """
        question = expected
        
        # Check if already graded or queued
        with _results_lock:
            if question in _grading_results:
                result = _grading_results[question]
                if "error" not in result:
                    return result["normalized_score"]
                return 0.0
        
        # Submit to thread pool
        if _executor:
            _executor.submit(_grade_one, question, answer)
        
        return 0.0
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def _lazy_init():
    """Initialize client, executor, and load document once."""
    global _client, _doc_data, _initialized, _executor
    
    if _initialized:
        return
    
    _client = genai.Client()
    doc_url = "https://my.uopeople.edu/pluginfile.php/57436/mod_book/chapter/37620/Database%20System%20Concepts%204th%20Edition%20By%20Silberschatz-Korth-Sudarshan.pdf"
    _doc_data = httpx.get(doc_url).content
    _executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="llm_judge")
    _initialized = True


def _perform_attempt(question: str, answer: str) -> Dict:
    """Perform a single grading attempt."""
    # Rate limiting: enforce minimum interval between requests
    with _rate_limit_lock:
        global _last_request_time
        elapsed = time.time() - _last_request_time
        if elapsed < _min_request_interval:
            time.sleep(_min_request_interval - elapsed)
        _last_request_time = time.time()
    
    prompt = _build_grading_prompt(question, answer)
    
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=_doc_data,
                mime_type='application/pdf',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GradingResult,
        )
    )
    
    grading = GradingResult.model_validate_json(response.text)
    normalized_score = (grading.score - 1) / 4.0
    
    return {
        "score": grading.score,
        "normalized_score": normalized_score,
        "accuracy": grading.accuracy,
        "completeness": grading.completeness,
        "clarity": grading.clarity,
        "overall_reasoning": grading.overall_reasoning,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    }


def _grade_one(question: str, answer: str):
    """Grade a single Q&A pair in background thread with retry logic."""
    if not _initialized:
        return
    
    max_retries = 3
    base_delay = 20.0 # 20 seconds
    
    for attempt in range(max_retries):
        try:
            result = _perform_attempt(question, answer)
            with _results_lock:
                _grading_results[question] = result
            return  # Success
            
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"⚠️  Rate limit hit, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            
            # Final attempt failed or non-rate-limit error
            with _results_lock:
                _grading_results[question] = {
                    "error": error_str,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat()
                }
            break


def wait_for_grading(timeout: float = 300):
    """Wait for all grading tasks to complete."""
    if _executor:
        _executor.shutdown(wait=True, cancel_futures=False)


def get_results() -> Dict[str, Dict]:
    """Get current grading results."""
    with _results_lock:
        return _grading_results.copy()


def save_results(results_file: Path):
    """Save results to file."""
    results = get_results()
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)


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

