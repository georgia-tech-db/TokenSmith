import subprocess
import pytest
import json
import sys
from pathlib import Path
from utils.similarity import SimilarityScorer
from utils.answer_parser import extract_answer_from_output

scorer = SimilarityScorer()

def test_tokensmith_benchmark(benchmarks, test_config, results_dir):
    """Test TokenSmith with all benchmark questions."""
    
    if test_config["skip_slow"]:
        pytest.skip("Skipping slow end-to-end test")
    
    for benchmark in benchmarks:
        _run_single_benchmark(benchmark, test_config, results_dir)

def _run_single_benchmark(benchmark, test_config, results_dir):
    """Run a single benchmark test."""
    question = benchmark["question"]
    expected_answer = benchmark["expected_answer"]
    keywords = benchmark.get("keywords", [])
    threshold = benchmark.get("similarity_threshold", 0.6)
    
    # Construct command to run TokenSmith
    cmd = [
        sys.executable, "-m", "src.main", "chat",
        "--model_path", test_config["model_path"],
        "--index_prefix", test_config["index_prefix"]
    ]
    
    # Prepare input (question + exit command)
    input_text = f"{question}\nexit\n"
    
    try:
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=test_config["timeout"],
            cwd=Path(__file__).parent.parent
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Test timed out after {test_config['timeout']} seconds for: {question}")
    
    if proc.returncode != 0:
        pytest.fail(f"TokenSmith failed for '{question}' with exit code {proc.returncode}\n"
                   f"STDERR: {proc.stderr}\n"
                   f"STDOUT: {proc.stdout}")
    
    # Extract answer from output
    retrieved_answer = extract_answer_from_output(proc.stdout)
    
    # Calculate similarity scores
    scores = scorer.comprehensive_score(retrieved_answer, expected_answer, keywords)
    
    passed = scores["final_score"] >= threshold

    result_data = {
        "test_id": benchmark["id"],
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_answer": retrieved_answer,
        "keywords": keywords,
        "threshold": threshold,
        "scores": scores,
        "passed": passed,
        "stdout": proc.stdout,
        "stderr": proc.stderr
    }
    
    # Append to results file
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "a") as f:
        json.dump(result_data, f)
        f.write("\n")
    
    if not passed:
        fail_msg = (
            f"Benchmark failed for question: '{question}'\n"
            f"Expected: {expected_answer}\n"
            f"Retrieved: {retrieved_answer}\n"
            f"Final Score: {scores['final_score']:.3f} (threshold: {threshold})\n"
            f"Text Similarity: {scores['text_similarity']:.3f}\n"
            f"Semantic Similarity: {scores['semantic_similarity']:.3f}\n"
            f"Keywords Matched: {scores['keywords_matched']}/{len(keywords)}"
        )
        
        failed_log = results_dir / "failed_tests.log"
        with open(failed_log, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(fail_msg)
            f.write(f"\n{'='*50}\n")
        
        print(f"\n❌ Failed: {question}")
        print(f"Score: {scores['final_score']:.3f} (threshold: {threshold})")
    else:
        print(f"\n✅ Passed: {question}")
        print(f"Score: {scores['final_score']:.3f} (threshold: {threshold})")
