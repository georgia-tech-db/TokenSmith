# import subprocess
# import pytest
# import json
# import sys
# from pathlib import Path
# from .utils.metrics import SimilarityScorer
# from .utils.answer_parser import extract_answer_from_output

# def test_tokensmith_benchmark(benchmarks, test_config, results_dir):
#     """Test TokenSmith with all benchmark questions using selected metrics."""
    
#     if test_config["skip_slow"]:
#         pytest.skip("Skipping slow end-to-end test")
    
#     # Initialize scorer with selected metrics
#     scorer = SimilarityScorer(enabled_metrics=test_config["metrics"])
    
#     print(f"\nUsing metrics: {test_config['metrics']}")
#     print(f"Available metrics: {scorer.registry.list_metric_names()}")
    
#     for benchmark in benchmarks:
#         _run_single_benchmark(benchmark, test_config, results_dir, scorer)

# def _run_single_benchmark(benchmark, test_config, results_dir, scorer):
#     """Run a single benchmark test with selected metrics."""
#     question = benchmark["question"]
#     expected_answer = benchmark["expected_answer"]
#     keywords = benchmark.get("keywords", [])
    
#     # Use threshold override if provided
#     threshold = test_config["threshold_override"] or benchmark.get("similarity_threshold", 0.6)
    
#     # Run TokenSmith subprocess
#     cmd = [
#         sys.executable, "-m", "src.main", "chat",
#         "--index_prefix", test_config["index_prefix"],
#         "--model_path", test_config["model_path"]
#     ]
    
#     input_text = f"{question}\nexit\n"
    
#     try:
#         proc = subprocess.run(
#             cmd,
#             input=input_text,
#             text=True,
#             capture_output=True,
#             timeout=test_config["timeout"],
#             cwd=Path(__file__).parent.parent
#         )
#     except subprocess.TimeoutExpired:
#         pytest.fail(f"Test timed out after {test_config['timeout']} seconds for: {question}")
    
#     if proc.returncode != 0:
#         pytest.fail(f"TokenSmith failed for '{question}' with exit code {proc.returncode}\n"
#                    f"STDERR: {proc.stderr}\n"
#                    f"STDOUT: {proc.stdout}")
    
#     # Extract answer
#     retrieved_answer = extract_answer_from_output(proc.stdout)
    
#     # Calculate scores using selected metrics
#     scores = scorer.calculate_scores(retrieved_answer, expected_answer, keywords)
    
#     # Determine if test passed
#     passed = scores.get("final_score", 0) >= threshold
    
#     # Save detailed results
#     result_data = {
#         "test_id": benchmark["id"],
#         "question": question,
#         "expected_answer": expected_answer,
#         "retrieved_answer": retrieved_answer,
#         "keywords": keywords,
#         "threshold": threshold,
#         "scores": scores,
#         "passed": passed,
#         "active_metrics": scores.get("active_metrics", []),
#         "stdout": proc.stdout,
#         "stderr": proc.stderr
#     }
    
#     # Append to results file
#     results_file = results_dir / "benchmark_results.json"
#     with open(results_file, "a") as f:
#         json.dump(result_data, f)
#         f.write("\n")
    
#     # Assert based on results
#     if not passed:
#         fail_msg = (
#             f"Benchmark failed for question: '{question}'\n"
#             f"Expected: {expected_answer}\n"
#             f"Retrieved: {retrieved_answer}\n"
#             f"Final Score: {scores.get('final_score', 0):.3f} (threshold: {threshold})\n"
#             f"Active Metrics: {', '.join(scores.get('active_metrics', []))}"
#         )
        
#         # Log failed test
#         failed_log = results_dir / "failed_tests.log"
#         with open(failed_log, "a") as f:
#             f.write(f"\n{'='*50}\n{fail_msg}\n{'='*50}\n")
        
#         print(f"\n❌ Failed: {question}")
#         print(f"Score: {scores.get('final_score', 0):.3f} (threshold: {threshold})")
#     else:
#         print(f"\n✅ Passed: {question}")
#         print(f"Score: {scores.get('final_score', 0):.3f} (threshold: {threshold})")



import subprocess
import pytest
import json
import sys
from pathlib import Path
from .utils.metrics import SimilarityScorer
from .utils.answer_parser import extract_answer_from_output


def test_tokensmith_benchmark(benchmarks, test_config, results_dir):
    """Test TokenSmith with all benchmark questions using selected modular metrics."""
    
    if test_config["skip_slow"]:
        pytest.skip("Skipping slow end-to-end test")
    
    try:
        scorer = SimilarityScorer(enabled_metrics=test_config["metrics"])
    except Exception as e:
        pytest.fail(f"Failed to initialize similarity scorer: {e}")
    
    available_metrics = scorer.registry.list_metric_names()
    active_metrics = list(scorer._get_active_metrics().keys())
    
    print(f"\nBenchmark Configuration:")
    print(f"  Requested metrics: {test_config['metrics']}")
    print(f"  Available metrics: {', '.join(available_metrics)}")
    print(f"  Active metrics: {', '.join(active_metrics)}")
    print(f"  Total benchmarks: {len(benchmarks)}")
    
    if not active_metrics:
        pytest.fail("No metrics are available for testing")
    
    passed_count = 0
    failed_count = 0
    
    for benchmark in benchmarks:
        result = _run_single_benchmark(benchmark, test_config, results_dir, scorer)
        if result:
            passed_count += 1
        else:
            failed_count += 1
    
    print(f"\nOverall Results: {passed_count} passed, {failed_count} failed")


def _run_single_benchmark(benchmark, test_config, results_dir, scorer):
    """Run a single benchmark test with selected metrics."""
    question = benchmark["question"]
    expected_answer = benchmark["expected_answer"]
    keywords = benchmark.get("keywords", [])
    benchmark_id = benchmark.get("id", "unknown")
    
    # Use threshold override if provided
    threshold = test_config["threshold_override"] or benchmark.get("similarity_threshold", 0.6)
    
    print(f"\n📋 Running benchmark: {benchmark_id}")
    print(f"   Question: {question[:60]}{'...' if len(question) > 60 else ''}")
    
    cmd = [
        sys.executable, "-m", "src.main", "chat",
        "--index_prefix", test_config["index_prefix"],
        "--model_path", test_config["model_path"]
    ]
    
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
        error_msg = f"Test timed out after {test_config['timeout']} seconds for: {question}"
        _log_benchmark_failure(results_dir, benchmark_id, error_msg)
        pytest.fail(error_msg)
    
    if proc.returncode != 0:
        error_msg = (f"TokenSmith failed for '{question}' with exit code {proc.returncode}\n"
                    f"STDERR: {proc.stderr}\n"
                    f"STDOUT: {proc.stdout}")
        _log_benchmark_failure(results_dir, benchmark_id, error_msg)
        pytest.fail(error_msg)
    
    # Extract answer
    retrieved_answer = extract_answer_from_output(proc.stdout)
    
    if not retrieved_answer or retrieved_answer.strip() == "":
        error_msg = f"No answer extracted from TokenSmith output for question: {question}"
        _log_benchmark_failure(results_dir, benchmark_id, error_msg)
        pytest.fail(error_msg)
    
    # Calculate scores using selected metrics
    try:
        scores = scorer.calculate_scores(retrieved_answer, expected_answer, keywords)
    except Exception as e:
        error_msg = f"Score calculation failed for question '{question}': {e}"
        _log_benchmark_failure(results_dir, benchmark_id, error_msg)
        pytest.fail(error_msg)
    
    if "error" in scores:
        error_msg = f"Scoring error for question '{question}': {scores['error']}"
        _log_benchmark_failure(results_dir, benchmark_id, error_msg)
        pytest.fail(error_msg)
    
    final_score = scores.get("final_score", 0)
    passed = final_score >= threshold
    
    result_data = {
        "test_id": benchmark_id,
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_answer": retrieved_answer,
        "keywords": keywords,
        "threshold": threshold,
        "scores": scores,
        "passed": passed,
        "active_metrics": scores.get("active_metrics", []),
        "metric_weights": _get_metric_weights(scorer, scores.get("active_metrics", [])),
        "timestamp": _get_timestamp(),
        "config": {
            "model_path": test_config["model_path"],
            "index_prefix": test_config["index_prefix"],
            "timeout": test_config["timeout"]
        },
        "stdout": proc.stdout,
        "stderr": proc.stderr
    }
    
    _save_benchmark_result(results_dir, result_data)
    _display_benchmark_result(benchmark_id, question, passed, final_score, threshold, scores)
    
    if not passed:
        fail_msg = _create_failure_message(question, expected_answer, retrieved_answer, 
                                         final_score, threshold, scores)
        _log_benchmark_failure(results_dir, benchmark_id, fail_msg)
    
    return passed


def _get_metric_weights(scorer, active_metric_names):
    """Get weights for active metrics."""
    weights = {}
    for name in active_metric_names:
        metric = scorer.registry.get_metric(name)
        if metric:
            weights[name] = metric.weight
    return weights


def _get_timestamp():
    """Get current timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()


def _save_benchmark_result(results_dir, result_data):
    """Save benchmark result to JSON file."""
    results_file = results_dir / "benchmark_results.json"
    with open(results_file, "a") as f:
        json.dump(result_data, f, indent=None)
        f.write("\n")


def _display_benchmark_result(benchmark_id, question, passed, final_score, threshold, scores):
    """Display formatted benchmark result."""
    status_icon = "✅" if passed else "❌"
    status_text = "PASSED" if passed else "FAILED"
    
    print(f"   {status_icon} {status_text}: {final_score:.3f} (threshold: {threshold:.3f})")
    
    # Show individual metric scores
    active_metrics = scores.get("active_metrics", [])
    if len(active_metrics) > 1:
        print(f"   📊 Metric breakdown:")
        for metric in active_metrics:
            metric_score = scores.get(f"{metric}_similarity", 0)
            print(f"      {metric}: {metric_score:.3f}")
        
        keywords_matched = scores.get("keywords_matched", 0)
        total_keywords = len(scores.get("keywords", []))
        if total_keywords > 0:
            print(f"      keywords: {keywords_matched}/{total_keywords}")


def _create_failure_message(question, expected_answer, retrieved_answer, 
                          final_score, threshold, scores):
    """Create detailed failure message."""
    msg_parts = [
        f"Benchmark failed for question: '{question}'",
        f"Expected: {expected_answer}",
        f"Retrieved: {retrieved_answer}",
        f"Final Score: {final_score:.3f} (threshold: {threshold:.3f})",
        f"Active Metrics: {', '.join(scores.get('active_metrics', []))}",
        ""
    ]
    
    # Add individual metric scores
    active_metrics = scores.get("active_metrics", [])
    if active_metrics:
        msg_parts.append("Individual Metric Scores:")
        for metric in active_metrics:
            metric_score = scores.get(f"{metric}_similarity", 0)
            msg_parts.append(f"  {metric}: {metric_score:.3f}")
        
        keywords_matched = scores.get("keywords_matched", 0)
        total_keywords = len(scores.get("keywords", []))
        if total_keywords > 0:
            msg_parts.append(f"  keywords: {keywords_matched}/{total_keywords}")
    
    return "\n".join(msg_parts)


def _log_benchmark_failure(results_dir, benchmark_id, message):
    """Log benchmark failure to dedicated log file."""
    failed_log = results_dir / "failed_tests.log"
    with open(failed_log, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"BENCHMARK FAILURE: {benchmark_id}\n")
        f.write(f"{'='*60}\n")
        f.write(f"{message}\n")
        f.write(f"{'='*60}\n")
