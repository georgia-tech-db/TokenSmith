import sys
import yaml
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def pytest_addoption(parser):
    """Add custom command line options."""
    group = parser.getgroup("tokensmith")
    
    # Existing options
    group.addoption("--model_path", default="models/qwen2.5-0.5b-instruct-q5_k_m.gguf",
                    help="Path to model file")
    group.addoption("--index_prefix", default="textbook_index", 
                    help="Index prefix for tests")
    group.addoption("--timeout", type=int, default=300,
                    help="Timeout for each test in seconds")
    group.addoption("--skip_slow", action="store_true",
                    help="Skip slow end-to-end tests")
    group.addoption("--benchmark_ids", default=None,
                    help="Comma-separated list of benchmark IDs to run")
    
    # New metric selection options
    group.addoption("--metric", action="append", dest="metrics",
                    help="Select specific metrics to evaluate. Options: text, semantic, keyword, bleu, nli, all")
    group.addoption("--threshold", type=float, default=None,
                    help="Override threshold for all tests")
    group.addoption("--list_metrics", action="store_true",
                    help="List available metrics and exit")

@pytest.fixture(scope="session")
def test_config(pytestconfig):
    """Load test configuration."""

    if pytestconfig.getoption("--list_metrics"):
        from .utils.metrics import MetricRegistry
        registry = MetricRegistry()
        available = registry.list_metric_names()
        print(f"\nAvailable metrics: {', '.join(available)}")
        pytest.exit("Metric listing complete")
    
    # Get selected metrics
    selected_metrics = pytestconfig.getoption("--metric") or ["all"]
    
    return {
        "model_path": pytestconfig.getoption("--model_path"),
        "index_prefix": pytestconfig.getoption("--index_prefix"),
        "timeout": pytestconfig.getoption("--timeout"),
        "skip_slow": pytestconfig.getoption("--skip_slow"),
        "metrics": selected_metrics,
        "threshold_override": pytestconfig.getoption("--threshold"),
    }

@pytest.fixture(scope="session")
def benchmarks(pytestconfig):
    """Load benchmark questions from YAML file."""
    benchmark_file = Path(__file__).parent / "benchmarks.yaml"
    with open(benchmark_file) as f:
        data = yaml.safe_load(f)
    
    all_benchmarks = data["benchmarks"]
    
    # Filter by selected IDs if provided
    selected_ids = pytestconfig.getoption("--benchmark_ids")
    if selected_ids:
        id_set = set(id.strip() for id in selected_ids.split(','))
        filtered_benchmarks = [b for b in all_benchmarks if b['id'] in id_set]
        print(f"Running {len(filtered_benchmarks)} selected benchmarks: {', '.join(id_set)}")
        return filtered_benchmarks
    
    return all_benchmarks

@pytest.fixture(scope="session")
def results_dir():
    """Create results directory."""
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    return results_path

@pytest.fixture(scope="session", autouse=True)
def setup_results_file(results_dir):
    """Initialize results file."""
    results_file = results_dir / "benchmark_results.json"
    if results_file.exists():
        results_file.unlink()
    return results_file

def pytest_sessionfinish(session, exitstatus):
    """Generate summary report after all tests complete."""
    from tests.utils import generate_summary_report
    results_dir = Path(__file__).parent / "results"
    generate_summary_report(results_dir)
