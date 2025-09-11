import sys
import yaml
import pytest
from pathlib import Path
from utils import generate_summary_report

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def pytest_addoption(parser):
    """Add custom command line options."""
    group = parser.getgroup("tokensmith")
    group.addoption("--index_prefix", default="textbook_index", 
                    help="Index prefix for tests")
    group.addoption("--model_path", default="models/Qwen3-4B-UD-Q8_K_XL.gguf",
                    help="Path to model file")
    group.addoption("--timeout", type=int, default=300,
                    help="Timeout for each test in seconds")
    group.addoption("--skip_slow", action="store_true",
                    help="Skip slow end-to-end tests")

@pytest.fixture(scope="session")
def test_config(pytestconfig):
    """Load test configuration."""
    return {
        "index_prefix": pytestconfig.getoption("--index_prefix"),
        "model_path": pytestconfig.getoption("--model_path"),
        "timeout": pytestconfig.getoption("--timeout"),
        "skip_slow": pytestconfig.getoption("--skip_slow"),
    }

@pytest.fixture(scope="session")
def benchmarks():
    """Load benchmark questions from YAML file."""
    benchmark_file = Path(__file__).parent / "benchmarks.yaml"
    with open(benchmark_file) as f:
        data = yaml.safe_load(f)
    return data["benchmarks"]

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
    # Clear previous results
    if results_file.exists():
        results_file.unlink()
    return results_file

def pytest_sessionfinish(session, exitstatus):
    """Generate summary report after all tests complete."""
    results_dir = Path(__file__).parent / "results"
    generate_summary_report(results_dir)
