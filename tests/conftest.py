import sys
import yaml
import pytest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pytest_addoption(parser):
    """Add custom command-line options for testing."""
    group = parser.getgroup("tokensmith", "TokenSmith Testing Options")
    
    # === Core Configuration ===
    group.addoption(
        "--config",
        default="config/config.yaml",
        help="Path to configuration YAML file (default: config/config.yaml)"
    )
    
    # === Output Control ===
    group.addoption(
        "--output-mode",
        choices=["terminal", "html"],
        default=None,
        help="Output mode: 'terminal' for console output, 'html' for HTML report (overrides config)"
    )
    
    # === Model Selection ===
    group.addoption(
        "--generator-model",
        default=None,
        help="Path to generator model (overrides config)"
    )
    group.addoption(
        "--embed-model",
        default=None,
        help="Path to embedding model (overrides config, default: Qwen3)"
    )
    
    # === Retrieval Configuration ===
    group.addoption(
        "--retrieval-method",
        choices=["hybrid", "faiss", "bm25", "tag"],
        default=None,
        help="Retrieval method to use (overrides config)"
    )
    group.addoption(
        "--faiss-weight",
        type=float,
        default=None,
        help="Weight for FAISS retrieval in hybrid mode (0.0-1.0)"
    )
    group.addoption(
        "--bm25-weight",
        type=float,
        default=None,
        help="Weight for BM25 retrieval in hybrid mode (0.0-1.0)"
    )
    group.addoption(
        "--tag-weight",
        type=float,
        default=None,
        help="Weight for tag-based retrieval in hybrid mode (0.0-1.0)"
    )
    
    # === Generator Configuration ===
    group.addoption(
        "--enable-chunks",
        action="store_true",
        default=None,
        help="Enable chunks in generator prompt"
    )
    group.addoption(
        "--disable-chunks",
        action="store_true",
        default=None,
        help="Disable chunks in generator prompt"
    )
    group.addoption(
        "--use-golden-chunks",
        action="store_true",
        default=None,
        help="Use golden chunks from benchmarks (overrides retrieval)"
    )
    group.addoption(
        "--system-prompt",
        choices=["baseline", "tutor", "concise", "detailed"],
        default=None,
        help="System prompt mode (overrides config)"
    )
    
    # === Testing Options ===
    group.addoption(
        "--index-prefix",
        default=None,
        help="Index prefix for tests (overrides config)"
    )
    group.addoption(
        "--benchmark-ids",
        default=None,
        help="Comma-separated list of benchmark IDs to run (e.g., 'transactions,er_modeling')"
    )
    group.addoption(
        "--metrics",
        action="append",
        dest="metrics_list",
        help="Metrics to use for evaluation (options: text, semantic, keyword, bleu, all)"
    )
    group.addoption(
        "--threshold",
        type=float,
        default=None,
        help="Override similarity threshold for all tests"
    )
    
    # === Utility Options ===
    group.addoption(
        "--list-metrics",
        action="store_true",
        help="List available metrics and exit"
    )


@pytest.fixture(scope="session")
def config(pytestconfig):
    """
    Load and merge configuration from YAML file and CLI arguments.
    
    Priority: CLI args > config.yaml
    """
    # Load config file
    config_path = Path(pytestconfig.getoption("--config"))
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    
    # Get testing section or create empty dict
    testing_cfg = cfg.get("testing", {})
    
    # Merge CLI arguments (higher priority)
    merged_config = {
        # Output
        "output_mode": pytestconfig.getoption("--output-mode") or testing_cfg.get("output_mode", "html"),
        
        # Models
        "generator_model": pytestconfig.getoption("--generator-model") or cfg.get("generator_model", "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"),
        "embed_model": pytestconfig.getoption("--embed-model") or cfg.get("embed_model", "/nethome/sbansal309/tokensmith/models/Qwen3-Embedding-4B-Q8_0.gguf"),
        
        # Retrieval
        "retrieval_method": pytestconfig.getoption("--retrieval-method") or cfg.get("retrieval_method", "hybrid"),
        "faiss_weight": pytestconfig.getoption("--faiss-weight") or cfg.get("faiss_weight", 0.5),
        "bm25_weight": pytestconfig.getoption("--bm25-weight") or cfg.get("bm25_weight", 0.3),
        "tag_weight": pytestconfig.getoption("--tag-weight") or cfg.get("tag_weight", 0.2),
        "top_k": cfg.get("top_k", 5),
        "halo_mode": cfg.get("halo_mode", "halo"),
        
        # Generator
        "system_prompt_mode": pytestconfig.getoption("--system-prompt") or cfg.get("system_prompt_mode", "tutor"),
        "max_gen_tokens": cfg.get("max_gen_tokens", 400),
        
        # Testing
        "index_prefix": pytestconfig.getoption("--index-prefix") or testing_cfg.get("index_prefix", "textbook_index"),
        "metrics": pytestconfig.getoption("--list-metrics") or testing_cfg.get("metrics", ["all"]),
        "threshold_override": pytestconfig.getoption("--threshold") or testing_cfg.get("threshold_override"),
    }
    
    # Handle enable/disable chunks
    enable_chunks_cli = pytestconfig.getoption("--enable-chunks")
    disable_chunks_cli = pytestconfig.getoption("--disable-chunks")
    
    if enable_chunks_cli:
        merged_config["enable_chunks"] = True
    elif disable_chunks_cli:
        merged_config["enable_chunks"] = False
    else:
        merged_config["enable_chunks"] = cfg.get("enable_chunks", True)
    
    # Handle golden chunks
    use_golden = pytestconfig.getoption("--use-golden-chunks")
    if use_golden is not None:
        merged_config["use_golden_chunks"] = use_golden
    else:
        merged_config["use_golden_chunks"] = testing_cfg.get("use_golden_chunks", False)
    
    return merged_config


@pytest.fixture(scope="session")
def benchmarks(pytestconfig, config):
    """
    Load benchmark questions from YAML file.
    
    Optionally filters by benchmark IDs if specified.
    """
    benchmark_file = Path(__file__).parent / "benchmarks.yaml"
    with open(benchmark_file) as f:
        data = yaml.safe_load(f)
    
    all_benchmarks = data["benchmarks"]
    
    # Filter by selected IDs if provided
    selected_ids = pytestconfig.getoption("--benchmark-ids")
    if selected_ids:
        id_set = set(id.strip() for id in selected_ids.split(','))
        filtered = [b for b in all_benchmarks if b['id'] in id_set]
        print(f"\n📋 Running {len(filtered)} selected benchmarks: {', '.join(id_set)}")
        return filtered
    
    print(f"\n📋 Running all {len(all_benchmarks)} benchmarks")
    return all_benchmarks


@pytest.fixture(scope="session")
def results_dir():
    """Create and return the results directory."""
    results_path = Path(__file__).parent / "results"
    results_path.mkdir(exist_ok=True)
    return results_path


@pytest.fixture(scope="session", autouse=True)
def setup_results_file(results_dir):
    """Initialize results file (clean previous results)."""
    results_file = results_dir / "benchmark_results.json"
    if results_file.exists():
        results_file.unlink()
    return results_file


def pytest_sessionstart(session):
    """Handle session start - check for list-metrics flag."""
    if session.config.getoption("--list-metrics"):
        from tests.utils.metrics import MetricRegistry
        registry = MetricRegistry()
        available = registry.list_metric_names()
        print(f"\n📊 Available metrics: {', '.join(available)}\n")
        pytest.exit("Metric listing complete", returncode=0)


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete (only if HTML mode)."""
    config = session.config
    
    # Get output mode from config
    config_path = Path(config.getoption("--config"))
    output_mode = config.getoption("--output-mode")
    
    # If not specified via CLI, check config file
    if not output_mode and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        output_mode = cfg.get("testing", {}).get("output_mode", "html")
    
    # Only generate HTML report if in html mode
    if output_mode == "html":
        from tests.utils import generate_summary_report
        results_dir = Path(__file__).parent / "results"
        generate_summary_report(results_dir)
    else:
        print("\n✅ Test session complete (terminal output mode)")
