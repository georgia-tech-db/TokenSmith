import os
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
        "--model-path",
        default=None,
        help="Path to generator model (overrides config)"
    )
    group.addoption(
        "--embed-model",
        default=None,
        help="Path to embedding model (overrides config, default: Qwen3)"
    )
    
    # === Retrieval Configuration ===
    # group.addoption(
    #     "--ensemble-method",
    #     choices=["linear", "weighted", "rrf"],
    #     default="rrf",
    #     help="Ensemble method to use (overrides config)"
    # )
    # group.addoption(
    #     "--rrf-k",
    #     type=int,
    #     default=60,
    #     help="RRF k value to use (overrides config)"
    # )
    # group.addoption(
    #     "--ranker-weights",
    #     type=dict,
    #     default=None,
    #     help="Ranker weights to use (overrides config)"
    # )
    # group.addoption(
    #     "--rerank-mode",
    #     choices=["none", "rerank"],
    #     default="none",
    #     help="Rerank mode to use (overrides config)"
    # )
    # group.addoption(
    #     "--seg-filter",
    #     default=None,
    #     help="Segment filter to use (overrides config)"
    # )
    
    # === Generator Configuration ===
    group.addoption(
        "--disable-chunks",
        action="store_true",
        default=None,
        help="Enable chunks in generator prompt"
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
        "--artifacts_dir",
        default=None,
        help="Artifacts folder for tests (overrides config)"
    )
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
        help="Metrics to use for evaluation (optionsRegistered metric:: text, semantic, keyword, bleu, all)"
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
    
    # Merge CLI arguments (higher priority)
    merged_config = {
        # Retrieval        
        "top_k": cfg.get("top_k", 5),
        "pool_size": cfg.get("pool_size", 60),
        "ensemble_method": cfg.get("ensemble_method", "rrf"),
        "rrf_k": cfg.get("rrf_k", 60),
        "ranker_weights": cfg.get("ranker_weights", {"faiss":0.6,"bm25":0.4}),
        "rerank_mode": cfg.get("rerank_mode", "none"),
        "seg_filter": cfg.get("seg_filter", None),
        "chunk_mode": cfg.get("chunk_mode", "sections"),
        "recursive_chunk_size": cfg.get("recursive_chunk_size", 1000),
        "recursive_overlap": cfg.get("recursive_overlap", 0),

        # Output
        "output_mode": pytestconfig.getoption("--output-mode") or cfg.get("output_mode", "terminal"),
        
        # Models
        "model_path": pytestconfig.getoption("--model-path") or cfg.get("model_path", "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"),
        "embed_model": pytestconfig.getoption("--embed-model") or cfg.get("embed_model", os.path.join(Path(__file__).parent.parent, "models", "Qwen3-Embedding-4B-Q8_0.gguf")),
        
        # Generator
        "system_prompt_mode": pytestconfig.getoption("--system-prompt") or cfg.get("system_prompt_mode", "baseline"),
        "max_gen_tokens": cfg.get("max_gen_tokens", 400),
        
        # Testing
        "artifacts_dir": pytestconfig.getoption("--artifacts_dir") or "index/tokens-200",
        "index_prefix": pytestconfig.getoption("--index-prefix") or cfg.get("index_prefix", "textbook_index"),
        "metrics": pytestconfig.getoption("--metrics") or cfg.get("metrics", ["all"]),
        "threshold_override": pytestconfig.getoption("--threshold") or cfg.get("threshold_override", None),
        
        # Query Enhancement (HyDE)
        "use_hyde": cfg.get("use_hyde", False),
        "hyde_max_tokens": cfg.get("hyde_max_tokens", 100),
    }

    # Handle enable/disable chunks
    disable_chunks_cli = pytestconfig.getoption("--disable-chunks")
    
    if disable_chunks_cli:
        merged_config["disable_chunks"] = True
    else:
        merged_config["disable_chunks"] = cfg.get("disable_chunks", False)
    
    # Handle golden chunks
    use_golden = pytestconfig.getoption("--use-golden-chunks")
    if use_golden is not None:
        merged_config["use_golden_chunks"] = use_golden
    else:
        merged_config["use_golden_chunks"] = cfg.get("use_golden_chunks", False)
    
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
        from tests.metrics import MetricRegistry
        registry = MetricRegistry()
        available = registry.list_metric_names()
        print(f"\n📊 Available metrics: {', '.join(available)}\n")
        pytest.exit("Metric listing complete", returncode=0)


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    config = session.config
    
    # Get output mode from config
    config_path = Path(config.getoption("--config"))
    output_mode = config.getoption("--output-mode")
    
    # If not specified via CLI, check config file
    if not output_mode and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        output_mode = cfg.get("testing", {}).get("output_mode", "html")
    
    # Wait for async LLM grading to complete
    _wait_for_async_grading()
    
    # Only generate HTML report if in html mode
    if output_mode == "html":
        from tests.utils import generate_summary_report
        results_dir = Path(__file__).parent / "results"
        generate_summary_report(results_dir)
    else:
        print("\n✅ Test session complete (terminal output mode)")


def _wait_for_async_grading():
    """Wait for async LLM grading threads to complete."""
    try:
        from tests.metrics.async_llm_judge import wait_for_grading, get_results, save_results
        
        print("\n" + "="*60)
        print("⏳ Waiting for async LLM grading to complete...")
        print("="*60)
        
        wait_for_grading(timeout=300)
        
        results = get_results()
        if results:
            # Save results
            logs_dir = Path(__file__).parent.parent / "logs"
            subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
            if subdirs:
                log_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
                results_file = log_dir / "async_llm_results.json"
                save_results(results_file)
                
                print(f"✅ Async LLM grading complete: {len(results)} answers graded")
                print(f"Results saved to: {results_file}\n")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  Async LLM grading failed: {e}")
