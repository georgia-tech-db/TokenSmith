import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def generate_summary_report(results_dir: Path):
    """Generate HTML summary report from JSON results."""
    results_file = results_dir / "benchmark_results.json"
    if not results_file.exists():
        return
    
    # Read all results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return
    
    # Load async LLM judge results if available
    async_llm_results = _load_async_llm_results()
    
    # Determine which metrics were used
    all_metrics = set()
    for result in results:
        all_metrics.update(result.get('active_metrics', []))
    
    # Generate adaptive HTML content
    html_content = _generate_html_template()
    html_content += _generate_summary_stats(results, all_metrics)
    html_content += _generate_detailed_results(results, all_metrics)
    
    # Add separate async LLM judge section if available
    if async_llm_results:
        html_content += _generate_async_llm_section(results, async_llm_results)
    
    html_content += "</body>\n</html>"
    
    # Write HTML report
    html_file = results_dir / "benchmark_summary.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"\nBenchmark results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  HTML: {html_file}")

def _generate_html_template() -> str:
    """Generate HTML template with styles."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>TokenSmith Benchmark Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .summary { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }
        .test-result { 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 15px; 
            border-radius: 5px; 
            min-width: 0;
        }
        .passed { border-left: 5px solid #4CAF50; }
        .failed { border-left: 5px solid #f44336; }
        .score { font-weight: bold; color: #2196F3; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 10px 0; }
        .metric-item { background: #f9f9f9; padding: 10px; border-radius: 3px; }
        pre { 
            background: #f9f9f9; 
            padding: 10px; 
            border-radius: 3px; 
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-width: 100%;
            width: 100%;
            box-sizing: border-box;
        }
        .answer-content {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 3px;
            line-height: 1.6;
        }
        .answer-content ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        .answer-content li {
            margin: 5px 0;
        }
        .answer-content p {
            margin: 10px 0;
        }
        details {
            margin: 10px 0;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 3px;
        }
        summary {
            cursor: pointer;
            font-weight: bold;
            user-select: none;
        }
        .chunk-item {
            background: #fff;
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid #2196F3;
        }
    </style>
</head>
<body>
    <h1>TokenSmith Benchmark Results</h1>
    """

def _generate_summary_stats(results: List[Dict[Any, Any]], active_metrics: set) -> str:
    """Generate summary statistics section."""
    scores = [r['scores']['final_score'] for r in results]
    avg_score = np.mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    passed = sum(1 for r in results if r['passed'])
    
    # Calculate per-metric averages
    metric_averages = {}
    for metric in active_metrics:
        metric_key = f"{metric}_similarity"
        metric_scores = [r['scores'].get(metric_key, 0) for r in results]
        metric_averages[metric] = np.mean(metric_scores) if metric_scores else 0
    
    html = f"""
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {len(results)}</p>
        <p><strong>Passed:</strong> {passed} ({passed/len(results)*100:.1f}%)</p>
        <p><strong>Failed:</strong> {len(results) - passed}</p>
        <p><strong>Average Score:</strong> {avg_score:.3f}</p>
        <p><strong>Score Range:</strong> {min_score:.3f} - {max_score:.3f}</p>
        <p><strong>Active Metrics:</strong> {', '.join(sorted(active_metrics))}</p>
        
        <h3>Per-Metric Averages</h3>
        <div class="metric-grid">
    """
    
    for metric, avg in sorted(metric_averages.items()):
        html += f'<div class="metric-item"><strong>{metric.title()}:</strong> {avg:.3f}</div>'
    
    html += "</div></div>"
    return html

def _convert_markdown_to_html(text: str) -> str:
    """Convert markdown to HTML."""
    try:
        import markdown
        return markdown.markdown(text)
    except ImportError:
        # Fallback to plain text if markdown not available
        return f'<pre style="white-space: pre-wrap;">{text}</pre>'


def _load_async_llm_results() -> Dict[str, Dict]:
    """Load async LLM judge results if available."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return {}
    
    subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return {}
    
    log_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
    results_file = log_dir / "async_llm_results.json"
    
    if not results_file.exists():
        return {}
    
    with open(results_file) as f:
        return json.load(f)


def _generate_detailed_results(results: List[Dict[Any, Any]], active_metrics: set) -> str:
    """Generate detailed results section."""
    html = "<h2>Detailed Results</h2>"
    
    for result in results:
        status_class = "passed" if result['passed'] else "failed"
        status_text = "PASSED" if result['passed'] else "FAILED"
        scores = result['scores']
        question = result['question']
        
        html += f"""
    <div class="test-result {status_class}">
        <h3>{question} - <span class="score">{status_text}</span></h3>
        <p><strong>Final Score:</strong> <span class="score">{scores['final_score']:.3f}</span></p>
        <p><strong>Threshold:</strong> {result['threshold']:.3f}</p>
        <p><strong>Active Metrics:</strong> {', '.join(result.get('active_metrics', []))}</p>
        
        <div class="metric-grid">
        """
        
        # Display scores for active metrics
        for metric in active_metrics:
            metric_key = f"{metric}_similarity"
            if metric_key in scores:
                html += f'<div class="metric-item"><strong>{metric.title()} Similarity:</strong> {scores[metric_key]:.3f}</div>'
        
        # Add keywords matched
        keywords_count = len(result.get('keywords', []))
        keywords_matched = scores.get('keywords_matched', 0)
        html += f'<div class="metric-item"><strong>Keywords Matched:</strong> {keywords_matched}/{keywords_count}</div>'
        
        html += "</div>"
        
        # Add chunks info if available
        chunks_info = result.get('chunks_info', [])
        if chunks_info:
            hyde_query = result.get('hyde_query')
            
            chunk_count = len(chunks_info)
            html += f"""
        <details>
            <summary>📦 Retrieved Chunks ({chunk_count} chunks)</summary>
            <div style="margin-top: 10px;">
            """
            
            # Show HyDE query if available
            if hyde_query:
                html += f"""
                <div style="background: #fff3cd; padding: 10px; margin-bottom: 10px; border-left: 3px solid #ffc107;">
                    <strong>🔍 HyDE Query (used for retrieval):</strong>
                    <pre style="margin-top: 5px; white-space: pre-wrap;">{hyde_query}</pre>
                </div>
                """
            
            for chunk in chunks_info:
                html += f"""
                <div class="chunk-item">
                    <strong>Rank {chunk['rank']}</strong> | Chunk ID: {chunk.get('chunk_id', '?')} | 
                    FAISS: rank #{chunk.get('faiss_rank', '?')} (score: {chunk['faiss_score']:.4f}) | 
                    BM25: rank #{chunk.get('bm25_rank', '?')} (score: {chunk['bm25_score']:.4f})
                    <pre style="margin-top: 5px;">{chunk['content']}</pre>
                </div>
                """
            html += """
            </div>
        </details>
        """
        
        html += f"""
        <h4>Expected Answer:</h4>
        <pre>{result['expected_answer']}</pre>
        
        <h4>Retrieved Answer:</h4>
        <div class="answer-content">{_convert_markdown_to_html(result['retrieved_answer'])}</div>
    </div>
        """
    
    return html


def _generate_async_llm_section(results: List[Dict[Any, Any]], async_llm_results: Dict[str, Dict]) -> str:
    """Generate separate async LLM judge results section."""
    html = """
    <h2>Async LLM Judge Results</h2>
    <div class="summary">
        <p><em>These evaluations are performed separately and do not affect pass/fail decisions.</em></p>
    """
    
    # Calculate average score
    scores = [r['score'] for r in async_llm_results.values() if 'error' not in r]
    if scores:
        avg_score = sum(scores) / len(scores)
        html += f"<p><strong>Average LLM Score:</strong> {avg_score:.2f}/5</p>"
    
    html += "</div>"
    
    for result in results:
        question = result['question']
        
        if question in async_llm_results:
            llm_result = async_llm_results[question]
            
            if "error" in llm_result:
                html += f"""
    <div class="test-result">
        <h3>{question}</h3>
        <p style="color: #f44336;"><strong>Error:</strong> {llm_result['error']}</p>
    </div>
                """
            else:
                html += f"""
    <div class="test-result">
        <h3>{question}</h3>
        <p><strong>LLM Score:</strong> <span class="score">{llm_result['score']}/5</span> ({llm_result['normalized_score']:.3f})</p>
        
        <div class="metric-grid">
            <div class="metric-item"><strong>Accuracy:</strong> {llm_result['accuracy']}</div>
            <div class="metric-item"><strong>Completeness:</strong> {llm_result['completeness']}</div>
            <div class="metric-item"><strong>Clarity:</strong> {llm_result['clarity']}</div>
        </div>
        
        <p><strong>Overall Reasoning:</strong> {llm_result['overall_reasoning']}</p>
    </div>
                """
    
    return html
