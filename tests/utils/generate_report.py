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
    
    # Determine which metrics were used
    all_metrics = set()
    for result in results:
        all_metrics.update(result.get('active_metrics', []))
    
    # Generate adaptive HTML content
    html_content = _generate_html_template()
    html_content += _generate_summary_stats(results, all_metrics)
    html_content += _generate_detailed_results(results, all_metrics)
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
            white-space: pre;
            overflow-x: auto;
            overflow-y: hidden;
            max-width: 100%;
            width: 100%;
            box-sizing: border-box;
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

def _generate_detailed_results(results: List[Dict[Any, Any]], active_metrics: set) -> str:
    """Generate detailed results section."""
    html = "<h2>Detailed Results</h2>"
    
    for result in results:
        status_class = "passed" if result['passed'] else "failed"
        status_text = "PASSED" if result['passed'] else "FAILED"
        scores = result['scores']
        
        html += f"""
    <div class="test-result {status_class}">
        <h3>{result['question']} - <span class="score">{status_text}</span></h3>
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
        
        html += """
        </div>
        
        <h4>Expected Answer:</h4>
        <pre>{}</pre>
        
        <h4>Retrieved Answer:</h4>
        <pre>{}</pre>
    </div>
        """.format(result['expected_answer'], result['retrieved_answer'])
    
    return html
