import json
import numpy as np

def generate_summary_report(results_dir):
    """Generate HTML summary report from JSON results."""
    results_file = results_dir / "benchmark_results.json"
    if not results_file.exists():
        return
	
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    if not results:
        return
    
    # Calculate summary statistics
    scores = [r['scores']['final_score'] for r in results]
    avg_score = np.mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    passed = sum(1 for r in results if r['passed'])
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TokenSmith Benchmark Results</title>
    <style>
		body {{ font-family: Arial, sans-serif; margin: 40px; }}
		.summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
		.test-result {{ 
			border: 1px solid #ddd; 
			margin: 10px 0; 
			padding: 15px; 
			border-radius: 5px; 
			min-width: 0;
		}}
		.passed {{ border-left: 5px solid #4CAF50; }}
		.failed {{ border-left: 5px solid #f44336; }}
		.score {{ font-weight: bold; color: #2196F3; }}
		pre {{ 
			background: #f9f9f9; 
			padding: 10px; 
			border-radius: 3px; 
			white-space: pre;
			overflow-x: auto;
			overflow-y: hidden;
			max-width: 100%;
			width: 100%;
			box-sizing: border-box;
		}}
	</style>
</head>
<body>
    <h1>TokenSmith Benchmark Results</h1>      
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Tests:</strong> {len(results)}</p>
        <p><strong>Passed:</strong> {passed} ({passed/len(results)*100:.1f}%)</p>
        <p><strong>Failed:</strong> {len(results) - passed}</p>
        <p><strong>Average Score:</strong> {avg_score:.3f}</p>
        <p><strong>Score Range:</strong> {min_score:.3f} - {max_score:.3f}</p>
    </div> 
<h2>Detailed Results</h2>
"""
    
    for result in results:
        status_class = "passed" if result['passed'] else "failed"
        status_text = "PASSED" if result['passed'] else "FAILED"
        
        html_content += f"""
    <div class="test-result {status_class}">
        <h3>{result['question']} - <span class="score">{status_text}</span></h3>
        <p><strong>Final Score:</strong> <span class="score">{result['scores']['final_score']:.3f}</span></p>
        <p><strong>Text Similarity:</strong> {result['scores']['text_similarity']:.3f}</p>
        <p><strong>Semantic Similarity:</strong> {result['scores']['semantic_similarity']:.3f}</p>
        <p><strong>Keywords Matched:</strong> {result['scores']['keywords_matched']}/{len(result.get('keywords', []))}</p>
        <h4>Expected Answer:</h4>
        <pre>{result['expected_answer']}</pre>
        <h4>Retrieved Answer:</h4>
        <pre>{result['retrieved_answer']}</pre>
    </div>
"""
    
    html_content += """
</body>
</html>
    """
    
    html_file = results_dir / "benchmark_summary.html"
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"\nBenchmark results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  HTML: {html_file}")
