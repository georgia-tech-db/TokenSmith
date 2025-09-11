import os
import json
import difflib
import warnings
import numpy as np
from sentence_transformers import SentenceTransformer, util

class SimilarityScorer:
    def __init__(self):
        # Set env variable to force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        try:
            # Force CPU device because GPU doesn't support CUDA
            self.device = 'cpu'
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")
            warnings.filterwarnings("ignore", message=".*cuda.*", category=UserWarning)
            
            # Load model with explicit CPU device
            self.model = SentenceTransformer('all-MiniLM-L12-v2', device=self.device)
            self.util = util
            self.use_embeddings = True
            
            print(f"SimilarityScorer initialized with device: {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load sentence transformer, using text similarity only. Error: {e}")
            self.use_embeddings = False
    
    def text_similarity(self, text1, text2):
        """Calculate text similarity using SequenceMatcher."""
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using embeddings on CPU."""
        if not self.use_embeddings:
            return self.text_similarity(text1, text2)
        
        try:
            embeddings = self.model.encode(
                [text1, text2], 
                convert_to_tensor=True, 
                device=self.device,
                show_progress_bar=True
            )
            
            if hasattr(embeddings, 'cpu'):
                embeddings = embeddings.cpu()
            
            similarity = self.util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            print(f"Warning: Semantic similarity failed, falling back to text similarity. Error: {e}")
            return self.text_similarity(text1, text2)
    
    def keyword_match_score(self, text, keywords):
        """Calculate keyword matching score."""
        if not keywords:
            return 0
        
        text_lower = text.lower()
        matched = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matched / len(keywords)
    
    def comprehensive_score(self, answer, expected, keywords):
        """Calculate comprehensive similarity score using CPU-only operations."""
        text_sim = self.text_similarity(answer, expected)
        semantic_sim = self.semantic_similarity(answer, expected) if self.use_embeddings else 0
        keyword_score = self.keyword_match_score(answer, keywords)
        
        # Weighted combination - arbitary for now
        if self.use_embeddings:
            final_score = 0.3 * text_sim + 0.5 * semantic_sim + 0.2 * keyword_score
        else:
            final_score = 0.7 * text_sim + 0.3 * keyword_score
        
        return {
            "text_similarity": text_sim,
            "semantic_similarity": semantic_sim,
            "keyword_score": keyword_score,
            "final_score": final_score,
            "keywords_matched": sum(1 for kw in keywords if kw.lower() in answer.lower())
        }


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
        .test-result {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .passed {{ border-left: 5px solid #4CAF50; }}
        .failed {{ border-left: 5px solid #f44336; }}
        .score {{ font-weight: bold; color: #2196F3; }}
        pre {{ background: #f9f9f9; padding: 10px; border-radius: 3px; overflow-x: auto; }}
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
