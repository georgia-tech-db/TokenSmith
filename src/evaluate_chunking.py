#!/usr/bin/env python3
"""
Comprehensive evaluation script for all chunking and retrieval strategies.
Generates an HTML table similar to the provided image with consolidated results.
"""

import json
import sys
import statistics
import re
from pathlib import Path
from typing import List, Dict, Any, Iterable, Set
from dataclasses import dataclass

# Add src to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retriever import load_artifacts, retrieve


# =============================================================================
# Token-level evaluation metrics for chunking / retrieval.
# =============================================================================

_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")


def tokenize(text: str) -> Set[str]:
    """
    Tokenize text into a set of lowercase word tokens.

    - Uses a conservative regex to capture alphanumerics and simple apostrophes
    - Lowercases everything
    - Returns a SET to ensure duplicate tokens are counted once
    """
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    iou: float
    f1: float
    size_gold_tokens: int
    size_retrieved_tokens: int
    size_intersection: int


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return 0.0
    return float(numer) / float(denom)


def compute_metrics_from_token_sets(gold_tokens: Iterable[str], retrieved_tokens: Iterable[str]) -> Metrics:
    """
    Compute Precision, Recall, IoU, and F1 given token iterables.

    Duplicates are ignored by converting to sets. All tokens are assumed to be
    already normalized/lowercased if desired by the caller.
    """
    te: Set[str] = set(gold_tokens)
    tr: Set[str] = set(retrieved_tokens)

    inter_size = len(te & tr)
    te_size = len(te)
    tr_size = len(tr)

    precision = safe_div(inter_size, tr_size)
    recall = safe_div(inter_size, te_size)
    denom_iou = te_size + tr_size - inter_size
    iou = safe_div(inter_size, denom_iou)
    f1 = safe_div(2.0 * precision * recall, (precision + recall))

    return Metrics(
        precision=precision,
        recall=recall,
        iou=iou,
        f1=f1,
        size_gold_tokens=te_size,
        size_retrieved_tokens=tr_size,
        size_intersection=inter_size,
    )


def compute_metrics(gold_text: str, retrieved_text: str) -> Metrics:
    """
    Convenience wrapper to compute metrics from raw strings.
    """
    te = tokenize(gold_text)
    tr = tokenize(retrieved_text)
    return compute_metrics_from_token_sets(te, tr)


def format_metrics_as_dict(m: Metrics) -> Dict[str, float]:
    return {
        "precision": m.precision,
        "recall": m.recall,
        "iou": m.iou,
        "f1": m.f1,
        "gold_tokens": m.size_gold_tokens,
        "retrieved_tokens": m.size_retrieved_tokens,
        "intersection_tokens": m.size_intersection,
    }


# =============================================================================
# Evaluation script for chunking and retrieval strategies.
# =============================================================================


# Embedded configs for all chunking and retrieval strategies
CONFIGS = [
    {
        "name": "chars_weighted",
        "index_prefix": "textbook_index",
        "chunking_strategy": "chars",
        "overlap": 0,
        "fusion": "weighted",
        "bm25_weight": 0.3,
        "tag_weight": 0.2,
        "top_k": 5,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    {
        "name": "chars_rrf",
        "index_prefix": "textbook_index",
        "chunking_strategy": "chars",
        "overlap": 0,
        "fusion": "rrf",
        "bm25_weight": 0.3,
        "tag_weight": 0.2,
        "top_k": 5,
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    },
    # {
    #     "name": "tokens_weighted",
    #     "index_prefix": "textbook_index",
    #     "chunking_strategy": "tokens",
    #     "overlap": 0,
    #     "fusion": "weighted",
    #     "bm25_weight": 0.3,
    #     "tag_weight": 0.2,
    #     "top_k": 5,
    #     "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    # },
    # {
    #     "name": "tokens_rrf",
    #     "index_prefix": "textbook_index",
    #     "chunking_strategy": "tokens",
    #     "overlap": 0,
    #     "fusion": "rrf",
    #     "bm25_weight": 0.3,
    #     "tag_weight": 0.2,
    #     "top_k": 5,
    #     "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    # },
    # {
    #     "name": "sliding_weighted",
    #     "index_prefix": "textbook_index",
    #     "chunking_strategy": "sliding-tokens",
    #     "overlap": 80,
    #     "fusion": "weighted",
    #     "bm25_weight": 0.3,
    #     "tag_weight": 0.2,
    #     "top_k": 5,
    #     "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    # },
    # {
    #     "name": "sliding_rrf",
    #     "index_prefix": "textbook_index",
    #     "chunking_strategy": "sliding-tokens",
    #     "overlap": 80,
    #     "fusion": "rrf",
    #     "bm25_weight": 0.3,
    #     "tag_weight": 0.2,
    #     "top_k": 5,
    #     "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
    # }
]


def run_retrieval(question: str, config: Dict) -> str:
    """
    Retrieve context for a question per config and return concatenated text.
    """
    index_prefix: str = config.get("index_prefix", "textbook_index")
    fusion: str = config.get("fusion", "weighted")
    bm25_weight: float = float(config.get("bm25_weight", 0.3))
    tag_weight: float = float(config.get("tag_weight", 0.2))
    top_k: int = int(config.get("top_k", 5))
    embed_model: str = config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    index, chunks, sources, vectorizer, chunk_tags = load_artifacts(index_prefix)

    retrieved_chunks = retrieve(
        question,
        top_k,
        index,
        chunks,
        embed_model=embed_model,
        fusion=fusion,
        bm25_weight=bm25_weight,
        tag_weight=tag_weight,
        preview=False,
        sources=sources,
        vectorizer=vectorizer,
        chunk_tags=chunk_tags,
    )

    return "\n\n".join((c or "") for c in retrieved_chunks)


def load_jsonl(file_path: Path) -> List[Dict[str, str]]:
    """Loads a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_html_table(results: List[Dict[str, Any]]) -> str:
    """Generates an HTML table from the evaluation results with per-question and average views."""
    
    # Group results by config
    config_groups = {}
    for result in results:
        config_name = result['config_name']
        if config_name not in config_groups:
            config_groups[config_name] = {
                'config': result['config'],
                'questions': []
            }
        config_groups[config_name]['questions'].append(result)
    
    # Calculate averages for each config
    config_averages = []
    for config_name, group in config_groups.items():
        config = group['config']
        questions = group['questions']
        
        # Calculate averages
        precisions = [q['precision'] for q in questions]
        recalls = [q['recall'] for q in questions]
        ious = [q['iou'] for q in questions]
        f1s = [q['f1'] for q in questions]
        
        config_averages.append({
            'name': config_name,
            'chunking': config.get('chunking_strategy', 'Unknown'),
            'overlap': config.get('overlap', 0),
            'fusion': config.get('fusion', 'weighted'),
            'precision_avg': statistics.mean(precisions),
            'recall_avg': statistics.mean(recalls),
            'iou_avg': statistics.mean(ious),
            'f1_avg': statistics.mean(f1s),
            'precision_std': statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
            'recall_std': statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
            'iou_std': statistics.stdev(ious) if len(ious) > 1 else 0.0,
            'f1_std': statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        })
    
    # Sort by F1 average descending
    config_averages.sort(key=lambda x: x['f1_avg'], reverse=True)
    
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chunking and Retrieval Evaluation Results</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 20px; 
            background-color: #f4f7f6; 
            color: #333; 
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 30px; 
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        table { 
            width: 100%; 
            border-collapse: collapse; 
            margin-bottom: 20px; 
            background-color: #fff; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 12px 15px; 
            text-align: left; 
        }
        th { 
            background-color: #e0f2f1; 
            color: #263238; 
            font-weight: bold; 
            text-transform: uppercase;
            font-size: 14px;
        }
        tr:nth-child(even) { 
            background-color: #f9f9f9; 
        }
        tr:hover { 
            background-color: #f1f1f1; 
        }
        .metric-value { 
            font-weight: bold; 
        }
        .config-name { 
            font-weight: bold; 
            color: #007bff; 
        }
        .best-metric {
            background-color: #d4edda;
            font-weight: bold;
        }
        .chunking-strategy {
            font-weight: bold;
            color: #495057;
        }
        .fusion-mode {
            color: #6c757d;
            font-style: italic;
        }
        .question-text {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .config-section {
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .config-header {
            background-color: #3498db;
            color: white;
            padding: 15px 20px;
            font-weight: bold;
            font-size: 18px;
        }
        .config-content {
            padding: 20px;
        }
        .avg-row {
            background-color: #e8f4f8 !important;
            font-weight: bold;
            border-top: 2px solid #3498db;
        }
    </style>
</head>
<body>
    <h1>Chunking and Retrieval Evaluation Results</h1>
    <p style="text-align: center; color: #6c757d; margin-bottom: 30px;">
        Token-level Precision, Recall, IoU, and F1 metrics across different chunking and retrieval strategies
    </p>
"""
    
    # Generate per-config sections
    for avg_data in config_averages:
        config_name = avg_data['name']
        config_questions = config_groups[config_name]['questions']
        
        html += f"""
    <div class="config-section">
        <div class="config-header">
            {config_name} ({avg_data['chunking']}, Overlap: {avg_data['overlap']}, Fusion: {avg_data['fusion']})
        </div>
        <div class="config-content">
            <table>
                <thead>
                    <tr>
                        <th>Question</th>
                        <th>Recall</th>
                        <th>Precision</th>
                        <th>IoU</th>
                        <th>F1</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add individual question results
        for question in config_questions:
            html += f"""
                    <tr>
                        <td class="question-text">{question['question']}</td>
                        <td class="metric-value">{question['recall']:.4f}</td>
                        <td class="metric-value">{question['precision']:.4f}</td>
                        <td class="metric-value">{question['iou']:.4f}</td>
                        <td class="metric-value">{question['f1']:.4f}</td>
                    </tr>
            """
        
        # Add average row
        html += f"""
                    <tr class="avg-row">
                        <td><strong>AVERAGE</strong></td>
                        <td class="metric-value">{avg_data['recall_avg']:.4f} ± {avg_data['recall_std']:.4f}</td>
                        <td class="metric-value">{avg_data['precision_avg']:.4f} ± {avg_data['precision_std']:.4f}</td>
                        <td class="metric-value">{avg_data['iou_avg']:.4f} ± {avg_data['iou_std']:.4f}</td>
                        <td class="metric-value">{avg_data['f1_avg']:.4f} ± {avg_data['f1_std']:.4f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
        """
    
    # Summary table of averages only
    html += """
    <h2>Summary: Config Averages</h2>
    <table>
        <thead>
            <tr>
                <th>Config</th>
                <th>Chunking</th>
                <th>Overlap</th>
                <th>Fusion</th>
                <th>Recall</th>
                <th>Precision</th>
                <th>IoU</th>
                <th>F1</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Find best metrics for highlighting in summary
    best_recall = max(config_averages, key=lambda x: x['recall_avg'])
    best_precision = max(config_averages, key=lambda x: x['precision_avg'])
    best_iou = max(config_averages, key=lambda x: x['iou_avg'])
    best_f1 = max(config_averages, key=lambda x: x['f1_avg'])
    
    for avg_data in config_averages:
        # Determine which metrics are best
        is_best_recall = avg_data['name'] == best_recall['name']
        is_best_precision = avg_data['name'] == best_precision['name']
        is_best_iou = avg_data['name'] == best_iou['name']
        is_best_f1 = avg_data['name'] == best_f1['name']
        
        html += """
            <tr>
                <td class="config-name">{name}</td>
                <td class="chunking-strategy">{chunking}</td>
                <td>{overlap}</td>
                <td class="fusion-mode">{fusion}</td>
                <td class="metric-value {recall_class}">
                    {recall_avg:.4f} ± {recall_std:.4f}
                </td>
                <td class="metric-value {precision_class}">
                    {precision_avg:.4f} ± {precision_std:.4f}
                </td>
                <td class="metric-value {iou_class}">
                    {iou_avg:.4f} ± {iou_std:.4f}
                </td>
                <td class="metric-value {f1_class}">
                    {f1_avg:.4f} ± {f1_std:.4f}
                </td>
            </tr>
        """.format(
            name=avg_data['name'],
            chunking=avg_data['chunking'],
            overlap=avg_data['overlap'],
            fusion=avg_data['fusion'],
            recall_class='best-metric' if is_best_recall else '',
            recall_avg=avg_data['recall_avg'],
            recall_std=avg_data['recall_std'],
            precision_class='best-metric' if is_best_precision else '',
            precision_avg=avg_data['precision_avg'],
            precision_std=avg_data['precision_std'],
            iou_class='best-metric' if is_best_iou else '',
            iou_avg=avg_data['iou_avg'],
            iou_std=avg_data['iou_std'],
            f1_class='best-metric' if is_best_f1 else '',
            f1_avg=avg_data['f1_avg'],
            f1_std=avg_data['f1_std']
        )
    
    html += """
        </tbody>
    </table>
    
    <div style="margin-top: 30px; padding: 20px; background-color: #e8f4f8; border-radius: 8px;">
        <h3>Key Insights:</h3>
        <ul>
            <li><strong>Best Recall:</strong> {best_recall_name} ({best_recall_mean:.4f} ± {best_recall_std:.4f})</li>
            <li><strong>Best Precision:</strong> {best_precision_name} ({best_precision_mean:.4f} ± {best_precision_std:.4f})</li>
            <li><strong>Best IoU:</strong> {best_iou_name} ({best_iou_mean:.4f} ± {best_iou_std:.4f})</li>
            <li><strong>Best F1:</strong> {best_f1_name} ({best_f1_mean:.4f} ± {best_f1_std:.4f})</li>
        </ul>
    </div>
    
    <div style="margin-top: 20px; font-size: 12px; color: #6c757d; text-align: center;">
        Generated on {timestamp}
    </div>
</body>
</html>
    """.format(
        best_recall_name=best_recall['name'],
        best_recall_mean=best_recall['recall_avg'],
        best_recall_std=best_recall['recall_std'],
        best_precision_name=best_precision['name'],
        best_precision_mean=best_precision['precision_avg'],
        best_precision_std=best_precision['precision_std'],
        best_iou_name=best_iou['name'],
        best_iou_mean=best_iou['iou_avg'],
        best_iou_std=best_iou['iou_std'],
        best_f1_name=best_f1['name'],
        best_f1_mean=best_f1['f1_avg'],
        best_f1_std=best_f1['f1_std'],
        timestamp=__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )
    
    return html


def main():
    dataset_path = Path("dataset.jsonl")
    output_html_path = Path("evaluation_results.html")

    if not dataset_path.exists():
        print(f"Error: Dataset file not found at {dataset_path}")
        print("Please create a dataset.jsonl with questions and gold answers.")
        sys.exit(1)

    dataset = load_jsonl(dataset_path)
    configs = CONFIGS

    all_results = []

    print(f"Starting evaluation for {len(dataset)} questions across {len(configs)} configurations...")

    for config in configs:
        config_name = config.get("name", "Unnamed Config")
        print(f"\n--- Evaluating Configuration: {config_name} ---")
        
        for i, item in enumerate(dataset):
            question = item["question"]
            gold_text = item["gold_text"]

            print(f"  Question {i+1}: {question[:70]}...")

            try:
                retrieved_text = run_retrieval(question, config)
                print(f"    Retrieved text: {retrieved_text}")
                metrics = compute_metrics(gold_text, retrieved_text)
                
                result = {
                    "config_name": config_name,
                    "config": config,
                    "question": question,
                    "gold_text": gold_text,
                    "retrieved_text": retrieved_text,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "iou": metrics.iou,
                    "f1": metrics.f1,
                    "gold_tokens_count": metrics.size_gold_tokens,
                    "retrieved_tokens_count": metrics.size_retrieved_tokens,
                    "intersection_tokens": metrics.size_intersection,
                }
                all_results.append(result)
                print(f"    Metrics: P={metrics.precision:.4f}, R={metrics.recall:.4f}, IoU={metrics.iou:.4f}, F1={metrics.f1:.4f}")
            except Exception as e:
                print(f"    Error during retrieval or metric computation for config {config_name}, question '{question[:50]}...': {e}")
                # Append a placeholder result for failed evaluations
                all_results.append({
                    "config_name": config_name,
                    "config": config,
                    "question": question,
                    "gold_text": gold_text,
                    "retrieved_text": "ERROR",
                    "precision": 0.0, "recall": 0.0, "iou": 0.0, "f1": 0.0,
                    "gold_tokens_count": 0, "retrieved_tokens_count": 0, "intersection_tokens": 0,
                    "error": str(e)
                })

    print("\nGenerating HTML report...")
    html_report = generate_html_table(all_results)
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    print(f"Evaluation complete. Results saved to {output_html_path.resolve()}")
    print("You can open this file in your browser to view the results.")

    # Also save raw results as JSON
    json_output_path = Path("evaluation_results.json")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Raw results also saved to {json_output_path.resolve()}")


if __name__ == "__main__":
    main()
