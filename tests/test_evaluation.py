#!/usr/bin/env python3
"""
Test script to verify the evaluation system works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from src.metrics import compute_metrics

def test_metrics():
    """Test the metrics computation."""
    print("Testing metrics computation...")
    
    gold_text = "Database normalization is the process of organizing data to reduce redundancy."
    retrieved_text = "Database normalization organizes data to minimize redundancy and improve integrity."
    
    metrics = compute_metrics(gold_text, retrieved_text)
    
    print(f"Gold text: {gold_text}")
    print(f"Retrieved text: {retrieved_text}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"IoU: {metrics.iou:.4f}")
    print(f"F1: {metrics.f1:.4f}")
    print(f"Gold tokens: {metrics.size_gold_tokens}")
    print(f"Retrieved tokens: {metrics.size_retrieved_tokens}")
    print(f"Intersection tokens: {metrics.size_intersection}")
    
    print("\nMetrics computation test passed!")

def test_dataset_loading():
    """Test loading the dataset."""
    print("\nTesting dataset loading...")
    
    dataset_path = Path("dataset.jsonl")
    if not dataset_path.exists():
        print("Dataset file not found!")
        return False
    
    import json
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} questions from dataset")
    for i, item in enumerate(data[:2]):  # Show first 2
        print(f"  {i+1}. {item['question'][:50]}...")
    
    print("Dataset loading test passed!")
    return True

def test_configs_loading():
    """Test loading the configs."""
    print("\nTesting configs loading...")
    
    configs_path = Path("configs.json")
    if not configs_path.exists():
        print("Configs file not found!")
        return False
    
    import json
    with open(configs_path, 'r') as f:
        configs = json.load(f)
    
    print(f"Loaded {len(configs)} configurations")
    for config in configs:
        print(f"  - {config['name']}: {config['chunking_strategy']} + {config['fusion']}")
    
    print("Configs loading test passed!")
    return True

def main():
    """Run all tests."""
    print("Running evaluation system tests...\n")
    
    test_metrics()
    
    if test_dataset_loading() and test_configs_loading():
        print("\nAll tests passed! The evaluation system is ready.")
        print("\nTo run the full evaluation:")
        print("  python src/chunking/evaluate_all.py")
    else:
        print("\nSome tests failed. Please check the setup.")

if __name__ == "__main__":
    main()
