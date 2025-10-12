from typing import List, Optional, Union
from src.main import load_correct_fallback_config_file
from src.generator import answer
from src.config import QueryPlanConfig
from src.retriever import load_artifacts
import json, pathlib

def test_with_manual_chunks(query: str, chunks: List[str]) -> str:
    cfg = load_correct_fallback_config_file()
    model_path = cfg.model_path
    
    result = answer(query, chunks, model_path, max_tokens=500)
    
    print(f"\nGenerated Answer:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
    return result


if __name__ == "__main__":
    script_dir = pathlib.Path(__file__).resolve().parent
    golden_chunks_path = script_dir / "golden_chunks.json"
    with open(golden_chunks_path, 'r') as f:
        test_cases = json.load(f)

    for test_case in test_cases:
        print(f"\nQuery: {test_case["query"]}")
        print("*" * 50 + "No chunks"+"*" * 50)
        test_with_manual_chunks(query=test_case["query"], chunks=[])
        print("*" * 50 + "With golden chunks"+"*" * 50)
        test_with_manual_chunks(query=test_case["query"], chunks=test_case["golden_chunks"])
        



