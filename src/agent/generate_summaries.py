"""
generate_summaries.py

Offline summarization pipeline using a large "Thinking" model (Qwen2.5-72B or QwQ-32B).
Uses a recursive rolling window approach to maintain context across long sections.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from llama_cpp import Llama

# --- Configuration ---
# Resolve model path relative to project root (parent of src/)
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
MODEL_PATH = str(_PROJECT_ROOT / "models" / "Qwen3-30B-A3B-Q6_K.gguf")

# Context settings
CTX_SIZE = 16384  # Qwen handles 32k+, P6000s have room
MAX_PARA_TOKENS = 1000  # Budget for input paragraph chunks
SUMMARY_BUDGET = 500    # Budget for the output summary
RECURSION_DEPTH = 0     # Track depth to prevent infinite loops

class ThinkingSummarizer:
    def __init__(self, model_path: str, n_ctx: int = 8192):
        print(f"Loading model: {model_path}...")
        # split_mode=1 (layer split) is usually best for llama.cpp on dual GPUs
        # tensor_split=[24, 24] assumes equal VRAM on both P6000s
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,      # Offload all layers to GPU
            tensor_split=[24, 24], # Split evenly across 2 cards
            verbose=False,
            n_batch=512
        )

    def clean_thinking_tokens(self, text: str) -> str:
        """Remove <think>...</think> blocks if using a reasoning model."""
        # Remove thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        # Remove generic clutter
        return text.strip()

    def generate_update(self, current_summary: str, new_text: str) -> str:
        """
        Updates the running summary with new information.
        """
        # Compact prompt for high-intelligence models
        prompt = f"""<|im_start|>system
You are an expert technical summarizer. You are maintaining a dense, running summary of a technical document.
<|im_end|>
<|im_start|>user
Current Summary:
{current_summary or "(None)"}

New Content to Incorporate:
{new_text}

Task: Update the 'Current Summary' to include key information from 'New Content'.
Constraints:
1. Keep the total length under {SUMMARY_BUDGET} tokens.
2. Do not lose previous key details.
3. Output ONLY the updated summary.
<|im_end|>
<|im_start|>assistant
"""
        output = self.llm.create_completion(
            prompt,
            max_tokens=SUMMARY_BUDGET + 100, # Buffer
            temperature=0.3, # Low temp for factual consistency
            stop=["<|im_end|>"]
        )
        result = output["choices"][0]["text"]
        return self.clean_thinking_tokens(result)

    def summarize_recursive(self, text: str, current_summary: str = "") -> str:
        """
        Recursively processes text chunks.
        1. If text + summary fits context, update summary.
        2. If not, split text and recurse.
        """
        # Heuristic token estimation (4 chars ~= 1 token)
        est_tokens = (len(text) + len(current_summary)) / 3.5

        # Base Case: Fits in processing window
        if est_tokens < (CTX_SIZE - SUMMARY_BUDGET - 1000):
            return self.generate_update(current_summary, text)
        
        # Recursive Case: Split text in half
        # Find the nearest sentence boundary in the middle
        mid = len(text) // 2
        split_match = re.search(r'[.!?]\s', text[mid:])
        
        if split_match:
            split_idx = mid + split_match.end()
        else:
            split_idx = mid # Fallback hard split

        part1 = text[:split_idx]
        part2 = text[split_idx:]

        # Process first half
        updated_summary = self.summarize_recursive(part1, current_summary)
        
        # Process second half using result of first
        final_summary = self.summarize_recursive(part2, updated_summary)
        
        return final_summary

    def process_section(self, section_text: str) -> str:
        """
        Reads a section paragraph-by-paragraph to build a rolling summary.
        """
        paragraphs = section_text.split('\n\n')
        running_summary = ""

        # Buffer paragraphs to reduce LLM calls (batching small paras)
        buffer = ""
        
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            buffer += "\n" + para
            
            # If buffer gets large enough, process it
            if len(buffer) > 1500: # ~400 tokens
                print(f"  > Processing chunk {i+1}/{len(paragraphs)}...")
                running_summary = self.summarize_recursive(buffer, running_summary)
                buffer = ""

        # Process remaining buffer
        if buffer:
            running_summary = self.summarize_recursive(buffer, running_summary)

        return running_summary

def main():
    input_path = _PROJECT_ROOT / "data" / "extracted_sections.json"
    output_path = _PROJECT_ROOT / "data" / "section_summaries.json"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run extraction first.")
        return

    # Initialize Summarizer
    summarizer = ThinkingSummarizer(MODEL_PATH, n_ctx=CTX_SIZE)

    with open(input_path, "r", encoding="utf-8") as f:
        sections = json.load(f)

    summaries = []
    total = len(sections)

    for i, section in enumerate(sections, 1):
        heading = section.get("heading", "Untitled")
        content = section.get("content", "")
        
        print(f"\n[{i}/{total}] Summarizing: {heading} ({len(content)} chars)")
        
        if not content.strip():
            summary_text = "No content."
        else:
            try:
                summary_text = summarizer.process_section(content)
            except Exception as e:
                print(f"Error processing section '{heading}': {e}")
                summary_text = "Error generating summary."

        summaries.append({
            "heading": heading,
            "summary": summary_text,
            "content_length": len(content)
        })
        
        # Incremental save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Summaries saved to {output_path}")

if __name__ == "__main__":
    main()