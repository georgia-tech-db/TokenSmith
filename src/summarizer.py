#!/usr/bin/env python3
"""Summarization utilities for building recursive section summaries."""

import os
import textwrap
from typing import Dict, List, Optional

from tqdm import tqdm

from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning


def generate_section_summaries(
    sections: List[Dict],
    model_path: Optional[os.PathLike] = None,
    max_summary_tokens: int = 120,
    chunk_size: int = 5000,
    overlap: int = 500,
) -> List[Dict]:
    """Recursively summarize each section, handling very long content."""
    if model_path is None:
        raise ValueError("model_path must not be null.")

    summaries = []

    print("=" * 60)
    print(f"Generating recursive summaries for {len(sections)} sections...")
    print("=" * 60)

    def summarize_text_block(text: str, heading: str) -> str:
        """Summarize a single text block with the configured LLM."""
        prompt = textwrap.dedent(f"""\
            <|im_start|>system
            You are a textbook summarizer. Write a concise, factual summary (2-3 sentences)
            capturing the key ideas and technical details in a clear academic tone.
            <|im_end|>
            <|im_start|>user
            Section: {heading}

            Content:
            {text[:chunk_size + overlap]}

            Provide the summary for the above section. End the answer with {ANSWER_END}.
            <|im_end|>
            <|im_start|>assistant
            {ANSWER_START}
            Summary:""")
        cleaned = text_cleaning(prompt)
        return run_llama_cpp(cleaned, str(model_path), max_tokens=max_summary_tokens).strip()

    def recursive_summarize(text: str, heading: str) -> str:
        """Break text into overlapping chunks, summarize each, and merge recursively."""
        if len(text) <= chunk_size:
            return summarize_text_block(text, heading)

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap  # slide window with overlap

        partial_summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  Summarizing chunk {i}/{len(chunks)} of section '{heading}'...")
            summary = summarize_text_block(chunk, f"{heading} (Part {i})")
            partial_summaries.append(summary)

        merged_text = " ".join(partial_summaries)
        if len(merged_text) > chunk_size:
            print(f"  Recursively summarizing merged text for '{heading}'...")
            return recursive_summarize(merged_text, heading)
        return summarize_text_block(merged_text, f"{heading} (Final Summary)")

    for section in tqdm(sections, desc="Summarizing sections"):
        heading = section.get("heading", "Untitled Section")
        content = section.get("content", "")

        try:
            summary_text = recursive_summarize(content, heading)
            summaries.append({
                "heading": heading,
                "summary": summary_text.strip(),
                "original_length": len(content),
                "type": "llm_recursive"
            })
        except Exception as exc:
            print(f"Error summarizing section '{heading}': {exc}")
            summaries.append({
                "heading": heading,
                "summary": f"[Error summarizing: {exc}]",
                "original_length": len(content),
                "type": "error"
            })

    print(f"Generated {len(summaries)} section summaries (recursive mode).")
    for s in summaries:
        print(f"\n--- Summary for section: {s['heading']} ---\n{s['summary']}\n")

    return summaries
