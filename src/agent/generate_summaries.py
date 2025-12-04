"""
Generate dense summaries using sliding window recursive approach.
"""

import json
from pathlib import Path
from typing import List, Dict
from llama_cpp import Llama

SUMMARY_MODEL_PATH = "models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf"
CHUNK_SIZE = 3000
OVERLAP = 500
MAX_SUMMARY_TOKENS = 150


def load_sections(sections_path: str) -> List[Dict]:
    with open(sections_path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def summarize_chunk(text: str, model: Llama) -> str:
    """Generate dense summary of a single chunk."""
    prompt = f"""<|im_start|>system
You are a technical summarizer. Extract only key facts and concepts. Be dense and factual. No fluff.
<|im_end|>
<|im_start|>user
Summarize this text. Include only essential information:

{text}
<|im_end|>
<|im_start|>assistant
"""
    result = model.create_completion(
        prompt,
        max_tokens=MAX_SUMMARY_TOKENS,
        temperature=0.0,
        stop=["<|im_end|>"],
    )
    return result["choices"][0]["text"].strip()


def summarize_recursive(text: str, model: Llama) -> str:
    """Recursively summarize large text using sliding window."""
    if len(text) <= CHUNK_SIZE:
        return summarize_chunk(text, model)

    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    chunk_summaries = []

    for chunk in chunks:
        summary = summarize_chunk(chunk, model)
        chunk_summaries.append(summary)

    combined = "\n\n".join(chunk_summaries)

    if len(combined) <= CHUNK_SIZE:
        return summarize_chunk(combined, model)

    return summarize_recursive(combined, model)


def generate_all_summaries(
    sections_path: str,
    output_path: str,
    model_path: str = SUMMARY_MODEL_PATH,
) -> None:
    """Generate summaries for all sections using recursive sliding window."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    sections = load_sections(sections_path)
    summaries = []

    for i, section in enumerate(sections, 1):
        heading = section.get("heading", "")
        content = section.get("content", "")

        if not content:
            summaries.append({
                "heading": heading,
                "summary": "",
                "content_length": 0,
            })
            continue

        print(f"[{i}/{len(sections)}] {heading[:60]}")

        summary = summarize_recursive(content, model)

        summaries.append({
            "heading": heading,
            "summary": summary,
            "content_length": len(content),
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(summaries)} summaries → {output_path}")


if __name__ == "__main__":
    generate_all_summaries(
        sections_path="data/extracted_sections.json",
        output_path="data/section_summaries.json",
        model_path=SUMMARY_MODEL_PATH,
    )

