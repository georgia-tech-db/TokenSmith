import json
from pathlib import Path
from src.agent.summarizer import ThinkingSummarizer
from src.agent.summary_db import init_db, save_section_to_db

_PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTRACTED_PATH = _PROJECT_ROOT / "data" / "extracted_sections.json"
NAV_DB_PATH = _PROJECT_ROOT / "data" / "nav_index.sqlite3"
MODEL_PATH = str(_PROJECT_ROOT / "models" / "Qwen3-30B-A3B-Q6_K.gguf")

def process_all():
    if not EXTRACTED_PATH.exists():
        print("No extracted sections found.")
        return

    with open(EXTRACTED_PATH) as f:
        sections = json.load(f)

    summarizer = ThinkingSummarizer(MODEL_PATH)
    init_db(NAV_DB_PATH)

    for i, sec in enumerate(sections):
        print(f"Processing section {i+1}/{len(sections)}...")
        text = sec.get("content", "")
        if not text: continue

        # Dense summary
        dense = summarizer.summarize_recursive(text)
        
        # One liner
        one_line = summarizer.one_line(text)

        # Paragraphs
        paras = []
        for p_idx, para in enumerate(text.split("\n\n")):
            if not para.strip(): continue
            p_summary = summarizer.one_line(para)
            paras.append({
                "index": p_idx,
                "text": para,
                "summary": p_summary
            })

        save_section_to_db(NAV_DB_PATH, i+1, sec.get("heading", ""), dense, one_line, paras)

if __name__ == "__main__":
    process_all()
