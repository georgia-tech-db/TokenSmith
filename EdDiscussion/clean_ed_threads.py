import json
import re
from html import unescape

INPUT_FILE = "threads_reduced.json"
OUTPUT_FILE = "threads_clean.json"

TAG_RE = re.compile(r"<[^>]+>")

def clean_html(text):
    if not isinstance(text, str):
        return ""

    cleaned = TAG_RE.sub("", text)

    cleaned = unescape(cleaned)

    cleaned = "\n".join(
        line.strip() for line in cleaned.splitlines() if line.strip()
    )

    return cleaned


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    cleaned_output = []

    for entry in data:
        thread_id = entry.get("id")
        raw_text = entry.get("combined_text", "")

        clean_text = clean_html(raw_text)

        cleaned_output.append({
            "id": thread_id,
            "clean_text": clean_text
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(cleaned_output, f, indent=2)

    print(f"Cleaned {len(cleaned_output)} threads -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
