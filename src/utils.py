import re

def text_cleaning(prompt):
    _CONTROL_CHARS_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")
    _DANGEROUS_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
        r"system\s+override",
        r"reveal\s+prompt",
    ]
    text = _CONTROL_CHARS_RE.sub("", prompt)
    text = re.sub(r"\s+", " ", text).strip()
    for pat in _DANGEROUS_PATTERNS:
        text = re.sub(pat, "[FILTERED]", text, flags=re.IGNORECASE)
    return text
