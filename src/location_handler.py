"""
Location query detection and response handling.
"""
import re
from typing import List, Dict, Tuple


def is_location_query(text: str) -> bool:
    """
    Detect if a query is asking for location information.
    
    Args:
        text: The user's query text
        
    Returns:
        True if this is a location query, False otherwise
    """
    t = text.lower().strip()
    # Multiple patterns to catch various "where" question formats
    patterns = [
        r"^where\s+is\s+",  # "where is X"
        r"^where\s+can\s+i\s+find",  # "where can I find X"
        r"^where\s+do\s+i\s+find",   # "where do I find X"
        r"^where\s+is\s+.*\s+(located|found|discussed|covered|explained|described)",  # "where is X located/found/etc"
        r"^where\s+can\s+.*\s+(find|locate|get)",  # "where can I find X"
        r"^where\s+does\s+.*\s+(appear|occur|show)",  # "where does X appear"
        r"^in\s+which\s+(section|chapter|part)",  # "in which section is X"
        r"^what\s+(section|chapter|part).*",  # "what section covers X"
    ]
    return any(re.search(pattern, t) for pattern in patterns)


def format_location_response(topk_idxs: List[int], metadata: List[Dict], max_locations: int = 5) -> str:
    """
    Format a location response from the top retrieved chunks.
    
    Args:
        topk_idxs: List of chunk indices that were selected
        metadata: List of metadata dictionaries for each chunk
        max_locations: Maximum number of locations to return
        
    Returns:
        Formatted string with numbered location list
    """
    seen = set()
    locations = []
    
    for i in topk_idxs:
        sec = str(metadata[i].get("section", "")).strip()
        if sec.startswith("## "):
            sec = sec[3:].strip()
        if sec and sec not in seen:
            seen.add(sec)
            locations.append(sec)
        if len(locations) >= max_locations:
            break
    
    if locations:
        return "\n".join(f"{rank}. {s}" for rank, s in enumerate(locations, 1))
    else:
        return "(no matching sections found)"


def format_citations(topk_idxs: List[int], metadata: List[Dict], max_citations: int = 3) -> str:
    """
    Format inline citations from the top retrieved chunks.
    
    Args:
        topk_idxs: List of chunk indices that were selected
        metadata: List of metadata dictionaries for each chunk
        max_citations: Maximum number of citations to return
        
    Returns:
        Formatted citations string
    """
    seen = set()
    sections = []
    
    for i in topk_idxs:
        sec = str(metadata[i].get("section", "")).strip()
        if not sec:
            continue
        # remove markdown heading markers if present
        if sec.startswith("## "):
            sec = sec[3:].strip()
        if sec not in seen:
            seen.add(sec)
            sections.append(sec)
        if len(sections) >= max_citations:
            break
    
    if sections:
        return "; ".join(f"[{s}]" for s in sections)
    else:
        return ""
