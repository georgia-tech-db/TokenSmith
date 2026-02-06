"""Minimal logging for agent pipeline."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class AgentLogger:
    """Logs agent interactions to JSONL file."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path("logs") / "agent"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / f"agent_{self.session_id}.jsonl"

    def _write(self, data: Dict[str, Any]) -> None:
        data["ts"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_step(self, step: int, thought: str, tool_name: Optional[str], tool_args: Dict[str, Any], result: Optional[str], success: bool) -> None:
        """Log a reasoning step with full thought (no truncation)."""
        self._write({
            "event": "step",
            "step": step,
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "success": success,
        })

    def log_query_complete(self, question: str, answer: str, registry_metadata: Dict[str, Dict[str, Any]]) -> None:
        """Log query completion with full registry lifecycle."""
        self._write({
            "event": "query_complete",
            "question": question,
            "answer": answer,
            "registry_entries": registry_metadata,
        })
