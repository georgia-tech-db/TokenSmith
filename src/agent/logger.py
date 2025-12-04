"""
Logging for agent pipeline - captures all LLM inputs/outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class AgentLogger:
    """Logs all LLM interactions in the agent pipeline."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path("logs") / "agent"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.logs_dir / f"agent_{self.session_id}.jsonl"
        self.step_count = 0
        self.query_count = 0

    def _write(self, data: Dict[str, Any]) -> None:
        data["timestamp"] = datetime.now().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_session_start(self, config: Dict[str, Any]) -> None:
        self._write({
            "event": "session_start",
            "session_id": self.session_id,
            "config": config,
        })

    def log_query_start(self, question: str) -> None:
        self.query_count += 1
        self.step_count = 0
        self._write({
            "event": "query_start",
            "query_id": self.query_count,
            "question": question,
        })

    def log_reasoning_step(
        self,
        prompt: str,
        response: str,
        parsed_step: Optional[Dict[str, Any]],
    ) -> None:
        self.step_count += 1
        self._write({
            "event": "reasoning_step",
            "query_id": self.query_count,
            "step": self.step_count,
            "prompt": prompt,
            "response": response,
            "parsed": parsed_step,
            "parse_success": parsed_step is not None,
        })

    def log_tool_execution(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        result: str,
        ref_id: str,
    ) -> None:
        self._write({
            "event": "tool_execution",
            "query_id": self.query_count,
            "step": self.step_count,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "result": result,
            "ref_id": ref_id,
        })

    def log_synthesis(
        self,
        prompt: str,
        response: str,
        keep_ids: list,
    ) -> None:
        self._write({
            "event": "synthesis",
            "query_id": self.query_count,
            "prompt": prompt,
            "response": response,
            "keep_ids": keep_ids,
        })

    def log_query_complete(self, answer: str, metadata: Dict[str, Any]) -> None:
        self._write({
            "event": "query_complete",
            "query_id": self.query_count,
            "total_steps": self.step_count,
            "answer": answer,
            "metadata": metadata,
        })

