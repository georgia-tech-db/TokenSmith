"""
Agent orchestrator for the dynamic context budgeted agent.

Manages the investigation loop:
1. Investigation phase: SLM queries tools and manages context registry
2. Synthesis phase: Generate final answer from curated context
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

from src.agent.context_manager import ContextRegistry
from src.agent.tools import AgentToolkit
from src.generator import get_llama_model, ANSWER_END


@dataclass
class AgentConfig:
    reasoning_limit: int = 5
    tool_limit: int = 20
    max_reasoning_tokens: int = 500
    max_generation_tokens: int = 400


@dataclass
class AgentStep:
    thought: str
    tool_name: Optional[str]
    tool_args: Dict[str, Any]
    context_action: Dict[str, Any]
    signal: str


AGENT_SYSTEM_PROMPT = """You are an investigative agent that retrieves information to answer questions.

You work in a loop: think → use a tool → observe → repeat until ready.

{tool_descriptions}

## Output Format (strict JSON)
```json
{{
  "thought": "Your reasoning about current state and next steps",
  "tool_name": "name_of_tool or null if done",
  "tool_args": {{"arg1": "value1"}},
  "context_action": {{
    "keep": ["obs_1", "obs_3"],
    "discard": ["obs_2"],
    "notes": "Why keeping these"
  }},
  "signal": "continue or finish"
}}
```

## Rules
- Use search_index first to find relevant chunks
- Use read_content to get full text of promising chunks
- Use grep_text for exact matches (code, variables, specific terms)
- Signal "finish" when you have enough information
- Keep only observations needed for the final answer
- Discard observations that are irrelevant or redundant

Current observations in registry: {observation_ids}
"""

SYNTHESIS_PROMPT = """Based on the following curated context, answer the question concisely.

Context:
{context}

Question: {question}

Answer:"""


def parse_agent_response(text: str) -> Optional[AgentStep]:
    """Extract JSON from agent response."""
    text = text.strip()
    
    # Try to find JSON in markdown code block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to extract raw JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return None

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    return AgentStep(
        thought=data.get("thought", ""),
        tool_name=data.get("tool_name"),
        tool_args=data.get("tool_args", {}),
        context_action=data.get("context_action", {}),
        signal=data.get("signal", "continue"),
    )


class AgentOrchestrator:
    """Main agent loop coordinating tools, context, and LLM calls."""

    def __init__(
        self,
        toolkit: AgentToolkit,
        model_path: str,
        config: Optional[AgentConfig] = None,
    ):
        self.toolkit = toolkit
        self.model_path = model_path
        self.config = config or AgentConfig()
        self.registry = ContextRegistry()

    def _build_investigation_prompt(self, question: str, history: List[str]) -> str:
        """Build prompt for investigation step."""
        system = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=AgentToolkit.get_tool_descriptions(),
            observation_ids=self.registry.list_ids() or "[]",
        )

        history_text = "\n".join(history) if history else "No history yet."

        return f"""<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
Question: {question}

Investigation history:
{history_text}

What's your next step?
<|im_end|>
<|im_start|>assistant
```json
"""

    def _run_reasoning_step(self, prompt: str) -> str:
        """Run a single LLM call for reasoning."""
        model = get_llama_model(self.model_path)
        result = model.create_completion(
            prompt,
            max_tokens=self.config.max_reasoning_tokens,
            temperature=0.1,
            stop=["```\n", "<|im_end|>"],
        )
        return result["choices"][0]["text"]

    def _apply_context_action(self, action: Dict[str, Any]) -> None:
        """Apply keep/discard actions to the registry."""
        discard_ids = action.get("discard", [])
        if discard_ids:
            self.registry.prune(discard_ids)

    def investigate(self, question: str) -> List[str]:
        """
        Run investigation phase.
        Returns list of observation IDs to use for synthesis.
        """
        history: List[str] = []
        reasoning_count = 0
        tool_count = 0

        while reasoning_count < self.config.reasoning_limit:
            reasoning_count += 1

            prompt = self._build_investigation_prompt(question, history)
            response = self._run_reasoning_step(prompt)

            step = parse_agent_response(response)
            if step is None:
                history.append(f"Step {reasoning_count}: [Parse error] {response[:200]}")
                continue

            history.append(f"Step {reasoning_count}: {step.thought}")

            if step.signal == "finish" or step.tool_name is None:
                keep_ids = step.context_action.get("keep", self.registry.list_ids())
                return keep_ids

            if tool_count >= self.config.tool_limit:
                history.append(f"Tool limit ({self.config.tool_limit}) reached.")
                return step.context_action.get("keep", self.registry.list_ids())

            tool_count += 1
            observation = self.toolkit.execute(step.tool_name, step.tool_args)
            ref_id = self.registry.add_observation(observation)
            history.append(f"  Tool: {step.tool_name}({step.tool_args}) → {ref_id}")

            self._apply_context_action(step.context_action)

        return self.registry.list_ids()

    def synthesize(self, question: str, keep_ids: List[str]) -> str:
        """Generate final answer from curated context."""
        context = self.registry.get_context(keep_ids)

        prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions based on the provided context.
<|im_end|>
<|im_start|>user
{SYNTHESIS_PROMPT.format(context=context, question=question)}
<|im_end|>
<|im_start|>assistant
"""

        model = get_llama_model(self.model_path)
        result = model.create_completion(
            prompt,
            max_tokens=self.config.max_generation_tokens,
            temperature=0.2,
            stop=[ANSWER_END, "<|im_end|>"],
        )
        return result["choices"][0]["text"].strip()

    def run(self, question: str) -> Dict[str, Any]:
        """
        Full agent run: investigate → synthesize.
        Returns dict with answer, observations, and metadata.
        """
        self.registry.clear()

        keep_ids = self.investigate(question)
        answer = self.synthesize(question, keep_ids)

        return {
            "answer": answer,
            "kept_observations": keep_ids,
            "total_observations": len(self.registry),
            "context_used": self.registry.get_context(keep_ids),
        }

    def stream_run(self, question: str):
        """
        Generator version of run for streaming output.
        Yields status updates during investigation, then final answer.
        """
        self.registry.clear()

        yield {"type": "status", "message": "Starting investigation..."}

        history: List[str] = []
        reasoning_count = 0
        tool_count = 0
        keep_ids = []

        while reasoning_count < self.config.reasoning_limit:
            reasoning_count += 1

            prompt = self._build_investigation_prompt(question, history)
            response = self._run_reasoning_step(prompt)

            step = parse_agent_response(response)
            if step is None:
                history.append(f"Step {reasoning_count}: [Parse error] {response[:100]}")
                yield {"type": "status", "message": f"Parse error at step {reasoning_count}, retrying..."}
                continue

            yield {"type": "thought", "step": reasoning_count, "thought": step.thought}
            history.append(f"Step {reasoning_count}: {step.thought}")

            if step.signal == "finish" or step.tool_name is None:
                keep_ids = step.context_action.get("keep", self.registry.list_ids())
                break

            if tool_count >= self.config.tool_limit:
                keep_ids = step.context_action.get("keep", self.registry.list_ids())
                break

            tool_count += 1
            yield {
                "type": "tool",
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
            }

            observation = self.toolkit.execute(step.tool_name, step.tool_args)
            ref_id = self.registry.add_observation(observation)
            history.append(f"  Tool: {step.tool_name} → {ref_id}")

            self._apply_context_action(step.context_action)

        if not keep_ids:
            keep_ids = self.registry.list_ids()

        yield {"type": "status", "message": "Generating answer..."}

        answer = self.synthesize(question, keep_ids)

        yield {
            "type": "answer",
            "answer": answer,
            "kept_observations": keep_ids,
        }

