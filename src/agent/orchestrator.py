import json
import re
from typing import Optional, List, Any, Generator, Dict

from src.agent.types import AgentConfig, AgentStep
from src.agent.context import ContextRegistry, ContextBudgetExceeded
from src.agent.toolkit import AgentToolkit
from src.agent.llm import AgentLLM
from src.agent.prompts import INVESTIGATION_PROMPT_TEMPLATE, SYNTHESIS_PROMPT, AGENT_SYSTEM_PROMPT

class AgentOrchestrator:
    def __init__(self, toolkit: AgentToolkit, model_path: str, config: Optional[AgentConfig] = None, logger: Optional[Any] = None):
        self.toolkit = toolkit
        self.config = config or AgentConfig(model_path=model_path)
        self.llm = AgentLLM(model_path)
        self.registry = ContextRegistry(max_tokens=self.config.max_context_tokens)
        self.logger = logger

    def _parse_step(self, text: str) -> Optional[AgentStep]:
        text = text.strip()
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
        if not match: return None
        
        try:
            data = json.loads(match.group(1 if match.lastindex else 0))
            
            # Handle hallucinated "next_step": "tool(args)" format
            if "next_step" in data and not data.get("tool_name"):
                call_str = data["next_step"]
                # Parse tool(arg=val)
                m_call = re.match(r"(\w+)\((.*)\)", call_str)
                if m_call:
                    tool_name = m_call.group(1)
                    args_str = m_call.group(2)
                    args = {}
                    # Simple arg parser for key=value or key="value"
                    # This is brittle but handles the example seen
                    for arg_pair in args_str.split(","):
                        if "=" in arg_pair:
                            k, v = arg_pair.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'").strip('"')
                            # Try to convert to int/float
                            try:
                                if "." in v: v = float(v)
                                else: v = int(v)
                            except ValueError:
                                pass
                            args[k] = v
                    
                    return AgentStep(
                        thought=data.get("thought", f"Decided to call {tool_name}"),
                        tool_name=tool_name,
                        tool_args=args,
                        context_action={},
                        signal="continue"
                    )

            return AgentStep(
                thought=data.get("thought", "Recovering from simplified output..."),
                tool_name=data.get("tool_name"),
                tool_args=data.get("tool_args", {}),
                context_action=data.get("context_action", {}),
                signal=data.get("signal", "continue"),
            )
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            return None

    def _build_prompt(self, question: str, history: List[str]) -> str:
        obs_ids = self.registry.list_ids()
        
        # Format active context
        active_ctx = []
        read_chunk_ids = set()
        for ref_id in obs_ids:
            content = self.registry.get(ref_id)
            if content:
                active_ctx.append(f"[{ref_id}]\n{content}")
                # Track seen chunks (matches "Chunk 123" or "chunk_id=123")
                matches = re.findall(r"(?:Chunk|chunk_id=)\s*(\d+)", content)
                read_chunk_ids.update(int(c) for c in matches)

        full_context = "\n\n".join(active_ctx) if active_ctx else "No observations yet."
        
        # Lifecycle metadata
        meta = self.registry.get_all_metadata()
        lifecycle = [f"{k}: {v['lifecycle']}" for k, v in meta.items() if v['lifecycle'] != "no-events"]
        
        status = self.registry.status
        system_text = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=self.toolkit.get_tool_descriptions(),
            observation_ids=str(obs_ids),
            budget_status=f"{status['used']}/{status['total']} tokens"
        )
        
        return INVESTIGATION_PROMPT_TEMPLATE.format(
            system=system_text,
            question=question,
            full_context=full_context,
            read_chunks_str=str(sorted(list(read_chunk_ids))),
            lifecycle_str="\n".join(lifecycle) if lifecycle else "None",
            history_text="\n".join(history[-10:]) or "None"
        )

    def stream_run(self, question: str) -> Generator[Dict[str, Any], None, None]:
        """
        Run investigation and synthesis, yielding events.
        Events:
          - {"type": "thought", "step": int, "thought": str}
          - {"type": "tool", "tool_name": str, "tool_args": dict}
          - {"type": "answer", "answer": str, "kept_observations": List[str]}
        """
        self.registry.clear()
        
        # Seed initial search
        res, _ = self.toolkit.get_initial_context(question)
        self.registry.add(f"Initial search: {res}", step=0)

        history = []
        steps = 0
        keep_ids = []
        
        # --- Investigation Phase ---
        while steps < self.config.reasoning_limit:
            steps += 1
            # print(f"--- Step {steps} ---")
            prompt = self._build_prompt(question, history)
            
            response = self.llm.completion(prompt, max_tokens=self.config.max_reasoning_tokens)
            print(f"\n[DEBUG RAW RESPONSE]\n{response}\n[END DEBUG]\n")
            step = self._parse_step(response)
            
            if not step:
                # print("Failed to parse response")
                history.append(f"Step {steps}: [Use stricter JSON format]")
                continue
            
            # Yield thought event
            yield {
                "type": "thought",
                "step": steps,
                "thought": step.thought
            }
            
            # print(f"Thought: {step.thought}")
            # print(f"Tool: {step.tool_name}")
            
            history.append(f"Step {steps}: {step.thought} (Tool: {step.tool_name})")
            if self.logger:
                self.logger.log_step(steps, step.thought, step.tool_name, step.tool_args, None, True)

            # Handle context actions
            if step.context_action:
                for ref_id in step.context_action.get("discard", []):
                    self.registry.remove(ref_id, step=steps)

            # Finish logic
            if step.signal == "finish" or not step.tool_name:
                # Basic check: do we have any content?
                has_content = any("Chunk" in (self.registry.get(oid) or "") for oid in self.registry.list_ids())
                if has_content or steps >= self.config.reasoning_limit:
                    keep_ids = step.context_action.get("keep", self.registry.list_ids())
                    break
                
                # If trying to finish without content, force one more step
                history.append("System: You have search results but no full content. Read chunks before finishing.")
                continue

            # Yield tool event
            yield {
                "type": "tool",
                "tool_name": step.tool_name,
                "tool_args": step.tool_args
            }

            # Execute tool
            res, success = self.toolkit.execute(step.tool_name, step.tool_args)
            try:
                self.registry.add(f"Tool {step.tool_name} result:\n{res}", step=steps)
            except ContextBudgetExceeded:
                history.append("System: Context full. Discard irrelevant observations.")

            # Force Read check
            if step.tool_name == "search_index":
                 search_count = sum(1 for h in history if "Tool: search_index" in h)
                 if search_count >= 2:
                     history.append("System: You have enough search results. You MUST now use `read_content` to read a chunk. Do not search again.")

        if not keep_ids:
            keep_ids = self.registry.list_ids()

        # --- Synthesis Phase ---
        parts = []
        for ref_id in keep_ids:
            c = self.registry.get(ref_id)
            if c: parts.append(f"[{ref_id}]\n{c}")
        final_context = "\n\n".join(parts)
        
        prompt = SYNTHESIS_PROMPT.format(context=final_context, question=question)
        answer_text = self.llm.completion(prompt, max_tokens=self.config.max_generation_tokens)
        
        yield {
            "type": "answer",
            "answer": answer_text,
            "kept_observations": keep_ids
        }

    def investigate(self, question: str) -> List[str]:
        # Legacy/Support method wrapping stream_run
        last_ids = []
        for event in self.stream_run(question):
            if event["type"] == "answer":
                last_ids = event["kept_observations"]
        return last_ids

    def answer(self, question: str) -> str:
        # Legacy/Support method wrapping stream_run
        ans = ""
        for event in self.stream_run(question):
            if event["type"] == "answer":
                ans = event["answer"]
        return ans
