import re
from typing import Optional
from llama_cpp import Llama

class ThinkingSummarizer:
    """Large-model summarizer with thinking tag cleaning."""

    def __init__(self, model_path: str, n_ctx: int = 40960):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            tensor_split=[24, 24],
            verbose=False,
            n_batch=512,
        )

    def clean_thinking_tokens(self, text: str) -> str:
        """Strip <think>/<reasoning> blocks while keeping the final answer."""
        if not text: return ""
        flags = re.DOTALL | re.IGNORECASE
        
        # Keep everything after last closing tag
        for tag in ["think", "thinking", "reasoning"]:
            m = re.search(rf'</{tag}>\s*(.*)$', text, flags=flags)
            if m: text = m.group(1)
            text = re.sub(rf'</?{tag}[^>]*>', '', text, flags=flags)

        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    def generate_update(self, current_summary: str, new_text: str, budget: int = 500) -> str:
        prompt = f"""<|im_start|>system
You are an expert technical summarizer maintaining a dense running summary.
<|im_end|>
<|im_start|>user
Current Summary: {current_summary or "(None)"}
New Content: {new_text}
Task: Update summary.
Constraints: length < {budget} tokens. No thinking tags.
<|im_end|>
<|im_start|>assistant
"""
        output = self.llm.create_completion(prompt, max_tokens=budget + 100, temperature=0.3, stop=["<|im_end|>"])
        return self.clean_thinking_tokens(output["choices"][0]["text"])

    def summarize_recursive(self, text: str, current_summary: str = "", budget: int = 500) -> str:
        # Simple recursion based on length
        est_tokens = (len(text) + len(current_summary)) / 3.5
        if est_tokens < (40960 - budget - 1000):
            return self.generate_update(current_summary, text, budget)

        mid = len(text) // 2
        # Simple split
        part1, part2 = text[:mid], text[mid:]
        updated = self.summarize_recursive(part1, current_summary, budget)
        return self.summarize_recursive(part2, updated, budget)

    def one_line(self, text: str) -> str:
        if not text.strip(): raise ValueError("Empty text")
        prompt = f"""<|im_start|>system
Write one concise sentence description (< 200 chars).
<|im_end|>
<|im_start|>user
Text: {text}
<|im_end|>
<|im_start|>assistant
"""
        output = self.llm.create_completion(prompt, max_tokens=64, temperature=0.3, stop=["<|im_end|>"])
        return self.clean_thinking_tokens(output["choices"][0]["text"])
