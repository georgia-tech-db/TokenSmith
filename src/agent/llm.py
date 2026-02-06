from typing import List, Optional
from src.generator import get_llama_model

class AgentLLM:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def completion(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1, stop: Optional[List[str]] = None) -> str:
        model = get_llama_model(self.model_path)
        if stop is None:
            stop = ["<|im_end|>"]
            
        result = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return result["choices"][0]["text"]
