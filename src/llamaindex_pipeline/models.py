"""
Model factories for the LlamaIndex pipeline.

- LLM: Qwen 2.5 1.5B via llama-cpp-python (same model as TokenSmith)
- Embeddings: HuggingFace sentence-transformer (<5B params)
"""

from __future__ import annotations

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from .config import LlamaIndexConfig


# ── Prompt formatting for Qwen / ChatML ─────────────────────────────────


def _messages_to_prompt(messages) -> str:
    """Convert LlamaIndex messages to ChatML format expected by Qwen."""
    prompt = ""
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else m.role
        prompt += f"<|im_start|>{role}\n{m.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def _completion_to_prompt(completion: str) -> str:
    """Wrap a bare completion string in ChatML user format."""
    return (
        "<|im_start|>user\n"
        f"{completion}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── Factories ────────────────────────────────────────────────────────────


def build_llm(cfg: LlamaIndexConfig) -> LlamaCPP:
    """Build a LlamaCPP LLM instance from config."""
    return LlamaCPP(
        model_path=cfg.gen_model_path,
        temperature=cfg.gen_temperature,
        max_new_tokens=cfg.gen_max_tokens,
        context_window=cfg.gen_context_window,
        model_kwargs={"n_gpu_layers": cfg.n_gpu_layers},
        messages_to_prompt=_messages_to_prompt,
        completion_to_prompt=_completion_to_prompt,
        verbose=False,
    )


def build_embed_model(cfg: LlamaIndexConfig) -> HuggingFaceEmbedding:
    """Build a HuggingFace embedding model from config."""
    return HuggingFaceEmbedding(
        model_name=cfg.embed_model_name,
        device=cfg.embed_device,
        embed_batch_size=cfg.embed_batch_size,
    )
