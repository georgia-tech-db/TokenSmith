"""
Model factories for the LlamaIndex pipeline.

- LLM:        Qwen 2.5 1.5B GGUF via llama-cpp-python  (same as original)
- Embeddings: Qwen3-Embedding-4B Q5_K_M GGUF via llama-cpp-python (same as original)
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.llama_cpp import LlamaCPP

from .config import LlamaIndexConfig


# ── Prompt formatting for Qwen / ChatML ──────────────────────────────────


def _messages_to_prompt(messages) -> str:
    prompt = ""
    for m in messages:
        role = m.role.value if hasattr(m.role, "value") else m.role
        prompt += f"<|im_start|>{role}\n{m.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def _completion_to_prompt(completion: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{completion}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── GGUF Embedding model ────────────────────────────────────────────────


class LlamaCppEmbedding(BaseEmbedding):
    """Wraps llama-cpp-python for GGUF embedding models."""

    _model: Any = PrivateAttr()

    def __init__(self, model_path: str, n_gpu_layers: int = -1, n_ctx: int = 4096, **kwargs):
        super().__init__(**kwargs)
        from llama_cpp import Llama

        self._model = Llama(
            model_path=model_path,
            embedding=True,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def _embed(self, text: str) -> List[float]:
        output = self._model.embed(text)
        vec = np.array(output, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# ── Factories ────────────────────────────────────────────────────────────


def build_llm(cfg: LlamaIndexConfig) -> LlamaCPP:
    return LlamaCPP(
        model_path=cfg.gen_model,
        temperature=cfg.gen_temperature,
        max_new_tokens=cfg.max_gen_tokens,
        context_window=cfg.gen_context_window,
        model_kwargs={"n_gpu_layers": cfg.n_gpu_layers},
        messages_to_prompt=_messages_to_prompt,
        completion_to_prompt=_completion_to_prompt,
        verbose=False,
    )


def build_embed_model(cfg: LlamaIndexConfig) -> LlamaCppEmbedding:
    return LlamaCppEmbedding(
        model_path=cfg.embed_model,
        n_gpu_layers=cfg.n_gpu_layers,
        n_ctx=cfg.embed_n_ctx,
        n_batch=512
    )
