"""
src/llm_benchmark_generation/llm_client.py

Low-level OpenRouter HTTP client used by all pipeline stages.
Uses httpx for reliable SSL handling on macOS and a ThreadPoolExecutor
hard timeout to prevent silent hangs regardless of OS network stack quirks.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Optional

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ─────────────────────────────────────────────────────────────────────────────
# Client
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(
    messages:      list[dict],
    model:         str,
    api_key:       str,
    max_tokens:    int  = 8000,
    temperature:   float = 0.0,
    retries:       int  = 3,
    retry_delay:   int  = 10,
    connect_timeout: float = 10.0,
    read_timeout:    float = 180.0,
) -> tuple[Optional[str], Optional[dict]]:
    """
    POST a chat completion request to OpenRouter.

    Returns (response_text, usage_dict) on success.
    Returns (None, None) after all retries are exhausted.

    Uses a ThreadPoolExecutor hard timeout so the call never hangs
    silently regardless of OS / network stack behaviour.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }

    # Hard wall-clock timeout = connect + read, with a small buffer
    hard_timeout = connect_timeout + read_timeout + 5.0

    def _do_request() -> httpx.Response:
        with httpx.Client(
            timeout=httpx.Timeout(connect_timeout, read=read_timeout)
        ) as client:
            return client.post(OPENROUTER_URL, headers=headers, json=payload)

    for attempt in range(1, retries + 1):
        print(
            f"  [LLM] Attempt {attempt}/{retries} — {model} "
            f"(max_tokens={max_tokens:,}) ...",
            end=" ", flush=True,
        )
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_request)
                try:
                    resp = future.result(timeout=hard_timeout)
                except FuturesTimeout:
                    print(f"HARD TIMEOUT ({hard_timeout:.0f}s)")
                    if attempt < retries:
                        _wait(retry_delay, attempt)
                    continue

            resp.raise_for_status()
            body  = resp.json()
            text  = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage")
            print(
                f"OK  "
                f"[in:{(usage or {}).get('prompt_tokens', '?'):,}  "
                f"out:{(usage or {}).get('completion_tokens', '?'):,}]"
            )
            return text, usage

        except httpx.ConnectTimeout:
            print(f"CONNECT TIMEOUT ({connect_timeout:.0f}s)")
        except httpx.ReadTimeout:
            print(f"READ TIMEOUT ({read_timeout:.0f}s)")
        except httpx.HTTPStatusError as exc:
            print(f"HTTP {exc.response.status_code} — {exc.response.text[:120]}")
        except Exception as exc:
            print(f"ERROR — {type(exc).__name__}: {exc}")

        if attempt < retries:
            _wait(retry_delay, attempt)

    print(f"  [LLM] All {retries} attempts failed — giving up")
    return None, None


def _wait(base: int, attempt: int) -> None:
    delay = base * attempt
    print(f"  [LLM] Retrying in {delay}s ...")
    time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_response(raw: str) -> Optional[dict | list]:
    """
    Strip optional markdown fences and parse JSON.
    Falls back to finding the outermost { } or [ ] block.
    Returns None on total failure.
    """
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        for pattern in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pattern, cleaned, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
    print(f"  [WARN] JSON parse failed. First 200 chars: {raw[:200]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pricing fetch (for cost estimation)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_model_pricing(model_id: str, api_key: str) -> Optional[dict]:
    """
    Fetch per-token pricing for a model from the OpenRouter /api/v1/models
    endpoint. Returns {"input_per_1m": float, "output_per_1m": float} or None.

    The API returns pricing in USD per token; we convert to per-1M tokens.
    """
    url = "https://openrouter.ai/api/v1/models"
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
            resp = client.get(url, headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        models = resp.json().get("data", [])
        for m in models:
            if m.get("id") == model_id:
                pricing = m.get("pricing", {})
                inp  = float(pricing.get("prompt",     0)) * 1_000_000
                out  = float(pricing.get("completion", 0)) * 1_000_000
                return {"input_per_1m": inp, "output_per_1m": out, "model": model_id}
    except Exception as exc:
        print(f"  [WARN] Could not fetch pricing for {model_id}: {exc}")
    return None