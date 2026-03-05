"""LLM module — Qwen2.5-7B-Instruct-AWQ inference on GPU."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    logger, DEVICE, HF_TOKEN,
    QWEN_MODEL, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT,
)

_model = None
_tokenizer = None


def load_model():
    """Load Qwen2.5-7B-Instruct-AWQ (INT4, ~5GB VRAM)."""
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info(f"[LLM] Loading {QWEN_MODEL}...")
    t0 = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL,
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    logger.info(f"[LLM] Loaded in {time.time() - t0:.1f}s")


def generate(
    messages: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, float]:
    """
    Generate a full response (no streaming) for Phase 1.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}.
        system_prompt: Override default system prompt.
        max_new_tokens: Override default max tokens.
        temperature: Override default temperature.

    Returns:
        (response_text, latency_seconds)
    """
    load_model()

    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    max_tokens = max_new_tokens or LLM_MAX_NEW_TOKENS
    temp = temperature or LLM_TEMPERATURE

    # Build chat messages with system prompt
    full_messages = [{"role": "system", "content": sys_prompt}] + messages

    # Apply chat template
    text = _tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_p=0.8,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
        )
    latency = time.time() - t0

    # Decode only new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    logger.info(f"[LLM] '{response[:80]}...' ({latency:.2f}s, {len(new_tokens)} tokens)")
    return response, latency
