"""LLM module — Qwen2.5-7B-Instruct-AWQ inference on GPU."""
import time
import threading
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

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


def _prepare_inputs(
    messages: list[dict],
    system_prompt: str | None = None,
):
    """Prepare tokenized inputs from messages."""
    load_model()
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    full_messages = [{"role": "system", "content": sys_prompt}] + messages

    text = _tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return _tokenizer(text, return_tensors="pt").to(DEVICE)


def generate(
    messages: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, float]:
    """
    Generate a full response (no streaming) for Phase 1.

    Returns:
        (response_text, latency_seconds)
    """
    inputs = _prepare_inputs(messages, system_prompt)
    max_tokens = max_new_tokens or LLM_MAX_NEW_TOKENS
    temp = temperature or LLM_TEMPERATURE

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

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    response = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    logger.info(f"[LLM] '{response[:80]}...' ({latency:.2f}s, {len(new_tokens)} tokens)")
    return response, latency


def generate_stream(
    messages: list[dict],
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
) -> Generator[str, None, None]:
    """
    Stream tokens as they are generated (Phase 2).

    Yields token strings one at a time. LLM runs on a background thread,
    tokens are yielded on the calling thread via TextIteratorStreamer.
    """
    inputs = _prepare_inputs(messages, system_prompt)
    max_tokens = max_new_tokens or LLM_MAX_NEW_TOKENS
    temp = temperature or LLM_TEMPERATURE

    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.8,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
    }

    # Run generate() on background thread — tokens flow via streamer
    thread = threading.Thread(
        target=lambda: _model.generate(**gen_kwargs),
        daemon=True,
    )
    thread.start()

    logger.info("[LLM] Streaming started...")
    for token_text in streamer:
        if token_text:
            yield token_text

    thread.join()
    logger.info("[LLM] Streaming complete")
