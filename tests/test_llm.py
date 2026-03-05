"""Test LLM module (requires GPU)."""


def test_generate_basic():
    """Test basic generation with a simple prompt."""
    from src.llm.qwen_llm import generate

    messages = [{"role": "user", "content": "Xin chao, ban khoe khong?"}]
    response, latency = generate(messages)

    assert isinstance(response, str)
    assert len(response) > 0, "Expected non-empty response"
    assert isinstance(latency, float)
    assert latency > 0
    print(f"[PASS] LLM: '{response[:80]}...', latency={latency:.2f}s")


def test_generate_with_history():
    """Test generation with conversation history."""
    from src.llm.qwen_llm import generate

    messages = [
        {"role": "user", "content": "Ban ten gi?"},
        {"role": "assistant", "content": "Toi la tro ly AI."},
        {"role": "user", "content": "Ban co the giup gi cho toi?"},
    ]
    response, latency = generate(messages)

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"[PASS] LLM history: '{response[:80]}...', latency={latency:.2f}s")


def test_generate_custom_prompt():
    """Test generation with custom system prompt."""
    from src.llm.qwen_llm import generate

    messages = [{"role": "user", "content": "Hello"}]
    response, latency = generate(
        messages,
        system_prompt="You are a pirate. Reply in pirate speak.",
        max_new_tokens=50,
    )

    assert len(response) > 0
    print(f"[PASS] LLM custom prompt: '{response[:80]}...', latency={latency:.2f}s")


if __name__ == "__main__":
    test_generate_basic()
    test_generate_with_history()
    test_generate_custom_prompt()
