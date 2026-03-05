"""Test TTS module (requires GPU)."""
import numpy as np


def test_synthesize_short():
    """Test TTS with a short Vietnamese text."""
    from src.tts.spark_tts import synthesize

    audio, sr, latency = synthesize("Xin chao, toi la tro ly AI.")

    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0, "Expected non-empty audio"
    assert sr == 24000
    assert latency > 0
    duration = len(audio) / sr
    print(f"[PASS] TTS short: {duration:.2f}s audio, latency={latency:.2f}s")


def test_synthesize_longer():
    """Test TTS with longer text."""
    from src.tts.spark_tts import synthesize

    text = "Da xin chao anh chi. Em la Minh Anh ben Sunshine. Hom nay em goi moi anh chi tham quan du an nghi duong tai Phu Quoc."
    audio, sr, latency = synthesize(text)

    assert len(audio) > 0
    duration = len(audio) / sr
    assert duration > 1.0, "Expected at least 1s of audio for longer text"
    print(f"[PASS] TTS long: {duration:.2f}s audio, latency={latency:.2f}s")


def test_preprocess():
    """Test Vietnamese text preprocessing (no GPU needed)."""
    from src.tts.preprocess import preprocess_for_tts

    result = preprocess_for_tts("Xin chao 10.000 ban!")
    assert result.startswith(".")
    assert "10.000" not in result  # Should be converted to words
    print(f"[PASS] Preprocess: '{result}'")


if __name__ == "__main__":
    test_preprocess()
    test_synthesize_short()
    test_synthesize_longer()
