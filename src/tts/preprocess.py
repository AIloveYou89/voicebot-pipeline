"""Vietnamese text normalization for TTS."""
import re
from typing import List

from num2words import num2words

PUNCS = r".?!…"

# Compiled regex patterns
_number_pattern = re.compile(r"(\d{1,3}(?:\.\d{3})*)(?:\s*(%|[^\W\d_]+))?", re.UNICODE)
_whitespace_pattern = re.compile(r"\s+")
_comma_pattern = re.compile(r"\s*,\s*")
_punct_spacing_pattern = re.compile(r"\s+([,;:])")
_repeated_punct_pattern = re.compile(rf"[{PUNCS}]{{2,}}")
_punct_no_space_pattern = re.compile(rf"([{PUNCS}])(?=\S)")


def normalize_text_vn(text: str) -> str:
    """Normalize Vietnamese text: whitespace, numbers, punctuation."""
    text = text.strip()
    text = _whitespace_pattern.sub(" ", text)
    text = _comma_pattern.sub(", ", text)
    text = text.lower()

    def repl_number_with_unit(m):
        num_str = m.group(1).replace(".", "")
        unit = m.group(2) or ""
        try:
            return num2words(int(num_str), lang="vi") + (" " + unit if unit else "")
        except Exception:
            return m.group(0)

    text = _number_pattern.sub(repl_number_with_unit, text)
    text = _punct_spacing_pattern.sub(r"\1", text)
    text = _repeated_punct_pattern.sub(lambda m: m.group(0)[0], text)
    text = _punct_no_space_pattern.sub(r"\1 ", text)
    return text.strip()


def _ensure_punctuation(s: str) -> str:
    s = s.strip()
    if not s.endswith(tuple(PUNCS)):
        s += "."
    return s


def _ensure_leading_dot(s: str) -> str:
    s = s.lstrip()
    if s and s[0] not in PUNCS:
        return ". " + s
    return s


def preprocess_for_tts(text: str) -> str:
    """
    Normalize Vietnamese text for TTS input.
    No chunking — pipeline sends full LLM response as one piece.

    Returns normalized text with leading dot (SparkTTS convention).
    """
    clean = normalize_text_vn(text)
    clean = _ensure_punctuation(clean)
    clean = _ensure_leading_dot(clean)
    return clean
