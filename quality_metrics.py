"""
Quality metrics for Voicebot Pipeline — STT / LLM / TTS
Đo lường end-to-end quality cho từng cuộc gọi.

Sử dụng:
    from quality_metrics import compute_wer, analyze_stt, analyze_llm, analyze_tts, summarize_metrics

Endpoint: GET /metrics/dashboard → tổng hợp từ call_metrics.jsonl
"""
import re
import json
import numpy as np
from datetime import datetime, timezone


# ============================================================
# 1. STT Metrics
# ============================================================

def compute_wer(reference, hypothesis):
    """Word Error Rate (WER) — edit distance ở mức từ.
    WER = (S + D + I) / N
    S=substitution, D=deletion, I=insertion, N=total words in reference.
    """
    ref_words = reference.lower().strip().split()
    hyp_words = hypothesis.lower().strip().split()
    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 1.0 if m > 0 else 0.0

    # DP matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost  # substitution
            )

    return round(d[n][m] / n, 4)


def analyze_stt(metrics_entry):
    """Phân tích chất lượng STT từ 1 entry metrics.
    Returns dict với các chỉ số quality.
    """
    result = {}

    # Confidence (avg_log_prob từ Whisper, range ~ -1.0 đến 0.0)
    conf = metrics_entry.get("stt_confidence")
    if conf is not None:
        result["stt_confidence"] = conf
        # Mapping: > -0.2 = tốt, -0.2 ~ -0.5 = trung bình, < -0.5 = kém
        if conf > -0.2:
            result["stt_confidence_grade"] = "good"
        elif conf > -0.5:
            result["stt_confidence_grade"] = "ok"
        else:
            result["stt_confidence_grade"] = "poor"

    # SNR (Signal-to-Noise Ratio)
    snr = metrics_entry.get("input_snr_est")
    if snr is not None:
        result["input_snr"] = snr
        if snr > 20:
            result["snr_grade"] = "clean"
        elif snr > 10:
            result["snr_grade"] = "noisy"
        else:
            result["snr_grade"] = "very_noisy"

    # Audio level (RMS)
    rms = metrics_entry.get("input_rms")
    if rms is not None:
        result["input_rms"] = rms
        if rms < 0.01:
            result["audio_grade"] = "silent"
        elif rms < 0.03:
            result["audio_grade"] = "quiet"
        elif rms > 0.3:
            result["audio_grade"] = "clipping"
        else:
            result["audio_grade"] = "normal"

    # Duration
    dur = metrics_entry.get("input_duration_s")
    if dur is not None:
        result["input_duration_s"] = dur
        if dur < 0.5:
            result["duration_grade"] = "too_short"
        elif dur > 30:
            result["duration_grade"] = "too_long"
        else:
            result["duration_grade"] = "ok"

    # Latency
    lat = metrics_entry.get("stt_latency_ms")
    if lat is not None:
        result["stt_latency_ms"] = lat
        # RTF = latency / audio_duration
        if dur and dur > 0:
            rtf = (lat / 1000) / dur
            result["stt_rtf"] = round(rtf, 3)  # < 1.0 = faster than realtime

    # Hallucination
    result["hallucination"] = metrics_entry.get("hallucination", False)

    # Transcript quality heuristics
    text = metrics_entry.get("transcript", "")
    result["transcript_len"] = len(text)
    result["transcript_words"] = len(text.split()) if text else 0

    # Detect likely STT garbage
    if text:
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        result["word_unique_ratio"] = round(unique_ratio, 3)
        # Low unique ratio = repetitive = likely hallucination
        if unique_ratio < 0.3 and len(words) > 5:
            result["stt_quality_flag"] = "repetitive"
        elif len(text) < 3:
            result["stt_quality_flag"] = "too_short"
        else:
            result["stt_quality_flag"] = "ok"

    return result


# ============================================================
# 2. LLM Metrics
# ============================================================

def analyze_llm(metrics_entry):
    """Phân tích chất lượng LLM response."""
    result = {}

    lat = metrics_entry.get("llm_latency_ms")
    if lat is not None:
        result["llm_latency_ms"] = lat
        if lat < 500:
            result["llm_speed_grade"] = "fast"
        elif lat < 1500:
            result["llm_speed_grade"] = "ok"
        else:
            result["llm_speed_grade"] = "slow"

    response_len = metrics_entry.get("response_len", 0)
    result["response_len"] = response_len

    # Tokens per second estimate (rough: 1 token ~ 4 chars Vietnamese)
    if lat and lat > 0 and response_len > 0:
        est_tokens = response_len / 4
        tps = est_tokens / (lat / 1000)
        result["est_tokens_per_sec"] = round(tps, 1)

    transcript = metrics_entry.get("transcript", "")
    result["user_words"] = len(transcript.split()) if transcript else 0

    return result


# ============================================================
# 3. TTS Metrics
# ============================================================

def analyze_tts(metrics_entry):
    """Phân tích chất lượng TTS."""
    result = {}

    lat = metrics_entry.get("tts_latency_ms")
    if lat is not None:
        result["tts_latency_ms"] = lat
        if lat < 500:
            result["tts_speed_grade"] = "fast"
        elif lat < 1500:
            result["tts_speed_grade"] = "ok"
        else:
            result["tts_speed_grade"] = "slow"

    response_len = metrics_entry.get("response_len", 0)
    if lat and lat > 0 and response_len > 0:
        # Chars per second synthesis speed
        cps = response_len / (lat / 1000)
        result["tts_chars_per_sec"] = round(cps, 1)

    return result


# ============================================================
# 4. End-to-End Metrics
# ============================================================

def analyze_e2e(metrics_entry):
    """End-to-end pipeline metrics."""
    result = {}

    total = metrics_entry.get("total_latency_ms")
    if total is not None:
        result["total_latency_ms"] = total
        # Target: < 2s for real-time voice
        if total < 1500:
            result["e2e_grade"] = "realtime"
        elif total < 3000:
            result["e2e_grade"] = "acceptable"
        else:
            result["e2e_grade"] = "slow"

    # Breakdown percentages
    stt_lat = metrics_entry.get("stt_latency_ms", 0)
    llm_lat = metrics_entry.get("llm_latency_ms", 0)
    tts_lat = metrics_entry.get("tts_latency_ms", 0)
    if total and total > 0:
        result["stt_pct"] = round(stt_lat / total * 100, 1)
        result["llm_pct"] = round(llm_lat / total * 100, 1)
        result["tts_pct"] = round(tts_lat / total * 100, 1)

    return result


# ============================================================
# 5. Dashboard Summary
# ============================================================

def summarize_metrics(metrics_file, last_n=100):
    """Tổng hợp metrics từ file JSONL → dashboard summary."""
    entries = []
    try:
        with open(metrics_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return {"error": "No metrics file found", "total_calls": 0}

    if not entries:
        return {"error": "No metrics data", "total_calls": 0}

    # Take last N entries
    entries = entries[-last_n:]

    # Collect arrays
    stt_lats = [e["stt_latency_ms"] for e in entries if "stt_latency_ms" in e]
    llm_lats = [e["llm_latency_ms"] for e in entries if "llm_latency_ms" in e]
    tts_lats = [e["tts_latency_ms"] for e in entries if "tts_latency_ms" in e]
    total_lats = [e["total_latency_ms"] for e in entries if "total_latency_ms" in e]
    snrs = [e["input_snr_est"] for e in entries if "input_snr_est" in e]
    confs = [e["stt_confidence"] for e in entries if "stt_confidence" in e]
    rmses = [e["input_rms"] for e in entries if "input_rms" in e]
    hallucinations = [e.get("hallucination", False) for e in entries]

    def _stats(arr):
        if not arr:
            return {}
        a = np.array(arr)
        return {
            "mean": round(float(a.mean()), 2),
            "median": round(float(np.median(a)), 2),
            "p95": round(float(np.percentile(a, 95)), 2),
            "min": round(float(a.min()), 2),
            "max": round(float(a.max()), 2),
        }

    # Quality grades
    stt_grades = [analyze_stt(e) for e in entries]
    e2e_grades = [analyze_e2e(e) for e in entries]

    snr_dist = {}
    for g in stt_grades:
        grade = g.get("snr_grade", "unknown")
        snr_dist[grade] = snr_dist.get(grade, 0) + 1

    e2e_dist = {}
    for g in e2e_grades:
        grade = g.get("e2e_grade", "unknown")
        e2e_dist[grade] = e2e_dist.get(grade, 0) + 1

    return {
        "total_calls": len(entries),
        "time_range": {
            "from": entries[0].get("timestamp", ""),
            "to": entries[-1].get("timestamp", ""),
        },
        "latency": {
            "stt_ms": _stats(stt_lats),
            "llm_ms": _stats(llm_lats),
            "tts_ms": _stats(tts_lats),
            "total_ms": _stats(total_lats),
        },
        "stt_quality": {
            "confidence": _stats(confs),
            "snr": _stats(snrs),
            "rms": _stats(rmses),
            "hallucination_rate": round(sum(hallucinations) / len(hallucinations) * 100, 1) if hallucinations else 0,
            "snr_distribution": snr_dist,
        },
        "e2e_quality": {
            "grade_distribution": e2e_dist,
            "target_realtime_pct": round(
                sum(1 for t in total_lats if t < 2000) / len(total_lats) * 100, 1
            ) if total_lats else 0,
        },
    }
