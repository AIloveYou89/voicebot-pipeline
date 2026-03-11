"""
Reloadable handlers for voicebot server.
Hot-reload: POST /reload — models stay in GPU memory, only this file reloads.

Edit THIS file for logic changes (prompts, patterns, TTS params, etc.)
then call POST /reload to apply without restarting.
"""
import os
import io
import time
import base64
import wave
import re
import json
import random
import threading
from math import gcd
from datetime import datetime, timezone

import torch
import numpy as np

from tools import TOOL_SCHEMAS, execute_tool

# ============================================================
# Model context — injected by server via init()
# ============================================================
_ctx = {}
_sessions = {}  # id(ws) -> session_state


def _new_session_state():
    return {
        "customer_name": None, "customer_phone": None,
        "preferred_date": None, "preferred_time": None, "purpose": None,
        "confirmed": {
            "customer_name": False, "customer_phone": False,
            "preferred_date": False, "preferred_time": False,
        },
        "last_updated_turn": 0,
    }


def _get_session(ws):
    ws_id = id(ws)
    if ws_id not in _sessions:
        _sessions[ws_id] = _new_session_state()
    return _sessions[ws_id]


def _cleanup_session(ws):
    _sessions.pop(id(ws), None)


def init(ctx):
    """Inject model references from the main server. Called on startup + each reload."""
    global _ctx
    _ctx = ctx


# ============================================================
# Behavioral config (re-read from env on each reload)
# ============================================================
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT",
    "Bạn là nhân viên telesale đặt lịch hẹn qua điện thoại cho công ty. "
    "CHỈ nói tiếng Việt. TUYỆT ĐỐI KHÔNG dùng tiếng Trung Quốc, tiếng Anh, hay bất kỳ ngôn ngữ nào khác. "
    "TUYỆT ĐỐI KHÔNG dùng ký tự Trung Quốc (Hán tự). Chỉ dùng chữ Việt có dấu. "
    "KHÔNG BAO GIỜ nói: 'hẹn gặp lại', 'subscribe', 'video', 'kênh', 'like share', 'đăng ký'. "
    "Bạn KHÔNG PHẢI YouTuber, streamer, hay MC. "
    # --- Tool calling ---
    "Khi cần dữ liệu từ hệ thống (thông tin công ty, lịch trống, đặt lịch), "
    "PHẢI gọi tool. KHÔNG ĐƯỢC tự bịa thông tin. "
    "Khi khách hỏi địa chỉ, SĐT, hotline, giờ làm việc, dịch vụ → gọi get_company_info. "
    "Khi khách hỏi lịch trống → gọi check_schedule. "
    "Khi khách muốn đặt lịch → gọi book_appointment. "
    "Nếu không cần tool, trả lời bình thường. "
    # --- Booking flow ---
    "QUY TRÌNH ĐẶT LỊCH — PHẢI TUÂN THỦ: "
    "Bước 1: Khách nói muốn đặt lịch/hẹn → LẬP TỨC gọi check_schedule (date='ngày mai' hoặc ngày khách yêu cầu). "
    "Bước 2: Dựa vào kết quả tool, nói cho khách 2-3 khung giờ còn trống cụ thể. VD: 'Ngày mai còn slot 9 giờ, 10 giờ 30, và 2 giờ chiều. Anh chị chọn giờ nào ạ?' "
    "Bước 3: Kiểm tra [THÔNG TIN ĐÃ THU THẬP]. Nếu đã có tên, SĐT, ngày, giờ → KHÔNG hỏi lại. Chỉ hỏi thông tin còn thiếu. "
    "Bước 4: Có đủ ngày + giờ + tên → gọi book_appointment ngay. "
    "Nếu khách sửa thông tin, ưu tiên thông tin mới nhất. "
    "Nếu đã có trong [THÔNG TIN ĐÃ THU THẬP] thì không cần hỏi lại, trừ khi cần xác nhận. "
    "QUAN TRỌNG: KHÔNG BAO GIỜ nói 'để em kiểm tra' mà không gọi tool. Phải gọi check_schedule NGAY. "
    # --- Prosody rules ---
    "Viết mỗi câu trên 1 dòng riêng. Dấu chấm cuối câu khẳng định, dấu hỏi cuối câu hỏi. "
    "THƯỜNG XUYÊN nhưng KHÔNG PHẢI mọi câu, bắt đầu bằng từ lịch sự: dạ, vâng, dạ a, để em xem nhé. "
    "Ví dụ tốt: 'Dạ, shop có giao Hà Nội ạ.' rồi câu tiếp 'Khoảng 2-3 ngày.' "
    "Ví dụ sai: 'Dạ, shop có. Dạ, khoảng 2-3 ngày. Dạ, 30 nghìn.' "
    "Câu ngắn, tối đa 20 từ mỗi câu. Nếu dài hơn, tách thành 2 câu. "
    "KHÔNG dùng emoji, KHÔNG dùng ký tự đặc biệt. "
    "Trả lời ngắn gọn 1-2 câu, tự nhiên như người thật. "
    "QUAN TRỌNG: Câu đầu tiên phải cực ngắn (dưới 10 từ) để phản hồi nhanh. VD: 'Dạ, để em kiểm tra nhé.' rồi mới nói chi tiết câu sau. "
    "Nếu khách hỏi gì ngoài phạm vi, nói: 'Dạ, em chưa có thông tin về vấn đề này.'"
)

# TTS behavioral params (F5-TTS)
F5_TTS_SPEED = float(os.environ.get("F5_TTS_SPEED", "1.0"))
TTS_MAX_CHUNK_CHARS = int(os.environ.get("TTS_MAX_CHUNK_CHARS", "150"))

# Metrics
METRICS_FILE = os.environ.get("METRICS_FILE", "/workspace/voicebot-pipeline/call_metrics.jsonl")

# ============================================================
# Text normalization for F5-TTS
# Chuyển output LLM → text đọc được tự nhiên cho TTS
# ============================================================

# --- Số → chữ (đúng quy tắc tiếng Việt) ---

_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

def _read_two_digits(tens, ones, is_after_hundreds=False):
    """Đọc 2 chữ số (00-99) theo quy tắc tiếng Việt."""
    if tens == 0 and ones == 0:
        return ""
    if tens == 0:
        # 01-09 sau hàng trăm: "không một", "không năm"
        if is_after_hundreds:
            w = "lăm" if ones == 5 else _DIGITS[ones]
            return "không " + w
        return _DIGITS[ones]
    if tens == 1:
        # 10-19: mười, mười một, mười hai... mười lăm
        if ones == 0:
            return "mười"
        if ones == 1:
            return "mười một"
        if ones == 4:
            return "mười bốn"
        if ones == 5:
            return "mười lăm"
        return "mười " + _DIGITS[ones]
    # 20-99
    prefix = _DIGITS[tens] + " mươi"
    if ones == 0:
        return prefix
    if ones == 1:
        return prefix + " mốt"
    if ones == 4:
        return prefix + " tư"
    if ones == 5:
        return prefix + " lăm"
    return prefix + " " + _DIGITS[ones]


def _number_to_words(n):
    """Chuyển số nguyên (0 - 999,999,999,999) → chữ tiếng Việt."""
    if n < 0:
        return "âm " + _number_to_words(-n)
    if n == 0:
        return "không"

    parts = []

    # Tỷ
    if n >= 1_000_000_000:
        ty = n // 1_000_000_000
        parts.append(_number_to_words(ty) + " tỷ")
        n %= 1_000_000_000

    # Triệu
    if n >= 1_000_000:
        trieu = n // 1_000_000
        parts.append(_number_to_words(trieu) + " triệu")
        n %= 1_000_000

    # Nghìn
    if n >= 1_000:
        nghin = n // 1_000
        parts.append(_number_to_words(nghin) + " nghìn")
        n %= 1_000

    # Trăm + chục + đơn vị
    if n >= 100:
        tram = n // 100
        parts.append(_DIGITS[tram] + " trăm")
        n %= 100
        if n > 0:
            parts.append(_read_two_digits(n // 10, n % 10, is_after_hundreds=True))
    elif n > 0:
        # Còn lại < 100 nhưng có hàng nghìn/triệu phía trước → cần "không trăm"
        if parts:
            parts.append("không trăm")
            parts.append(_read_two_digits(n // 10, n % 10, is_after_hundreds=True))
        else:
            parts.append(_read_two_digits(n // 10, n % 10, is_after_hundreds=False))

    return " ".join(p for p in parts if p)


def _read_phone_digits(s):
    """Đọc từng chữ số, tách nhóm bằng dấu phẩy."""
    groups = s.split()
    result = []
    for group in groups:
        digits = [_DIGITS[int(c)] for c in group if c.isdigit()]
        result.append(" ".join(digits))
    return ", ".join(result)


# Pattern nhận diện số điện thoại / tổng đài
# 1900xxxx, 1800xxxx, 0xxx xxx xxx, hoặc nhóm số tách bằng dấu cách
_PHONE_PATTERN = re.compile(
    r'\b(1[89]00[\s.-]?\d{1,4}[\s.-]?\d{0,4})\b'   # 1900/1800 + nhóm số
    r'|'
    r'\b(0\d{2,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4})\b'   # 0xxx xxx xxxx
    r'|'
    r'\b(0\d{8,9})\b'                                 # 0xxxxxxxxx liền nhau (9-10 số)
    r'|'
    r'\b(\d{7,})\b'                                   # Dãy 7+ số liền — đọc từng chữ số
)

# Số + đơn vị (không phải điện thoại)
_NUM_WITH_UNIT = re.compile(
    r'(\d[\d.,]*\d|\d+)\s*(%|phần trăm|nghìn|ngàn|triệu|tỷ|đồng|vnđ|vnd|kg|km|m2|m²|m|cm|mm|lít|giờ|phút|giây|tuổi|người|căn|phòng|tầng|lầu|năm|tháng|ngày|suất|lô|nền|bộ|cái|chiếc|chai|hộp|gói)',
    re.IGNORECASE
)

# Số đứng một mình
_STANDALONE_NUM = re.compile(r'\b(\d[\d.,]*\d|\d+)\b')

# Phần trăm
_PERCENT = re.compile(r'(\d[\d.,]*\d|\d+)\s*%')

# Khoảng số: 2-3, 50-60
_RANGE_NUM = re.compile(r'(\d+)\s*[-–]\s*(\d+)')


def _parse_number(s):
    """Parse string số (có thể có dấu . hoặc , phân cách) → int."""
    s = s.replace(".", "").replace(",", "")
    try:
        return int(s)
    except ValueError:
        return None


# Giờ:phút — 7:30 → "bảy giờ ba mươi", 14:05 → "mười bốn giờ không năm"
_TIME_PATTERN = re.compile(r'\b(\d{1,2})\s*[:hH]\s*(\d{2})\b')
# Giờ đứng 1 mình kèm "giờ/sáng/chiều/tối" — 7 giờ, 3 giờ chiều
_TIME_HOUR_ONLY = re.compile(r'\b(\d{1,2})\s*giờ\b', re.IGNORECASE)


def _normalize_numbers(text):
    """Chuyển tất cả số trong text → chữ tiếng Việt."""

    # Bước 0: Giờ:phút → "X giờ Y phút"
    def _replace_time(m):
        h = int(m.group(1))
        mi = int(m.group(2))
        if h > 24 or mi > 59:
            return m.group(0)  # không phải giờ hợp lệ
        result = _number_to_words(h) + " giờ"
        if mi > 0:
            if mi < 10:
                result += " không " + _DIGITS[mi]
            else:
                result += " " + _number_to_words(mi)
        return result
    text = _TIME_PATTERN.sub(_replace_time, text)

    def _replace_hour_only(m):
        h = int(m.group(1))
        if h > 24:
            return m.group(0)
        return _number_to_words(h) + " giờ"
    text = _TIME_HOUR_ONLY.sub(_replace_hour_only, text)

    # Bước 1: Số điện thoại / tổng đài → đọc từng số
    def _replace_phone(m):
        matched = m.group(0)
        return _read_phone_digits(matched)
    text = _PHONE_PATTERN.sub(_replace_phone, text)

    # Bước 2: Phần trăm
    def _replace_percent(m):
        n = _parse_number(m.group(1))
        if n is None:
            return m.group(0)
        return _number_to_words(n) + " phần trăm"
    text = _PERCENT.sub(_replace_percent, text)

    # Bước 3: Khoảng số (2-3 ngày → hai đến ba ngày)
    def _replace_range(m):
        n1 = _parse_number(m.group(1))
        n2 = _parse_number(m.group(2))
        if n1 is None or n2 is None:
            return m.group(0)
        return _number_to_words(n1) + " đến " + _number_to_words(n2)
    text = _RANGE_NUM.sub(_replace_range, text)

    # Bước 4: Số + đơn vị
    def _replace_num_unit(m):
        n = _parse_number(m.group(1))
        unit = m.group(2)
        if n is None:
            return m.group(0)
        return _number_to_words(n) + " " + unit
    text = _NUM_WITH_UNIT.sub(_replace_num_unit, text)

    # Bước 5: Số đứng một mình (còn sót)
    def _replace_standalone(m):
        n = _parse_number(m.group(1))
        if n is None:
            return m.group(0)
        return _number_to_words(n)
    text = _STANDALONE_NUM.sub(_replace_standalone, text)

    return text


# --- Chữ hoa có dấu → viết thường (F5-TTS đọc sai chữ hoa có dấu) ---

def _lowercase_vietnamese(text):
    """Viết thường toàn bộ — F5-TTS xử lý tốt hơn với lowercase."""
    return text.lower()


# --- Từ tiếng Anh phổ biến → phiên âm tiếng Việt ---

_ENGLISH_PHONETIC = {
    "tennis": "ten nít",
    "mega mall": "mê ga mol",
    "mall": "mol",
    "email": "i meo",
    "e-mail": "i meo",
    "website": "uép sai",
    "web": "uép",
    "online": "on lai",
    "offline": "óp lai",
    "wifi": "wai fai",
    "ok": "ô kê",
    "feedback": "phít bách",
    "sale": "seo",
    "telesale": "te lờ seo",
    "marketing": "ma két tinh",
    "showroom": "sô rum",
    "villa": "vi la",
    "penthouse": "pent hao",
    "duplex": "đu plếch",
    "studio": "xtu đi ô",
    "gym": "dim",
    "spa": "xpa",
    "golf": "gôn",
    "resort": "ri dọt",
    "check-in": "chếch in",
    "checkin": "chếch in",
    "smart home": "xmát hôm",
    "smart": "xmát",
    "view": "viu",
    "vip": "vi ai pi",
    "app": "ép",
    "zalo": "da lô",
}

_ENGLISH_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(_ENGLISH_PHONETIC.keys(), key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


def _phonetic_english(text):
    """Phiên âm từ tiếng Anh phổ biến → tiếng Việt."""
    def _replace(m):
        return _ENGLISH_PHONETIC.get(m.group(0).lower(), m.group(0))
    return _ENGLISH_PATTERN.sub(_replace, text)


# --- Vietnamese abbreviation expander ---

_VN_ABBREVIATIONS = {
    # Loại hình doanh nghiệp
    "TNHH MTV": "trách nhiệm hữu hạn một thành viên",
    "TNHH": "trách nhiệm hữu hạn",
    "DNTN": "doanh nghiệp tư nhân",
    "DNNVV": "doanh nghiệp nhỏ và vừa",
    "HTX": "hợp tác xã",
    "BHXH": "bảo hiểm xã hội",
    "BHYT": "bảo hiểm y tế",
    "BHTN": "bảo hiểm thất nghiệp",

    # Tổ chức / cơ quan
    "UBND": "ủy ban nhân dân",
    "HĐND": "hội đồng nhân dân",
    "HĐQT": "hội đồng quản trị",
    "BGĐ": "ban giám đốc",
    "GĐ": "giám đốc",
    "PGĐ": "phó giám đốc",
    "TGĐ": "tổng giám đốc",
    "ĐHQG": "đại học quốc gia",

    # Tài chính / ngân hàng
    "NHNN": "ngân hàng nhà nước",
    "NHTM": "ngân hàng thương mại",
    "TCTD": "tổ chức tín dụng",
    "TTCK": "thị trường chứng khoán",
    "GDP": "gi đi pi",
    "VAT": "vi ây ti",
    "GTGT": "giá trị gia tăng",
    "TNCN": "thu nhập cá nhân",
    "TNDN": "thu nhập doanh nghiệp",

    # Giấy tờ / pháp lý
    "CMND": "chứng minh nhân dân",
    "CCCD": "căn cước công dân",
    "GCNĐKKD": "giấy chứng nhận đăng ký kinh doanh",
    "ĐKKD": "đăng ký kinh doanh",
    "GPKD": "giấy phép kinh doanh",
    "HĐLĐ": "hợp đồng lao động",
    "HĐMB": "hợp đồng mua bán",
    "MST": "mã số thuế",
    "SĐT": "số điện thoại",
    "ĐT": "điện thoại",

    # Địa danh / hành chính
    "TP": "thành phố",
    "HCM": "hồ chí minh",
    "TPHCM": "thành phố hồ chí minh",
    "TP.HCM": "thành phố hồ chí minh",
    "HN": "hà nội",
    "ĐN": "đà nẵng",
    "ĐBSCL": "đồng bằng sông cửu long",

    # Thương hiệu Việt Nam (đọc theo chữ cái)
    "FPT": "ép pi ti",
    "VNPT": "vi en pi ti",
    "EVN": "i vi en",
    "VTV": "vi ti vi",
    "VTC": "vi ti xi",
    "VNA": "vi en ây",
    "VNĐ": "việt nam đồng",
    "VND": "việt nam đồng",
    "VN": "việt nam",
    "BIDV": "bi ai đi vi",
    "ACB": "ây xi bi",
    "MB": "em bi",
    "TCB": "ti xi bi",
    "VPB": "vi pi bi",
    "VCB": "vi xi bi",
    "STB": "ét ti bi",
    "SHB": "ét hát bi",
    "TPB": "ti pi bi",
    "LPB": "eo pi bi",
    "OCB": "ô xi bi",
    "MSB": "em ét bi",
    "KLB": "ca eo bi",
    "NAB": "en ây bi",
    "BAB": "bi ây bi",
    "ABB": "ây bi bi",
    "PGB": "pi gi bi",
    "SSB": "ét ét bi",
    "VISA": "vi da",

    # Giáo dục
    "ĐH": "đại học",
    "CĐ": "cao đẳng",
    "THPT": "trung học phổ thông",
    "THCS": "trung học cơ sở",
    "TH": "tiểu học",
    "MN": "mầm non",
    "GV": "giáo viên",
    "SV": "sinh viên",
    "HS": "học sinh",

    # Y tế
    "BV": "bệnh viện",
    "PK": "phòng khám",
    "BS": "bác sĩ",
    "ĐD": "điều dưỡng",
    "KCB": "khám chữa bệnh",

    # Giao thông / vận tải
    "GPLX": "giấy phép lái xe",
    "ĐKLX": "đăng ký lái xe",
    "CSGT": "cảnh sát giao thông",
    "PCCC": "phòng cháy chữa cháy",
    "CA": "công an",

    # Bất động sản
    "BĐS": "bất động sản",
    "GCNQSDĐ": "giấy chứng nhận quyền sử dụng đất",
    "QSDĐ": "quyền sử dụng đất",
    "SĐ": "sổ đỏ",

    # Công nghệ
    "CNTT": "công nghệ thông tin",
    "TMĐT": "thương mại điện tử",
    "AI": "ây ai",
    "IT": "ai ti",
    "CEO": "xi i ô",
    "CTO": "xi ti ô",
    "CFO": "xi ép ô",
    "HR": "ết cha",
    "PR": "pi a",
    "KPI": "kây pi ai",
    "OT": "ô ti",

    # Đơn vị / khác
    "kg": "ki lô gam",
    "km": "ki lô mét",
    "cm": "xen ti mét",
    "mm": "mi li mét",
    "m2": "mét vuông",
    "m3": "mét khối",
    "km2": "ki lô mét vuông",
}

# Sort by length descending so longer abbreviations match first (TNHH MTV before TNHH)
_VN_ABBR_PATTERN = re.compile(
    r'(?<![a-zA-ZÀ-ỹ])(' + '|'.join(re.escape(k) for k in sorted(_VN_ABBREVIATIONS.keys(), key=len, reverse=True)) + r')(?![a-zA-ZÀ-ỹ])',
)


def _expand_abbreviations(text):
    """Mở rộng viết tắt tiếng Việt → dạng đầy đủ cho TTS."""
    def _replace(m):
        key = m.group(0)
        # Try exact match first, then uppercase
        return _VN_ABBREVIATIONS.get(key, _VN_ABBREVIATIONS.get(key.upper(), key))
    return _VN_ABBR_PATTERN.sub(_replace, text)


# --- Main normalizer ---

def normalize_for_tts(text):
    """Normalize LLM output → text tối ưu cho F5-TTS Vietnamese.
    Gọi trước khi đưa text vào TTS synthesis.
    Pipeline: viết tắt → English phonetic → numbers → lowercase → cleanup
    """
    if not text or not text.strip():
        return text
    text = _expand_abbreviations(text)
    text = _phonetic_english(text)
    text = _normalize_numbers(text)
    text = _lowercase_vietnamese(text)
    # Loại bỏ ký tự Trung/Nhật/Hàn (Qwen hay bị leak)
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]+', '', text)
    # Dọn khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# Hallucination filter
# ============================================================
HALLUCINATION_PATTERNS = [
    "subscribe", "kenh", "la la school", "lala school",
    "hen gap lai cac ban", "video tiep theo",
    "dang ky kenh", "like share", "bam chuong",
    "xin chao cac ban da den voi", "cam on cac ban da xem",
    "nho dang ky", "theo doi kenh", "ung ho kenh",
    "hay subscribe", "bo lo nhung video",
]

# ============================================================
# "Im lặng" (silence/mute) mode
# ============================================================
_muted = False

MUTE_PATTERNS = [
    "im lặng", "im lang", "im đi", "im di",
    "câm", "cam",
    "đừng nói nữa", "dung noi nua",
    "không nói nữa", "khong noi nua",
    "ngừng nói", "ngung noi",
    "yên lặng", "yen lang",
]

UNMUTE_PATTERNS = [
    "nói đi", "noi di",
    "nói lại đi", "noi lai di",
    "bỏ im lặng", "bo im lang",
    "hết im lặng", "het im lang",
    "nói tiếp", "noi tiep",
    "nói lại", "noi lai",
]

# ============================================================
# Backchannel config
# ============================================================
BACKCHANNEL_PROBABILITY = float(os.environ.get("BACKCHANNEL_PROB", "0"))
BACKCHANNEL_TEXTS = ["Dạ", "Vâng", "Một chút nhé", "Để em xem"]


def pre_synthesize_backchannels():
    """Pre-synthesize backchannel clips at warmup. Called from all_in_one_server.py."""
    clips = []
    for text in BACKCHANNEL_TEXTS:
        wav = _tts_synthesize_single(text)
        if wav is not None:
            clips.append(waveform_to_wav_bytes(wav))
    _ctx["backchannel_clips"] = clips
    print(f"[BC] Pre-synthesized {len(clips)} backchannel clips")
    return clips


# ============================================================
# Sentence / chunk splitter
# ============================================================
SENTENCE_END = re.compile(r'[.?!。!?]\s*$')
# For streaming WS: flush TTS early at clause boundaries (comma etc.) if buffer is long enough
CLAUSE_BREAK = re.compile(r'[,;:，；]\s*$')
EARLY_FLUSH_MIN_CHARS = 15  # flush at comma only if buffer >= this many chars


MAX_WORDS_PER_CHUNK = 20
MIN_WORDS_PER_CHUNK = 3


def _split_tts_chunks(text, max_chars=None):
    """Split by sentence boundaries first, then by comma for long sentences.
    Merge chunks that are too short (<3 words) with the previous chunk."""
    text = text.strip()
    if not text:
        return []

    # Step 1: Split by sentence end (.?!)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Step 2: Split long sentences (>20 words) at comma
    expanded = []
    for sent in sentences:
        words = sent.split()
        if len(words) > MAX_WORDS_PER_CHUNK:
            # Split at comma boundaries
            clauses = re.split(r'([,;]\s*)', sent)
            current = ""
            for part in clauses:
                if len((current + part).split()) > MAX_WORDS_PER_CHUNK and current.strip():
                    expanded.append(current.strip())
                    current = part
                else:
                    current += part
            if current.strip():
                expanded.append(current.strip())
        else:
            expanded.append(sent)

    # Step 3: Merge chunks that are too short (<3 words) with previous
    if len(expanded) <= 1:
        return expanded

    merged = [expanded[0]]
    for chunk in expanded[1:]:
        if len(chunk.split()) < MIN_WORDS_PER_CHUNK and merged:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged if merged else [text]


# ============================================================
# Helper functions
# ============================================================

def _check_mute_command(text):
    """Returns 'mute', 'unmute', or None."""
    lower = text.lower().strip()
    for p in UNMUTE_PATTERNS:
        if p in lower:
            return "unmute"
    for p in MUTE_PATTERNS:
        if p in lower:
            return "mute"
    return None


def _is_hallucination(text):
    lower = text.lower().strip()
    if not lower or len(lower) < 3:
        return True
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in lower:
            return True
    words = lower.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


def _resample(audio_np, sr_from, sr_to):
    if sr_from == sr_to:
        return audio_np
    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(audio_np, sr_to // g, sr_from // g)


def noise_suppress(audio_np, sr=16000):
    df_model = _ctx.get("df_model")
    df_state = _ctx.get("df_state")
    df_enhance_fn = _ctx.get("df_enhance_fn")
    if df_model is None or df_state is None or df_enhance_fn is None:
        return audio_np
    try:
        target_sr = df_state.sr()
        audio_up = _resample(audio_np, sr, target_sr)
        audio_tensor = torch.from_numpy(audio_up).float().unsqueeze(0)
        enhanced = df_enhance_fn(df_model, df_state, audio_tensor)
        if isinstance(enhanced, torch.Tensor):
            enhanced_np = enhanced.cpu().detach().squeeze().numpy()
        else:
            enhanced_np = np.asarray(enhanced).squeeze()
        enhanced_orig = _resample(enhanced_np, target_sr, sr)
        if len(enhanced_orig) > len(audio_np):
            enhanced_orig = enhanced_orig[:len(audio_np)]
        elif len(enhanced_orig) < len(audio_np):
            enhanced_orig = np.pad(enhanced_orig, (0, len(audio_np) - len(enhanced_orig)))
        return enhanced_orig
    except Exception as e:
        print(f"[NS] DeepFilterNet error (fallback to raw audio): {e}")
        return audio_np


def estimate_snr(audio_np, sr=16000):
    frame_size = int(sr * 0.1)
    n_frames = len(audio_np) // frame_size
    if n_frames < 2:
        return 0.0
    frames = audio_np[:n_frames * frame_size].reshape(n_frames, frame_size)
    energies = np.mean(frames ** 2, axis=1)
    energies_sorted = np.sort(energies)
    n_noise = max(1, n_frames // 5)
    n_signal = max(1, n_frames // 2)
    noise_e = float(np.mean(energies_sorted[:n_noise]))
    signal_e = float(np.mean(energies_sorted[-n_signal:]))
    if noise_e < 1e-10:
        return 60.0
    return round(10 * np.log10(signal_e / noise_e), 1)


def log_metrics(metrics_dict):
    try:
        metrics_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(metrics_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[METRICS] Log failed: {e}")


# ============================================================
# STT
# ============================================================

def stt_transcribe(audio_bytes):
    """Transcribe audio bytes. Returns (text, metrics_dict)."""
    stt_model = _ctx["stt_model"]
    enable_deepfilter = _ctx.get("enable_deepfilter", False)

    buf = io.BytesIO(audio_bytes)
    import soundfile as sf
    audio_np, sr = sf.read(buf)
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    snr = estimate_snr(audio_np, int(sr))
    duration = round(len(audio_np) / sr, 2)

    metrics = {
        "input_rms": round(rms, 5),
        "input_snr_est": snr,
        "input_duration_s": duration,
    }

    if rms < 0.01:
        metrics["rejected"] = "low_rms"
        return "", metrics

    if enable_deepfilter and _ctx.get("df_model") is not None:
        ns_t0 = time.time()
        audio_np = noise_suppress(audio_np, int(sr))
        metrics["ns_latency_ms"] = round((time.time() - ns_t0) * 1000, 1)

    segments, info = stt_model.transcribe(
        audio_np, language="vi",
        beam_size=1,                        # faster decoding (default=5)
        condition_on_previous_text=False,    # faster, no context dependency
        vad_filter=True,
        no_speech_threshold=0.7,
        vad_parameters=dict(
            min_speech_duration_ms=200,
            min_silence_duration_ms=150,
            speech_pad_ms=80,
            threshold=0.45,
        ),
    )
    import unicodedata
    seg_list = list(segments)
    text = unicodedata.normalize("NFC", " ".join([s.text for s in seg_list]).strip())

    if seg_list:
        try:
            avg_lp = [getattr(s, 'avg_log_prob', None) or getattr(s, 'avg_logprob', 0) for s in seg_list]
            metrics["stt_confidence"] = round(float(np.mean(avg_lp)), 3)
        except Exception:
            pass

    metrics["stt_text_len"] = len(text)
    hallucinated = _is_hallucination(text)
    metrics["hallucination"] = hallucinated
    return ("" if hallucinated else text), metrics


# ============================================================
# LLM
# ============================================================

ENABLE_TOOLS = os.environ.get("ENABLE_TOOLS", "1") == "1"
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "3"))


def _llm_generate_once(messages, stream=False):
    """Single LLM generation round. Returns full text or yields tokens if stream=True."""
    from transformers import TextIteratorStreamer

    llm_model = _ctx["llm_model"]
    llm_tokenizer = _ctx["llm_tokenizer"]
    device = _ctx["device"]

    # Apply chat template with tools
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if ENABLE_TOOLS:
        template_kwargs["tools"] = TOOL_SCHEMAS

    text = llm_tokenizer.apply_chat_template(messages, **template_kwargs)
    inputs = llm_tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        **inputs,
        "max_new_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": 0.8,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
    }

    if stream:
        # Streaming mode — use TextIteratorStreamer in a thread
        streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        gen_error = [None]

        def _generate():
            try:
                llm_model.generate(**gen_kwargs)
            except Exception as e:
                gen_error[0] = e

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()
        for token_text in streamer:
            if token_text:
                yield token_text
        thread.join(timeout=60)
        if gen_error[0] is not None:
            raise RuntimeError(f"LLM generation failed: {gen_error[0]}")
        if thread.is_alive():
            print("[LLM] WARNING: generation thread timed out (60s)")
            raise RuntimeError("LLM generation timed out")
    else:
        # Non-streaming mode — generate and decode at once
        print(f"[LLM-GEN] Non-streaming, input_len={input_len}, max_new_tokens={LLM_MAX_TOKENS}")
        with torch.inference_mode():
            output_ids = llm_model.generate(**gen_kwargs)
        new_ids = output_ids[0][input_len:]
        result = llm_tokenizer.decode(new_ids, skip_special_tokens=False)
        print(f"[LLM-GEN] Raw decode: '{result[:300]}'")
        # Strip special tokens but keep <tool_call> tags
        # Qwen adds <|im_end|> at the end
        result = result.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        print(f"[LLM-GEN] Cleaned: '{result[:300]}'")
        yield result


def _parse_tool_calls(text):
    """Parse tool calls from LLM output.
    Tries multiple patterns:
    1. <tool_call>{"name": ..., "arguments": ...}</tool_call>
    2. Bare JSON with "name" and "arguments" keys
    Returns list of (name, arguments) tuples, or empty list if no tool calls.
    """
    calls = []

    # Pattern 1: <tool_call>...</tool_call> (standard Qwen format)
    pattern1 = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)
    for match in pattern1.finditer(text):
        try:
            obj = json.loads(match.group(1))
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if name:
                calls.append((name, arguments))
        except json.JSONDecodeError:
            continue

    if calls:
        return calls

    # Pattern 2: Fallback — find JSON with "name" key (model sometimes drops tags)
    pattern2 = re.compile(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', re.DOTALL)
    for match in pattern2.finditer(text):
        try:
            name = match.group(1)
            arguments = json.loads(match.group(2))
            from tools import TOOL_EXECUTORS
            if name in TOOL_EXECUTORS:
                calls.append((name, arguments))
        except (json.JSONDecodeError, ImportError):
            continue

    return calls


def _agent_step(messages, round_i=0, session_state=None):
    """Recursive agent step: generate → check tool call → execute → repeat.
    Returns final text response (no tool calls).
    """
    if round_i >= MAX_TOOL_ROUNDS:
        print(f"[AGENT] Max rounds ({MAX_TOOL_ROUNDS}) reached")
        return "Dạ, em xin lỗi, hệ thống đang bận. Anh chị thử lại sau ạ."

    # Generate non-streaming
    full_text = ""
    for chunk in _llm_generate_once(messages, stream=False):
        full_text += chunk

    print(f"[AGENT] Round {round_i}: '{full_text[:200]}'")

    # Parse tool call
    tool_calls = _parse_tool_calls(full_text)

    if not tool_calls:
        # No tool call → this is the final answer
        clean = re.sub(r'</?tool_call>', '', full_text).strip()
        print(f"[AGENT] Round {round_i}: final reply ({len(clean)} chars)")
        return clean

    # Execute tool(s) and recurse
    messages.append({"role": "assistant", "content": full_text})
    for name, arguments in tool_calls:
        # Enrich missing args from session state
        if session_state:
            arguments = _enrich_tool_args(name, arguments, session_state)
        print(f"[AGENT] Round {round_i}: tool '{name}' args={arguments}")
        result = execute_tool(name, arguments)
        messages.append({
            "role": "tool",
            "name": name,
            "content": json.dumps(result, ensure_ascii=False),
        })
        # Extract info from tool args into session state
        if session_state:
            _extract_session_info(None, None, session_state, tool_args=arguments)

    # Recurse — let LLM use tool result to write natural response
    return _agent_step(messages, round_i + 1, session_state=session_state)


# Intent patterns for auto-tool injection
_SCHEDULE_INTENTS = re.compile(
    r'(đặt lịch|đặt hẹn|hẹn gặp|book|lịch trống|lịch hẹn|slot|còn lịch|có lịch|xếp lịch|'
    r'muốn hẹn|muốn gặp|thăm khám|tư vấn trực tiếp|đến công ty|ghé công ty)',
    re.IGNORECASE
)
_COMPANY_INTENTS = re.compile(
    r'(ở đâu|địa chỉ|hotline|số điện thoại|liên hệ|giờ làm việc|giờ mở cửa|'
    r'dịch vụ gì|chi nhánh|email|website|công ty)',
    re.IGNORECASE
)


def _auto_inject_tools(user_text, messages):
    """Detect intent và tự động gọi tool, inject kết quả vào messages.
    Giúp model không phải tự quyết định gọi tool — data có sẵn để trả lời luôn.
    """
    injected = False

    if _SCHEDULE_INTENTS.search(user_text):
        # Extract date from user text
        date = "ngày mai"  # default
        date_patterns = {
            r'hôm nay': 'hôm nay', r'ngày mai': 'ngày mai', r'ngày kia': 'ngày kia',
            r'thứ hai|thứ 2': 'thứ 2', r'thứ ba|thứ 3': 'thứ 3',
            r'thứ tư|thứ 4': 'thứ 4', r'thứ năm|thứ 5': 'thứ 5',
            r'thứ sáu|thứ 6': 'thứ 6', r'thứ bảy|thứ 7': 'thứ 7',
        }
        for pattern, val in date_patterns.items():
            if re.search(pattern, user_text, re.IGNORECASE):
                date = val
                break

        result = execute_tool("check_schedule", {"date": date})
        # Add as system context (not as tool call/response — just inject info)
        available = result.get("available_slots", [])
        date_str = result.get("date", date)
        day_name = result.get("day_of_week", "")
        if available:
            slot_text = ", ".join(available[:6])
            if len(available) > 6:
                slot_text += f" (và {len(available)-6} slot khác)"
            context = f"[HỆ THỐNG] Lịch ngày {date_str} ({day_name}): còn {len(available)} slot trống: {slot_text}. Hãy gợi ý 2-3 slot cho khách."
        else:
            context = f"[HỆ THỐNG] Ngày {date_str} ({day_name}) đã hết lịch. Hãy đề xuất ngày khác."
        messages.append({"role": "system", "content": context})
        print(f"[AGENT-AUTO] Injected schedule: {context[:100]}")
        injected = True

    if _COMPANY_INTENTS.search(user_text) and not _SCHEDULE_INTENTS.search(user_text):
        result = execute_tool("get_company_info", {"field": "all"})
        context = f"[HỆ THỐNG] Thông tin công ty: {json.dumps(result, ensure_ascii=False)}"
        messages.append({"role": "system", "content": context})
        print(f"[AGENT-AUTO] Injected company info")
        injected = True

    return injected


# ============================================================
# Session memory — extraction, formatting, enrichment
# ============================================================

_RE_PHONE = re.compile(r'0\d{9}')
_RE_TIME = re.compile(r'(\d{1,2})\s*(?:h|giờ|:)\s*(\d{0,2})', re.IGNORECASE)
_VN_UPPER = r'[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ]'
_VN_LOWER = r'[a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]'
_RE_NAME_PATTERNS = re.compile(
    r'(?:[Tt]ên\s+(?:là|em|anh|chị|tôi)\s+|[Tt]ôi\s+là\s+|[Aa]nh\s+|[Cc]hị\s+|[Ee]m\s+là\s+)'
    rf'({_VN_UPPER}{_VN_LOWER}+(?:\s+{_VN_UPPER}{_VN_LOWER}+)*)',
    re.UNICODE
)
_RE_DATE_EXTRACT = {
    r'hôm nay': 'hôm nay', r'ngày mai': 'ngày mai', r'ngày kia': 'ngày kia',
    r'thứ hai|thứ 2': 'thứ 2', r'thứ ba|thứ 3': 'thứ 3',
    r'thứ tư|thứ 4': 'thứ 4', r'thứ năm|thứ 5': 'thứ 5',
    r'thứ sáu|thứ 6': 'thứ 6', r'thứ bảy|thứ 7': 'thứ 7',
}
_RE_DATE_NUMERIC = re.compile(r'ngày\s+(\d{1,2}(?:[/\-\.]\d{1,2}(?:[/\-\.]\d{2,4})?)?)', re.IGNORECASE)


def _extract_session_info(transcript, reply, state, tool_args=None):
    """Extract booking info from transcript/reply/tool_args into session state.
    Priority: tool_args > transcript regex > reply (bot confirmation sync).
    """
    confirmed = state["confirmed"]

    # 1. tool_args — strongest signal
    if tool_args:
        for key in ("customer_name", "customer_phone", "preferred_date", "preferred_time", "purpose"):
            if key in tool_args and tool_args[key]:
                state[key] = tool_args[key]
                if key in confirmed:
                    confirmed[key] = True

    # 2. transcript — regex extraction
    if transcript:
        # Phone — unambiguous
        phone_match = _RE_PHONE.search(transcript)
        if phone_match:
            state["customer_phone"] = phone_match.group()
            confirmed["customer_phone"] = True

        # Time
        time_match = _RE_TIME.search(transcript)
        if time_match:
            h = time_match.group(1)
            m = time_match.group(2) or "00"
            state["preferred_time"] = f"{int(h):02d}:{int(m):02d}" if m else f"{int(h):02d}:00"
            confirmed["preferred_time"] = True

        # Date — relative patterns
        for pattern, val in _RE_DATE_EXTRACT.items():
            if re.search(pattern, transcript, re.IGNORECASE):
                state["preferred_date"] = val
                confirmed["preferred_date"] = True
                break
        else:
            # Date — numeric "ngày 15", "ngày 15/3"
            date_num = _RE_DATE_NUMERIC.search(transcript)
            if date_num:
                state["preferred_date"] = f"ngày {date_num.group(1)}"
                confirmed["preferred_date"] = True

        # Name — cautious, only from clear patterns
        name_match = _RE_NAME_PATTERNS.search(transcript)
        if name_match:
            state["customer_name"] = name_match.group(1).strip()
            # Name from regex stays unconfirmed — wait for tool_args or bot confirmation
            # confirmed["customer_name"] stays False

    # 3. reply — bot confirmation sync (e.g. "Vâng, anh Minh")
    if reply and state["customer_name"] and not confirmed.get("customer_name"):
        name = state["customer_name"]
        if name.lower() in reply.lower():
            confirmed["customer_name"] = True


def _missing_slots(state):
    """Return list of missing required fields."""
    required = ["customer_name", "customer_phone", "preferred_date", "preferred_time"]
    return [f for f in required if not state.get(f)]


def _format_session_state(state):
    """Format session state for injection into LLM context."""
    lines = []
    labels = {
        "customer_name": "Tên khách",
        "customer_phone": "SĐT",
        "preferred_date": "Ngày hẹn",
        "preferred_time": "Giờ hẹn",
        "purpose": "Mục đích",
    }
    has_any = False
    for key, label in labels.items():
        if state.get(key):
            lines.append(f"- {label}: {state[key]}")
            has_any = True

    if not has_any:
        return ""

    missing = _missing_slots(state)
    result = "[THÔNG TIN ĐÃ THU THẬP]\n" + "\n".join(lines)
    if missing:
        missing_labels = {
            "customer_name": "tên khách", "customer_phone": "SĐT",
            "preferred_date": "ngày hẹn", "preferred_time": "giờ hẹn",
        }
        result += "\nThông tin còn thiếu: " + ", ".join(missing_labels.get(f, f) for f in missing)
    result += "\nNếu thông tin đã có thì không hỏi lại. Nếu khách sửa, ưu tiên thông tin mới nhất."
    return result


def _enrich_tool_args(name, args, state):
    """Auto-fill missing tool args from session state."""
    if name != "book_appointment":
        return args
    mapping = {
        "customer_name": "customer_name",
        "customer_phone": "customer_phone",
        "preferred_date": "preferred_date",
        "preferred_time": "preferred_time",
    }
    for tool_key, state_key in mapping.items():
        if (not args.get(tool_key)) and state.get(state_key):
            args[tool_key] = state[state_key]
            print(f"[SESSION] Enriched {tool_key}={state[state_key]} from session")
    return args


def llm_generate_stream(user_text, conversation=None, session_state=None):
    """Stream tokens from local Qwen model — with Agent tool-calling loop.

    If tools enabled: auto-inject tool data + run agent loop.
    If tools disabled: stream directly (legacy).
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject session state as system context
    if session_state:
        session_text = _format_session_state(session_state)
        if session_text:
            messages.append({"role": "system", "content": session_text})

    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": user_text})

    if not ENABLE_TOOLS:
        yield from _llm_generate_once(messages, stream=True)
        return

    # Auto-inject tool results based on intent detection
    auto_injected = _auto_inject_tools(user_text, messages)

    # Agent loop — may take 1-3 rounds if tools are called
    final_text = _agent_step(messages, session_state=session_state)
    yield final_text


def llm_generate_api(user_text, conversation=None):
    """Generate via OpenAI API."""
    llm_client = _ctx["llm_client"]
    llm_model_name = _ctx["llm_model_name"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": user_text})

    response = llm_client.chat.completions.create(
        model=llm_model_name,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# TTS
# ============================================================

@torch.inference_mode()
def _tts_synthesize_single(text, nfe_step=None):
    """Synthesize text using F5-TTS."""
    f5_model = _ctx.get("f5_tts_model")
    ref_audio = _ctx.get("f5_tts_ref_audio", "")
    ref_text = _ctx.get("f5_tts_ref_text", "")
    speed = _ctx.get("f5_tts_speed", F5_TTS_SPEED)
    if nfe_step is None:
        nfe_step = int(os.environ.get("F5_TTS_NFE_STEP", "16"))

    import unicodedata
    text = unicodedata.normalize("NFC", text.strip())
    print(f"[TTS-IN] '{text}' nfe_step={nfe_step}")
    if not text or f5_model is None:
        return None

    try:
        wav, sr, _ = f5_model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
            nfe_step=nfe_step,
        )
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.squeeze()
        # Normalize
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / peak * 0.95
        return wav
    except Exception as e:
        print(f"[TTS] F5-TTS synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None


# Pause durations (seconds) between TTS chunks for natural prosody
PAUSE_AFTER_SENTENCE = 0.25   # after . (250ms)
PAUSE_AFTER_QUESTION = 0.20   # after ? (200ms)
PAUSE_AFTER_COMMA    = 0.10   # after , or clause split (100ms)


def _get_pause_duration(chunk_text):
    """Return pause duration (seconds) based on how the chunk ends."""
    text = chunk_text.rstrip()
    if text.endswith('?'):
        return PAUSE_AFTER_QUESTION
    elif text.endswith('.') or text.endswith('!'):
        return PAUSE_AFTER_SENTENCE
    else:
        return PAUSE_AFTER_COMMA


def tts_synthesize(text, nfe_step=None):
    """Synthesize text → waveform using F5-TTS. Auto-normalizes text."""
    text = text.strip()
    if not text:
        return None
    text = normalize_for_tts(text)
    print(f"[TTS-NORM] '{text}'")
    return _tts_synthesize_single(text, nfe_step=nfe_step)


def waveform_to_wav_bytes(waveform):
    tts_sample_rate = _ctx.get("tts_sample_rate", 24000)
    audio_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts_sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ============================================================
# Streaming pipeline: LLM → TTS per sentence
# ============================================================

def chat_streaming_pipeline(transcript, conversation=None, session_state=None):
    """Returns (full_response, wav_bytes, latency_dict)."""
    llm_t0 = time.time()
    sentence_buffer = ""
    full_response = ""
    audio_waveforms = []
    tts_time_total = 0.0
    first_sentence_time = None

    for token in llm_generate_stream(transcript, conversation, session_state=session_state):
        sentence_buffer += token
        full_response += token

        if SENTENCE_END.search(sentence_buffer):
            sentence = sentence_buffer.strip()
            sentence_buffer = ""

            if first_sentence_time is None:
                first_sentence_time = time.time() - llm_t0

            tts_t0 = time.time()
            waveform = tts_synthesize(sentence)
            tts_time_total += time.time() - tts_t0
            if waveform is not None:
                audio_waveforms.append(waveform)

    llm_time = time.time() - llm_t0 - tts_time_total

    if sentence_buffer.strip():
        tts_t0 = time.time()
        waveform = tts_synthesize(sentence_buffer.strip())
        tts_time_total += time.time() - tts_t0
        if waveform is not None:
            audio_waveforms.append(waveform)

    if audio_waveforms:
        combined = np.concatenate(audio_waveforms)
        wav_bytes = waveform_to_wav_bytes(combined)
    else:
        wav_bytes = None

    return full_response, wav_bytes, {
        "llm": round(llm_time, 2),
        "tts": round(tts_time_total, 2),
        "first_sentence": round(first_sentence_time, 2) if first_sentence_time else 0,
    }


# ============================================================
# Route handlers (called by Flask routes in server.py)
# ============================================================

def handle_chat(audio_b64, conversation):
    """Handle /chat request. Returns (response_dict, status_code)."""
    global _muted
    llm_mode = _ctx.get("llm_mode", "local")
    llm_model = _ctx.get("llm_model")
    tts_sample_rate = _ctx.get("tts_sample_rate", 24000)
    session = _new_session_state()  # temp session per HTTP request

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception:
        return {"error": "Invalid base64 audio"}, 400

    total_start = time.time()

    # STT
    try:
        t0 = time.time()
        transcript, stt_metrics = stt_transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[STT] {stt_time:.2f}s: {transcript} | SNR={stt_metrics.get('input_snr_est', '?')}dB")
        if not transcript:
            return {"error": "Empty transcript"}, 200
    except Exception as e:
        print(f"[STT] Error: {e}")
        return {"error": f"STT failed: {e}"}, 500

    # Mute/unmute
    mute_cmd = _check_mute_command(transcript)
    if mute_cmd == "mute":
        _muted = True
        print(f"[MUTE] Muted by user: '{transcript}'")
        return {
            "transcript": transcript,
            "response": "Dạ, em sẽ im lặng ạ.",
            "audio_b64": "",
            "sample_rate": tts_sample_rate,
            "latency": {"stt": round(stt_time, 2), "llm": 0, "tts": 0, "total": round(time.time() - total_start, 2)},
            "muted": True,
        }, 200
    elif mute_cmd == "unmute":
        _muted = False
        print(f"[MUTE] Unmuted by user: '{transcript}'")

    # LLM + TTS
    try:
        if llm_mode == "local" and llm_model is not None:
            reply, wav_bytes, latency = chat_streaming_pipeline(transcript, conversation, session_state=session)
            llm_time = latency["llm"]
            tts_time = latency["tts"]
            print(f"[LLM+TTS] Streaming: LLM {llm_time:.2f}s + TTS {tts_time:.2f}s | first_sentence {latency['first_sentence']:.2f}s")
        else:
            t0 = time.time()
            reply = llm_generate_api(transcript, conversation)
            llm_time = time.time() - t0
            print(f"[LLM] {llm_time:.2f}s: {reply}")
            t0 = time.time()
            waveform = tts_synthesize(reply)
            tts_time = time.time() - t0
            wav_bytes = waveform_to_wav_bytes(waveform) if waveform is not None else None

        # Extract session info from this turn
        _extract_session_info(transcript, reply, session)

        if _muted:
            print(f"[MUTE] Muted — skipping audio for: {reply[:80]}")
            wav_bytes = None

        print(f"[LLM] {llm_time:.2f}s: {reply[:80]}")
        audio_out_b64 = base64.b64encode(wav_bytes).decode() if wav_bytes else ""
    except Exception as e:
        print(f"[LLM/TTS] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"LLM/TTS failed: {e}"}, 500

    total = time.time() - total_start
    print(f"[TOTAL] {total:.2f}s | STT {stt_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time:.2f}")

    log_metrics({
        **stt_metrics,
        "stt_latency_ms": round(stt_time * 1000, 1),
        "llm_latency_ms": round(llm_time * 1000, 1),
        "tts_latency_ms": round(tts_time * 1000, 1),
        "total_latency_ms": round(total * 1000, 1),
        "transcript": transcript,
        "response_len": len(reply),
    })

    return {
        "transcript": transcript,
        "response": reply,
        "audio_b64": audio_out_b64,
        "sample_rate": tts_sample_rate,
        "muted": _muted,
        "latency": {
            "stt": round(stt_time, 2),
            "llm": round(llm_time, 2),
            "tts": round(tts_time, 2),
            "total": round(total, 2),
        },
        "metrics": {
            "input_snr": stt_metrics.get("input_snr_est"),
            "stt_confidence": stt_metrics.get("stt_confidence"),
            "noise_suppressed": stt_metrics.get("ns_latency_ms") is not None,
        },
    }, 200


def handle_ws_process(ws, audio_bytes, conversation, cancel_event=None, turn_id=0):
    """WebSocket handler: STT → streaming LLM → streaming TTS chunks."""
    import struct
    global _muted
    llm_mode = _ctx.get("llm_mode", "local")
    llm_model = _ctx.get("llm_model")
    session = _get_session(ws)

    total_start = time.time()
    turn_id_bytes = struct.pack('>I', turn_id)

    def _is_cancelled():
        return cancel_event and cancel_event.is_set()

    def _send_json(obj):
        if _is_cancelled():
            return
        obj["turn_id"] = turn_id
        ws.send(json.dumps(obj))

    def _send_audio(wav_bytes):
        if _is_cancelled():
            return
        ws.send(turn_id_bytes + wav_bytes)

    # STT
    try:
        t0 = time.time()
        transcript, stt_metrics = stt_transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[WS-STT] {stt_time:.2f}s: {transcript} | SNR={stt_metrics.get('input_snr_est', '?')}dB")

        if not transcript:
            _send_json({"type": "error", "error": "Empty transcript"})
            return

        _send_json({"type": "transcript", "text": transcript})
    except Exception as e:
        _send_json({"type": "error", "error": f"STT failed: {e}"})
        return

    # Mute/unmute
    mute_cmd = _check_mute_command(transcript)
    if mute_cmd == "mute":
        _muted = True
        print(f"[WS-MUTE] Muted by user: '{transcript}'")
        _send_json({
            "type": "response",
            "text": "Dạ, em sẽ im lặng ạ.",
            "muted": True,
            "latency": {"stt": round(stt_time, 2), "llm": 0, "tts": 0, "total": round(time.time() - total_start, 2)},
        })
        return
    elif mute_cmd == "unmute":
        _muted = False
        print(f"[WS-MUTE] Unmuted by user: '{transcript}'")

    if _is_cancelled():
        print(f"[WS] Turn {turn_id} cancelled after STT")
        return

    # Backchannel: instantly send a filler clip ~30% of the time
    backchannel_clips = _ctx.get("backchannel_clips", [])
    if backchannel_clips and random.random() < BACKCHANNEL_PROBABILITY:
        clip = random.choice(backchannel_clips)
        _send_audio(clip)
        print(f"[WS] Turn {turn_id} backchannel sent")

    # LLM + TTS
    try:
        if llm_mode == "local" and llm_model is not None:
            llm_t0 = time.time()
            sentence_buffer = ""
            full_response = ""
            tts_time_total = 0.0

            _cancelled = False
            _first_audio_sent = False
            for token in llm_generate_stream(transcript, conversation, session_state=session):
                if _is_cancelled():
                    print(f"[WS] Turn {turn_id} barge-in — stopping LLM")
                    _cancelled = True
                    break
                sentence_buffer += token
                full_response += token

                # Flush TTS at sentence end only (. ? !)
                # F5-TTS has ~1.7s fixed overhead per call, so fewer chunks = faster
                should_flush = SENTENCE_END.search(sentence_buffer)

                if should_flush:
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""
                    print(f"[LLM→TTS] Turn {turn_id} sentence: '{sentence}'")

                    if not _muted:
                        tts_t0 = time.time()
                        waveform = tts_synthesize(sentence)
                        tts_time_total += time.time() - tts_t0
                        if waveform is not None:
                            _send_audio(waveform_to_wav_bytes(waveform))
                            if not _first_audio_sent:
                                _first_audio_sent = True
                                print(f"[WS] Turn {turn_id} first audio at {(time.time()-total_start)*1000:.0f}ms")

            llm_time = time.time() - llm_t0 - tts_time_total

            if not _cancelled and sentence_buffer.strip() and not _muted:
                tts_t0 = time.time()
                waveform = tts_synthesize(sentence_buffer.strip())
                tts_time_total += time.time() - tts_t0
                if waveform is not None:
                    _send_audio(waveform_to_wav_bytes(waveform))

            reply = full_response
        else:
            t0 = time.time()
            reply = llm_generate_api(transcript, conversation)
            llm_time = time.time() - t0

            if _is_cancelled():
                print(f"[WS] Turn {turn_id} cancelled after LLM API")
                return

            if not _muted:
                t0 = time.time()
                waveform = tts_synthesize(reply)
                tts_time_total = time.time() - t0
                if waveform is not None:
                    _send_audio(waveform_to_wav_bytes(waveform))

        # Extract session info from this turn
        _extract_session_info(transcript, reply, session)
        session["last_updated_turn"] = turn_id
        session_summary = {k: v for k, v in session.items() if v and k != "confirmed"}
        print(f"[SESSION] Turn {turn_id}: {session_summary}")

        total = time.time() - total_start
        print(f"[WS] Turn {turn_id} | {total:.2f}s | STT {stt_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time_total:.2f}")
        print(f"  User: {transcript}")
        print(f"  Bot:  {reply[:80]}")

        log_metrics({
            **stt_metrics,
            "turn_id": turn_id,
            "cancelled": _cancelled if llm_mode == "local" else False,
            "stt_latency_ms": round(stt_time * 1000, 1),
            "llm_latency_ms": round(llm_time * 1000, 1),
            "tts_latency_ms": round(tts_time_total * 1000, 1),
            "total_latency_ms": round(total * 1000, 1),
            "transcript": transcript,
            "response_len": len(reply),
            "channel": "websocket",
        })

        if not _is_cancelled():
            _send_json({
                "type": "response",
                "text": reply,
                "muted": _muted,
                "latency": {
                    "stt": round(stt_time, 2),
                    "llm": round(llm_time, 2),
                    "tts": round(tts_time_total, 2),
                    "total": round(total, 2),
                },
                "metrics": {
                    "input_snr": stt_metrics.get("input_snr_est"),
                    "stt_confidence": stt_metrics.get("stt_confidence"),
                    "noise_suppressed": stt_metrics.get("ns_latency_ms") is not None,
                },
            })
        else:
            print(f"[WS] Turn {turn_id} cancelled — response not sent")
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
        _send_json({"type": "error", "error": f"Processing failed: {e}"})
