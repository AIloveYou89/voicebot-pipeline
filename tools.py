"""
Agent Tools cho Voicebot LLM — Qwen2.5 Function Calling
Mỗi tool có: schema (JSON) + executor (Python function)

Cách thêm tool mới:
    1. Thêm function executor vào đây
    2. Thêm schema vào TOOL_SCHEMAS
    3. Đăng ký trong TOOL_EXECUTORS
    4. Hot-reload: POST /reload
"""
import json
import os
from datetime import datetime, timedelta

# ============================================================
# Data Store (in-memory, có thể thay bằng DB sau)
# ============================================================

# Company info — cấu hình qua env hoặc sửa trực tiếp
COMPANY_INFO = {
    "name": os.environ.get("COMPANY_NAME", "Công ty TNHH Việt Phát"),
    "field": os.environ.get("COMPANY_FIELD", "bất động sản"),
    "address": os.environ.get("COMPANY_ADDRESS", "123 Nguyễn Huệ, Quận 1, TP.HCM"),
    "phone": os.environ.get("COMPANY_PHONE", "028 1234 5678"),
    "hotline": os.environ.get("COMPANY_HOTLINE", "1900 636 123"),
    "email": os.environ.get("COMPANY_EMAIL", "info@vietphat.vn"),
    "website": os.environ.get("COMPANY_WEBSITE", "vietphat.vn"),
    "working_hours": os.environ.get("COMPANY_HOURS", "Thứ 2 đến Thứ 7, 8:00 - 17:30"),
    "services": [
        "Tư vấn mua bán bất động sản",
        "Cho thuê văn phòng, mặt bằng",
        "Quản lý tòa nhà",
        "Tư vấn pháp lý bất động sản",
    ],
    "branches": [
        {"name": "Chi nhánh Quận 1", "address": "123 Nguyễn Huệ, Quận 1"},
        {"name": "Chi nhánh Quận 7", "address": "456 Nguyễn Thị Thập, Quận 7"},
        {"name": "Chi nhánh Thủ Đức", "address": "789 Võ Văn Ngân, Thủ Đức"},
    ],
}

# Schedule — mock data, mỗi slot 30 phút
# Format: {"2026-03-12": {"09:00": null, "09:30": "Nguyễn Văn A", ...}}
_bookings = {}


def _get_date_str(date_str):
    """Parse date string linh hoạt: 'ngày mai', 'thứ 2', '2026-03-12', '12/3'."""
    today = datetime.now()
    s = date_str.lower().strip()

    if s in ("hôm nay", "today", "nay"):
        return today.strftime("%Y-%m-%d")
    if s in ("ngày mai", "mai", "tomorrow"):
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    if s in ("ngày kia", "ngày mốt", "mốt"):
        return (today + timedelta(days=2)).strftime("%Y-%m-%d")

    # Thứ X tuần này/tuần sau
    weekday_map = {
        "thứ 2": 0, "thứ hai": 0, "t2": 0,
        "thứ 3": 1, "thứ ba": 1, "t3": 1,
        "thứ 4": 2, "thứ tư": 2, "t4": 2,
        "thứ 5": 3, "thứ năm": 3, "t5": 3,
        "thứ 6": 4, "thứ sáu": 4, "t6": 4,
        "thứ 7": 5, "thứ bảy": 5, "t7": 5,
        "chủ nhật": 6, "cn": 6,
    }
    for name, wd in weekday_map.items():
        if name in s:
            days_ahead = wd - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Try ISO format: 2026-03-12
    try:
        d = datetime.strptime(s, "%Y-%m-%d")
        return d.strftime("%Y-%m-%d")
    except ValueError:
        pass

    # Try dd/mm: 12/3
    try:
        parts = s.replace("-", "/").split("/")
        if len(parts) == 2:
            day, month = int(parts[0]), int(parts[1])
            d = today.replace(month=month, day=day)
            if d < today:
                d = d.replace(year=today.year + 1)
            return d.strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        pass

    return date_str  # fallback


def _seed_mock_bookings():
    """Tạo một số lịch đã đặt giả lập để demo realistic hơn."""
    today = datetime.now()
    for delta in range(0, 7):
        d = (today + timedelta(days=delta)).strftime("%Y-%m-%d")
        if d not in _bookings:
            _bookings[d] = {}
        # Giả lập một số slot đã đặt
        import random
        random.seed(hash(d) % 1000)  # deterministic per date
        busy_count = random.randint(3, 8)
        all_times = [f"{h:02d}:{m:02d}" for h in range(8, 17) for m in (0, 30)]
        busy_times = random.sample(all_times, min(busy_count, len(all_times)))
        names = ["Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Cường",
                 "Phạm Minh Đức", "Võ Thị Em", "Đặng Quốc Phong",
                 "Huỳnh Thanh Giang", "Bùi Thị Hoa"]
        for i, t in enumerate(busy_times):
            _bookings[d][t] = names[i % len(names)]

_seed_mock_bookings()


def _generate_slots(date_str):
    """Tạo danh sách slot cho 1 ngày (8:00 - 17:00, mỗi 30 phút)."""
    if date_str not in _bookings:
        _bookings[date_str] = {}
    slots = _bookings[date_str]

    all_slots = []
    for hour in range(8, 17):
        for minute in (0, 30):
            t = f"{hour:02d}:{minute:02d}"
            booked_by = slots.get(t)
            all_slots.append({
                "time": t,
                "available": booked_by is None,
                "booked_by": booked_by,
            })
    return all_slots


# ============================================================
# Tool Executors
# ============================================================

def tool_get_company_info(args):
    """Trả thông tin công ty."""
    field = args.get("field", "all")

    if field == "all":
        return {
            "name": COMPANY_INFO["name"],
            "field": COMPANY_INFO["field"],
            "address": COMPANY_INFO["address"],
            "phone": COMPANY_INFO["phone"],
            "hotline": COMPANY_INFO["hotline"],
            "working_hours": COMPANY_INFO["working_hours"],
            "services": COMPANY_INFO["services"],
        }
    elif field == "address":
        return {
            "address": COMPANY_INFO["address"],
            "branches": COMPANY_INFO["branches"],
        }
    elif field == "services":
        return {"services": COMPANY_INFO["services"]}
    elif field == "contact":
        return {
            "phone": COMPANY_INFO["phone"],
            "hotline": COMPANY_INFO["hotline"],
            "email": COMPANY_INFO["email"],
            "website": COMPANY_INFO["website"],
        }
    elif field == "hours":
        return {"working_hours": COMPANY_INFO["working_hours"]}
    else:
        return {"info": COMPANY_INFO.get(field, "Không có thông tin")}


def tool_check_schedule(args):
    """Xem lịch trống cho 1 ngày."""
    raw_date = args.get("date", "ngày mai")
    date_str = _get_date_str(raw_date)

    # Check if weekend (Sunday)
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        if d.weekday() == 6:  # Sunday
            return {
                "date": date_str,
                "day_of_week": "Chủ nhật",
                "note": "Chủ nhật không làm việc. Vui lòng chọn ngày khác (Thứ 2 - Thứ 7).",
                "available_slots": [],
            }
        day_names = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ nhật"]
        day_name = day_names[d.weekday()]
    except ValueError:
        day_name = ""

    slots = _generate_slots(date_str)
    available = [s["time"] for s in slots if s["available"]]

    return {
        "date": date_str,
        "day_of_week": day_name,
        "total_slots": len(slots),
        "available_count": len(available),
        "available_slots": available,
        "note": f"Còn {len(available)} slot trống" if available else "Hết slot, vui lòng chọn ngày khác",
    }


def tool_book_appointment(args):
    """Đặt lịch hẹn."""
    raw_date = args.get("date", "")
    time_slot = args.get("time", "")
    customer_name = args.get("customer_name", "")
    customer_phone = args.get("customer_phone", "")
    purpose = args.get("purpose", "Tư vấn")

    if not raw_date or not time_slot:
        return {"success": False, "error": "Cần có ngày và giờ để đặt lịch"}
    if not customer_name:
        return {"success": False, "error": "Cần có tên khách hàng"}

    date_str = _get_date_str(raw_date)

    # Validate time format
    try:
        h, m = time_slot.replace("h", ":").replace("H", ":").split(":")
        time_slot = f"{int(h):02d}:{int(m):02d}"
    except (ValueError, AttributeError):
        return {"success": False, "error": f"Giờ không hợp lệ: {time_slot}"}

    # Check if slot exists and available
    slots = _generate_slots(date_str)
    slot_times = [s["time"] for s in slots]

    if time_slot not in slot_times:
        return {
            "success": False,
            "error": f"Giờ {time_slot} không nằm trong khung giờ làm việc (08:00 - 17:00)",
        }

    if _bookings[date_str].get(time_slot) is not None:
        return {
            "success": False,
            "error": f"Slot {time_slot} ngày {date_str} đã có người đặt. Vui lòng chọn giờ khác.",
            "alternative_slots": [s["time"] for s in slots if s["available"]][:5],
        }

    # Book it
    _bookings[date_str][time_slot] = customer_name

    return {
        "success": True,
        "booking": {
            "date": date_str,
            "time": time_slot,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "purpose": purpose,
        },
        "message": f"Đã đặt lịch thành công cho {customer_name} lúc {time_slot} ngày {date_str}",
    }


# ============================================================
# Tool Schemas (OpenAI-compatible format, Qwen2.5 supports)
# ============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": "Lấy thông tin công ty: địa chỉ, số điện thoại, giờ làm việc, dịch vụ, chi nhánh. Gọi khi khách hỏi về công ty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["all", "address", "services", "contact", "hours"],
                        "description": "Loại thông tin cần lấy. 'all' = tất cả, 'address' = địa chỉ + chi nhánh, 'services' = dịch vụ, 'contact' = SĐT/email, 'hours' = giờ làm việc",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_schedule",
            "description": "Xem lịch hẹn còn trống ngày nào đó. Gọi khi khách muốn biết ngày/giờ nào còn trống.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Ngày cần kiểm tra. VD: 'ngày mai', 'thứ 2', '15/3', '2026-03-15'",
                    },
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Đặt lịch hẹn cho khách hàng. Cần có: ngày, giờ, tên khách. Gọi khi khách xác nhận muốn đặt lịch.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Ngày hẹn. VD: 'ngày mai', 'thứ 5', '20/3'",
                    },
                    "time": {
                        "type": "string",
                        "description": "Giờ hẹn. VD: '09:00', '14:30', '10h00'",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Tên khách hàng",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Số điện thoại khách hàng (nếu có)",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Mục đích cuộc hẹn. VD: 'Tư vấn mua nhà', 'Xem dự án'",
                    },
                },
                "required": ["date", "time", "customer_name"],
            },
        },
    },
]

# ============================================================
# Executor Registry
# ============================================================

TOOL_EXECUTORS = {
    "get_company_info": tool_get_company_info,
    "check_schedule": tool_check_schedule,
    "book_appointment": tool_book_appointment,
}


def execute_tool(name, arguments):
    """Execute a tool by name. Returns result dict."""
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        result = executor(arguments)
        print(f"[TOOL] {name}({arguments}) → {json.dumps(result, ensure_ascii=False)[:200]}")
        return result
    except Exception as e:
        print(f"[TOOL] {name} error: {e}")
        return {"error": str(e)}

