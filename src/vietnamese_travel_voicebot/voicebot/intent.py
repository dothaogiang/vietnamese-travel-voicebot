"""Small deterministic intent and entity recognizer for travel text queries."""

import re

from vietnamese_travel_voicebot.data.normalize import normalize_transcript

DESTINATIONS = (
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Huế", "Đà Lạt", "Nha Trang", "Hội An",
    "Phú Quốc", "Hạ Long", "Sa Pa", "Sapa", "Mũi Né", "Vũng Tàu", "Quy Nhơn",
)


def classify_intent(text: str) -> str:
    query = normalize_transcript(text)
    if any(term in query for term in ("khách sạn", "đặt phòng", "homestay", "nghỉ ở")):
        return "accommodation"
    if any(term in query for term in ("xe", "tàu", "máy bay", "di chuyển", "đi lại", "phương tiện")):
        return "transportation"
    if any(term in query for term in ("lịch trình", "mấy ngày", "ngày", "đêm", "hành trình")):
        return "itinerary"
    if any(term in query for term in ("chơi", "tham quan", "hoạt động", "làm gì")):
        return "activities"
    return "destination_info"


def extract_entities(text: str) -> dict[str, list[str]]:
    normalized = normalize_transcript(text)
    destinations = [name for name in DESTINATIONS if normalize_transcript(name) in normalized]
    duration = re.search(r"\b(\d+)\s*(ngày|đêm)\b", normalized)
    return {
        "destinations": destinations,
        "duration": [f"{duration.group(1)} {duration.group(2)}"] if duration else [],
    }
