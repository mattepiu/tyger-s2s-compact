"""The lookup tool, run for real: get_current_time, locally (zoneinfo). Pure stdlib. It returns a small JSON-able
dict; on failure {"error": ...}, so the model can tell the user.

There is no web search or weather tool here (removed 2026-10-11, the user's call): live facts (weather, news,
prices, scores) go to the expert or a background task, and the `smart` model looks them up with its own web search
(the `openai-responses` plugin).
"""
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from app.tools.registry import tool


# ---- local time ---------------------------------------------------------------------------------
_CITY_TZ = {"london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
            "tokyo": "Asia/Tokyo", "new york": "America/New_York", "los angeles": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles", "chicago": "America/Chicago", "sydney": "Australia/Sydney",
            "moscow": "Europe/Moscow", "dubai": "Asia/Dubai", "singapore": "Asia/Singapore",
            "hong kong": "Asia/Hong_Kong", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
            "utc": "UTC", "almaty": "Asia/Almaty"}


@tool("get_current_time", {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]})
def get_current_time(timezone):
    if ZoneInfo is None:
        return {"error": "zoneinfo unavailable"}
    tzname = _CITY_TZ.get(timezone.strip().lower(), timezone.strip())
    try:
        now = datetime.now(ZoneInfo(tzname))
    except Exception:
        return {"error": f"unknown timezone or city '{timezone}'"}
    return {"timezone": tzname, "time": now.strftime("%H:%M"), "date": now.strftime("%A, %d %B %Y"),
            "tz_abbrev": now.tzname()}
