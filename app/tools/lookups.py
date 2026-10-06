"""The lookup tools, run for real: web_search and get_weather via Tavily (env TAVILY_API_KEY, the key never
logged), get_current_time locally (zoneinfo). Schemas from exp-10's live-mic toolcall_eval. Pure stdlib (urllib).
Each returns a small JSON-able dict; on failure {"error": ...}, so the model can tell the user.
"""
import os, json, urllib.request, urllib.error
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from app.tools.registry import tool

# ---- Tavily (web_search + get_weather) ---------------------------------------------------------
_TAVILY_URL = "https://api.tavily.com/search"
def _tavily(query, max_results=5):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return {"error": "TAVILY_API_KEY is not set on the server"}
    body = json.dumps({"api_key": key, "query": query, "search_depth": "basic",
                       "include_answer": True, "max_results": max_results}).encode()
    req = urllib.request.Request(_TAVILY_URL, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"tavily http {e.code}"}
    except Exception as e:
        return {"error": f"tavily {type(e).__name__}"}
    results = [{"title": x.get("title"), "content": (x.get("content") or "")[:500], "url": x.get("url")}
               for x in d.get("results", [])[:max_results]]
    return {"answer": d.get("answer"), "results": results}

@tool("get_weather", {"type": "object", "properties": {"location": {"type": "string"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]})
def get_weather(location, unit="celsius"):
    r = _tavily(f"current weather right now in {location} in degrees {unit}", max_results=3)
    if "error" in r:
        return r
    return {"location": location, "unit": unit, "answer": r.get("answer"), "sources": r.get("results")}


@tool("web_search", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def web_search(query):
    return _tavily(query)


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
