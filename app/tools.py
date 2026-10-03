"""exp-12 EXTRA: tool layer for the function-calling MOUTH. Tool SCHEMAS are exp-10's (live-mic
toolcall_eval), the Hermes <tool_call>{"name","arguments"}</tool_call> format the Qwen responder emits.
Executors (this file): web_search + get_weather via Tavily (env TAVILY_API_KEY, key never logged),
get_current_time local (zoneinfo), set_reminder/send_message local stubs. Pure stdlib (urllib) so it runs
in venv_cbx with no extra deps. Any executor returns a small JSON-able dict; on failure it returns
{"error": ...} so the model can tell the user instead of crashing the turn.
"""
import os, json, re, urllib.request, urllib.error
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---- schemas (exp-10 live-mic toolcall_eval TOOLS, verbatim) -----------------------------------
TOOLS = [
 {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a location.",
   "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City, e.g. 'Paris'"},
     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}},
 {"type": "function", "function": {"name": "web_search", "description": "Search the web for current, real-time, or recent information. USE THIS whenever the user asks for anything time-sensitive or that may have changed since your training: current prices (crypto, stocks, currencies), sports scores and results, breaking or recent news, today's events, or any fact you are not certain is up to date. Prefer calling this over saying you don't know or telling the user to check a website.",
   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
 {"type": "function", "function": {"name": "get_current_time", "description": "Get the current time in a timezone or city.",
   "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "IANA tz or city"}}, "required": ["timezone"]}}},
 {"type": "function", "function": {"name": "set_reminder", "description": "Create a reminder for the user.",
   "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]}}},
 {"type": "function", "function": {"name": "send_message", "description": "Send a text message to a contact.",
   "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "body"]}}},
]

TCALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TFUNC = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
TPARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def _param(v):
    """An XML-format parameter value: JSON if it parses as a number/bool/list/object, else the raw string."""
    try:
        x = json.loads(v)
        return x if not isinstance(x, str) else v
    except Exception:
        return v


def parse_calls(text):
    """Tool calls in either format the Qwen chat templates emit, inside <tool_call>...</tool_call>:
    Hermes JSON (Qwen3-2507: {"name": ..., "arguments": {...}}) or XML (Qwen3.5:
    <function=NAME><parameter=P>value</parameter>...</function>)."""
    out = []
    for m in TCALL.finditer(text):
        body = m.group(1).strip()
        f = TFUNC.search(body)
        if f:
            out.append({"name": f.group(1), "arguments": {p: _param(v) for p, v in TPARAM.findall(f.group(2))}})
            continue
        try:
            out.append(json.loads(body))
        except Exception:
            out.append({"_unparseable": body[:120]})
    return out

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

def web_search(query):
    return _tavily(query)

def get_weather(location, unit="celsius"):
    r = _tavily(f"current weather right now in {location} in degrees {unit}", max_results=3)
    if "error" in r:
        return r
    return {"location": location, "unit": unit, "answer": r.get("answer"), "sources": r.get("results")}

# ---- local time ---------------------------------------------------------------------------------
_CITY_TZ = {"london": "Europe/London", "paris": "Europe/Paris", "berlin": "Europe/Berlin",
            "tokyo": "Asia/Tokyo", "new york": "America/New_York", "los angeles": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles", "chicago": "America/Chicago", "sydney": "Australia/Sydney",
            "moscow": "Europe/Moscow", "dubai": "Asia/Dubai", "singapore": "Asia/Singapore",
            "hong kong": "Asia/Hong_Kong", "delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
            "utc": "UTC", "almaty": "Asia/Almaty"}
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

# ---- placeholder stubs: NOT connected. Return a demo marker so the model TELLS the user it isn't done
# yet (it must not pretend a reminder was set or a message was sent). ----------------------------------
# Privacy: never log the reminder text / message body / recipient -- they're user content.
def set_reminder(text, time):
    return {"demo": True, "note": "Reminders aren't connected yet.", "text": text, "time": time}

def send_message(recipient, body):
    return {"demo": True, "note": "Messaging isn't connected yet.", "recipient": recipient, "body": body}

_DISPATCH = {"get_weather": get_weather, "web_search": web_search, "get_current_time": get_current_time,
             "set_reminder": set_reminder, "send_message": send_message}

_DELAY_S = float(os.environ.get("TYGER_TOOL_DELAY_S") or 0)   # testing only: a slow tool (L-04 / I-03 checks)


def execute_tool(name, args):
    if _DELAY_S:
        import time
        time.sleep(_DELAY_S)
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'"}
    if not isinstance(args, dict):
        return {"error": f"bad arguments for '{name}'"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for '{name}': {e}"}
    except Exception as e:
        return {"error": f"{name} failed: {type(e).__name__}"}
