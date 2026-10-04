"""exp-12 EXTRA: tool layer for the function-calling MOUTH. Tool SCHEMAS are exp-10's (live-mic
toolcall_eval), in the OpenAI function format; each LLM plugin converts them and parses its model's calls.
Executors (this file): web_search + get_weather via Tavily (env TAVILY_API_KEY, key never logged),
get_current_time local (zoneinfo), set_reminder/send_message local stubs. Pure stdlib (urllib) so it runs
in venv_cbx with no extra deps. Any executor returns a small JSON-able dict; on failure it returns
{"error": ...} so the model can tell the user instead of crashing the turn.
"""
import os, json, urllib.request, urllib.error
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# ---- schemas (exp-10 live-mic toolcall_eval TOOLS): names, types, required. The tool and parameter
# descriptions are prompts and live in prompts.yaml (`tools:`); build_tools joins the two. -------------------
TOOL_SCHEMAS = [
 {"type": "function", "function": {"name": "get_weather",
   "parameters": {"type": "object", "properties": {"location": {"type": "string"},
     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}},
 {"type": "function", "function": {"name": "web_search",
   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
 {"type": "function", "function": {"name": "get_current_time",
   "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}}},
 {"type": "function", "function": {"name": "set_reminder",
   "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]}}},
 {"type": "function", "function": {"name": "send_message",
   "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "body": {"type": "string"}}, "required": ["recipient", "body"]}}},
]
# escalation (PLUGINS.md step 10): offered to the `reply` model only when a `smart` model is configured. MOUTH runs
# it itself (the smart model answers the turn); it never reaches execute_tool.
EXPERT = "ask_expert"
EXPERT_SCHEMA = {"type": "function", "function": {"name": EXPERT,
   "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}}
# background tasks (PLUGINS.md steps 12-14): start / cancel offered to the `reply` model with ask_expert when
# mouth.tasks.enabled; finish_task only to the smart model running a task. MOUTH handles all three itself.
START, CANCEL, FINISH = "start_task", "cancel_task", "finish_task"
TASK_SCHEMAS = [
 {"type": "function", "function": {"name": START,
   "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}}},
 {"type": "function", "function": {"name": CANCEL,
   "parameters": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}}},
]
FINISH_SCHEMA = {"type": "function", "function": {"name": FINISH,
   "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "result": {"type": "string"}},
                  "required": ["summary", "result"]}}}
ALL_SCHEMAS = TOOL_SCHEMAS + [EXPERT_SCHEMA] + TASK_SCHEMAS + [FINISH_SCHEMA]


def build_tools(prompts, expert=False, tasks=False, finish=False):
    """The OpenAI function schemas with their descriptions from `prompts` (app.prompts.Prompts): the lookup /
    action tools, then ask_expert if `expert`, start_task / cancel_task if `tasks`, finish_task if `finish` (the
    smart model running a task). Key order is the original's (name, description, parameters; a property's type,
    then description), since chat templates render the JSON as given."""
    text = prompts.tool_text(); out = []
    for t in TOOL_SCHEMAS + ([EXPERT_SCHEMA] if expert else []) + (TASK_SCHEMAS if tasks else []) + \
            ([FINISH_SCHEMA] if finish else []):
        f = t["function"]; d = text[f["name"]]; pdesc = d.get("params") or {}
        props = {}
        for p, spec in f["parameters"]["properties"].items():
            props[p] = {"type": spec["type"], "description": pdesc[p],
                        **{k: v for k, v in spec.items() if k != "type"}} if p in pdesc else dict(spec)
        out.append({"type": "function", "function": {"name": f["name"], "description": d["description"],
                    "parameters": {**f["parameters"], "properties": props}}})
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
