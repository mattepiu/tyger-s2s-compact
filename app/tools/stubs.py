"""The action tools, placeholders: NOT connected. They return a demo marker so the model TELLS the user it isn't done
yet (it must not pretend a reminder was set or a message was sent).
Privacy: never log the reminder text / message body / recipient -- they're user content.
"""
from app.tools.registry import tool


@tool("set_reminder", {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}},
                       "required": ["text", "time"]})
def set_reminder(text, time):
    return {"demo": True, "note": "Reminders aren't connected yet.", "text": text, "time": time}


@tool("send_message", {"type": "object", "properties": {"recipient": {"type": "string"}, "body": {"type": "string"}},
                       "required": ["recipient", "body"]})
def send_message(recipient, body):
    return {"demo": True, "note": "Messaging isn't connected yet.", "recipient": recipient, "body": body}
