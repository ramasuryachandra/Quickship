import json
import os
from datetime import date

from openai import AsyncOpenAI

from calendar_tools import get_upcoming_events, get_events_on_date, search_events
from email_tools import send_email

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return _client


SYSTEM_PROMPT = """You are a helpful calendar assistant. Today's date is {today}.

You can:
- Retrieve upcoming events from the user's calendar
- Look up events on a specific date
- Search events by keyword (title, location, description)
- Send email reminders for events

Be concise and friendly. When listing events, format them clearly with time, title, and location.
If no calendar is loaded yet, tell the user to upload a .ics file using the button above the chat."""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": "Get upcoming calendar events within the next N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "How many days ahead to look (default 7, max 60).",
                        "default": 7,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events_on_date",
            "description": "Get all calendar events on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format.",
                    }
                },
                "required": ["date_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Search upcoming events by keyword in title, location, or description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword.",
                    },
                    "days": {
                        "type": "integer",
                        "description": "How many days ahead to search (default 30).",
                        "default": 30,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_reminder_email",
            "description": "Send an email reminder about a calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body with event details.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict) -> object:
    if name == "get_upcoming_events":
        return get_upcoming_events(**args)
    if name == "get_events_on_date":
        return get_events_on_date(**args)
    if name == "search_events":
        return search_events(**args)
    if name == "send_reminder_email":
        return send_email(args["to"], args["subject"], args["body"])
    return {"error": f"Unknown tool: {name}"}


async def run_agent(user_message: str, history: list[dict]) -> str:
    """Run one turn of the agent loop and return the assistant reply."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = _get_client()

    # Tool-calling loop — keeps going until the model returns a plain text reply
    while True:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        # Append the assistant's tool-call request to the context
        messages.append(msg)

        # Execute every requested tool and feed results back
        for tc in msg.tool_calls:
            result = _execute_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )
