from datetime import datetime, timedelta, date
from pathlib import Path
import os

from icalendar import Calendar
import recurring_ical_events


def _load_calendar() -> Calendar | None:
    path = Path(os.getenv("CALENDAR_PATH", "calendar.ics"))
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return Calendar.from_ical(f.read())


def _format_event(event: object) -> dict:
    dtstart = event.get("DTSTART")
    dtend = event.get("DTEND")

    start = dtstart.dt if dtstart else None
    end = dtend.dt if dtend else None

    def fmt(dt) -> str:
        if dt is None:
            return "Unknown"
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d %H:%M")
        return dt.strftime("%Y-%m-%d") + " (all-day)"

    return {
        "title": str(event.get("SUMMARY", "Untitled")),
        "start": fmt(start),
        "end": fmt(end),
        "location": str(event.get("LOCATION", "") or ""),
        "description": str(event.get("DESCRIPTION", "") or ""),
    }


def get_upcoming_events(days: int = 7) -> list[dict]:
    """Return events starting within the next `days` days."""
    cal = _load_calendar()
    if cal is None:
        return [{"error": "No calendar file found. Please upload a .ics file first."}]

    now = datetime.now()
    end = now + timedelta(days=days)
    events = recurring_ical_events.of(cal).between(now, end)
    return [_format_event(e) for e in sorted(events, key=lambda e: e.get("DTSTART").dt)]


def get_events_on_date(date_str: str) -> list[dict]:
    """Return all events on a specific date (YYYY-MM-DD)."""
    cal = _load_calendar()
    if cal is None:
        return [{"error": "No calendar file found. Please upload a .ics file first."}]

    try:
        target = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return [{"error": f"Invalid date format: {date_str}. Use YYYY-MM-DD."}]

    events = recurring_ical_events.of(cal).at(target)
    return [_format_event(e) for e in events]


def search_events(query: str, days: int = 30) -> list[dict]:
    """Search events by keyword in title, description, or location."""
    upcoming = get_upcoming_events(days)
    q = query.lower()
    return [
        e for e in upcoming
        if q in e.get("title", "").lower()
        or q in e.get("description", "").lower()
        or q in e.get("location", "").lower()
    ]
