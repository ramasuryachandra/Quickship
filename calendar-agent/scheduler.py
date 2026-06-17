import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler

from calendar_tools import get_upcoming_events
from email_tools import send_email

logger = logging.getLogger(__name__)


def _send_daily_reminders() -> None:
    """Runs at 8 AM daily — emails upcoming events for the next 24 hours."""
    recipient = os.getenv("REMINDER_EMAIL")
    if not recipient:
        logger.warning("REMINDER_EMAIL not set; skipping scheduled reminders.")
        return

    events = get_upcoming_events(days=1)
    if not events or (len(events) == 1 and "error" in events[0]):
        logger.info("No events found for daily reminder.")
        return

    lines = []
    for e in events:
        lines.append(f"• {e['title']}")
        lines.append(f"  When: {e['start']} – {e['end']}")
        if e.get("location"):
            lines.append(f"  Where: {e['location']}")
        if e.get("description"):
            lines.append(f"  Note: {e['description'][:120]}")
        lines.append("")

    body = (
        "Good morning! Here are your events for today:\n\n"
        + "\n".join(lines)
        + "\n— Your Calendar Agent"
    )

    result = send_email(
        to=recipient,
        subject=f"📅 Your events for today",
        body=body,
    )
    if result.get("success"):
        logger.info("Daily reminder sent to %s (%d events)", recipient, len(events))
    else:
        logger.error("Failed to send daily reminder: %s", result.get("error"))


def start_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        _send_daily_reminders,
        trigger="cron",
        hour=8,
        minute=0,
        id="daily_reminders",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — daily reminders fire at 08:00 UTC.")
    return scheduler
