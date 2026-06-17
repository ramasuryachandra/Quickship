import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email via Gmail SMTP using credentials from environment."""
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not gmail_user or not gmail_password:
        return {
            "success": False,
            "error": "Gmail credentials not set. Add GMAIL_USER and GMAIL_APP_PASSWORD to your .env file.",
        }

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, to, msg.as_string())
        return {"success": True, "message": f"Reminder sent to {to}"}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "Gmail authentication failed. Check your App Password.",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
