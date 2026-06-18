"""
security.py — QuickShip defence layer
======================================
Centralises all security checks so every entry point (chat, upload, etc.)
runs the same gauntlet before input reaches the LLM and after output leaves it.

Defence layers
--------------
1. Rate limiting   — blocks IPs that send too many requests.
2. Attack detection — regex + heuristic scan for prompt-injection, jailbreak,
                      recon, and code-execution patterns.
3. Input sanitisation — strips nulls, trims length, normalises whitespace.
4. Output filtering  — scrubs anything that looks like a system prompt leak,
                       API key, or internal stack trace from the LLM reply.
"""

import re
import time
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Rate limiting ────────────────────────────────────────────────────────────

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 20      # requests allowed …
RATE_LIMIT_WINDOW   = 60      # … per this many seconds
CHAT_MAX_CHARS      = 2000    # hard cap on user message length


def check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is within limits, False if it should be blocked."""
    now   = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW
    timestamps = _rate_store[client_ip]
    # Evict old entries
    _rate_store[client_ip] = [t for t in timestamps if t > cutoff]
    if len(_rate_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning("[RATE_LIMIT] IP=%s exceeded %d req/%ds", client_ip, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        return False
    _rate_store[client_ip].append(now)
    return True


# ── Attack detection ─────────────────────────────────────────────────────────

_ATTACK_PATTERNS: list[tuple[str, str]] = [
    # Prompt injection / instruction override
    (r"ignore\s+(all\s+)?(previous|prior|above|my\s+earlier)\s+instructions?",  "instruction_override"),
    (r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",            "instruction_override"),
    (r"forget\s+(everything|all|prior|previous|your|the)\s*(instructions?)?",    "instruction_override"),
    (r"override\s*[\:\-]",                                                        "instruction_override"),
    (r"new\s+instructions?\s*[\:\-]",                                             "instruction_override"),
    # Role / persona hijacking
    (r"you\s+are\s+now\s+(a|an|the)\s+",            "persona_hijack"),
    (r"act\s+as\s+(a|an|the|if)\s+",                "persona_hijack"),
    (r"pretend\s+(to\s+be|you\s+are)",               "persona_hijack"),
    (r"roleplay\s+as",                               "persona_hijack"),
    (r"simulate\s+a\s+",                             "persona_hijack"),
    (r"(developer|admin|god|root|sudo)\s*mode",      "persona_hijack"),
    (r"jailbreak",                                   "persona_hijack"),
    (r"\bdan\b.{0,20}mode",                          "persona_hijack"),
    # System prompt recon
    (r"(reveal|show|print|repeat|output|display|tell me)\s+(your\s+)?(system|instructions?|prompt|config|rules|directives?)", "recon"),
    (r"what\s+(are|is)\s+your\s+(instructions?|rules|prompt|system)",  "recon"),
    (r"how\s+(are\s+you|were\s+you)\s+(configured|programmed|instructed|trained)", "recon"),
    (r"(base64|hex|rot13|encode|decode)\s+(your|the|these)\s+(instructions?|prompt|rules)", "recon"),
    # Code / command execution
    (r"<script[\s>]",             "code_injection"),
    (r"javascript\s*:",           "code_injection"),
    (r"\beval\s*\(",              "code_injection"),
    (r"\bexec\s*\(",              "code_injection"),
    (r"__import__\s*\(",          "code_injection"),
    (r"os\.system\s*\(",          "code_injection"),
    (r"subprocess\.",             "code_injection"),
    # Bypass / exfiltration
    (r"bypass\s+(the\s+)?(security|filter|guard|check)", "bypass"),
    (r"(leak|exfiltrate|steal)\s+(data|info|key|token)",  "bypass"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE | re.DOTALL), label)
    for pat, label in _ATTACK_PATTERNS
]


def detect_attack(text: str) -> Optional[str]:
    """
    Scan text for adversarial patterns.
    Returns the attack category string if found, None if the input is clean.
    """
    for pattern, label in _COMPILED:
        if pattern.search(text):
            logger.warning("[ATTACK] category=%s snippet=%r", label, text[:120])
            return label
    return None


# ── Input sanitisation ────────────────────────────────────────────────────────

def sanitize_input(text: str) -> str:
    """Strip null bytes, collapse excess whitespace, enforce length cap."""
    text = text.replace("\x00", "")          # null bytes
    text = re.sub(r"[\r\n]{3,}", "\n\n", text)  # collapse blank lines
    text = text.strip()
    return text[:CHAT_MAX_CHARS]


# ── Output filtering ──────────────────────────────────────────────────────────

_OUTPUT_SCRUB: list[tuple[re.Pattern, str]] = [
    # Possible system prompt fragments
    (re.compile(r"CRITICAL\s+SECURITY\s+RULES?", re.I),           "[redacted]"),
    (re.compile(r"ABSOLUTE\s+RULES?",            re.I),           "[redacted]"),
    (re.compile(r"SYSTEM_?PROMPT",               re.I),           "[redacted]"),
    (re.compile(r"JSON\s+Output\s*:",            re.I),           "[redacted]"),
    (re.compile(r'"action"\s*:\s*"',             re.I),           "[redacted]"),
    # API / secret keys
    (re.compile(r"sk-[A-Za-z0-9]{20,}"),                         "[key-redacted]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", re.I),        "[token-redacted]"),
    # Internal paths / stack traces
    (re.compile(r'File ".*?", line \d+',         re.I),           "[trace-redacted]"),
    (re.compile(r"Traceback \(most recent",      re.I),           "[trace-redacted]"),
]


def filter_output(text: str) -> str:
    """Remove any sensitive fragments that the LLM might accidentally echo."""
    for pattern, replacement in _OUTPUT_SCRUB:
        text = pattern.sub(replacement, text)
    return text


# ── Unified entry-point check ─────────────────────────────────────────────────

BLOCKED_RESPONSE = (
    "I'm QuickShip Support and I'm here to help with your orders and shipping questions. "
    "If you have a concern about a shipment, refund, or delivery, I'd be happy to assist!"
)


def check_and_sanitize(text: str, client_ip: str = "unknown") -> tuple[str | None, str]:
    """
    Full pipeline: rate-limit → attack-detect → sanitise.

    Returns (None, sanitised_text) if safe, or (BLOCKED_RESPONSE, "") if blocked.
    The caller should return BLOCKED_RESPONSE immediately if it is not None.
    """
    if not check_rate_limit(client_ip):
        return ("I'm receiving too many requests right now. Please wait a moment and try again.", "")

    clean = sanitize_input(text)

    if detect_attack(clean):
        return (BLOCKED_RESPONSE, "")

    return (None, clean)
