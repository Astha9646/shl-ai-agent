"""
Rule-based guardrail checks for user input.
"""

from __future__ import annotations

import re

# --- prompt injection ---
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules)",
        r"disregard\s+(your\s+)?(instructions|rules|guidelines|programming)",
        r"forget\s+(everything|all)\s+(you\s+)?(were|have\s+been)\s+told",
        r"override\s+(your\s+)?(instructions|rules|safety|programming)",
        r"you\s+are\s+now\s+(a|an|in)\s+",
        r"act\s+as\s+(if\s+you\s+have\s+no|without)\s+(restrictions|limits|rules)",
        r"pretend\s+(you\s+are|to\s+be)\s+(not\s+)?(an?\s+)?(ai|assistant|chatbot)",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
        r"show\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions)",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions)",
        r"\b(jailbreak|dan\s+mode|developer\s+mode|sudo\s+mode)\b",
        r"do\s+anything\s+now",
        r"new\s+instructions\s*:",
        r"<\s*/?\s*system\s*>",
        r"\[INST\]|\[/INST\]",
        r"role\s*:\s*system",
        r"bypass\s+(your\s+)?(filters|guardrails|safety)",
        r"no\s+longer\s+follow\s+(any\s+)?rules",
    )
)

# --- legal advice ---
_LEGAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\blegal\s+advice\b",
        r"\b(need|want|give\s+me)\s+(some\s+)?legal\s+advice\b",
        r"\bshould\s+i\s+sue\b",
        r"\bcan\s+i\s+sue\b",
        r"\bam\s+i\s+liable\b",
        r"\bis\s+it\s+legal\s+to\b",
        r"\bwhat\s+(does|is)\s+the\s+law\s+(say|on)\b",
        r"\b(hire|consult)\s+(a\s+)?(lawyer|attorney|solicitor)\b",
        r"\b(lawsuit|litigation|class\s+action)\b",
        r"\bwrongful\s+termination\b.*\b(legal|sue|lawyer)\b",
        r"\b(contract|employment)\s+law\s+advice\b",
        r"\blegal\s+opinion\b",
    )
)

# Hiring / assessment domain (on-topic for this agent)
_ON_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(shl|assessment|assessments)\b",
        r"\b(hire|hiring|recruit|recruitment|candidate|candidates)\b",
        r"\b(job|role|position|vacancy|opening)\b",
        r"\b(employee|employees|staff|workforce|talent)\b",
        r"\b(interview|screening|evaluate|evaluation)\b",
        r"\b(skill|skills|competenc(y|ies)|aptitude)\b",
        r"\b(personality|cognitive|behavioral|behavioural)\b",
        r"\b(test|tests|testing|psychometric)\b",
        r"\b(senior|junior|mid[- ]level|entry[- ]level)\b",
        r"\b(developer|engineer|manager|analyst|designer)\b",
        r"\b(communication|numerical|verbal)\s+(skills?|reasoning|test)\b",
        r"\brecommend\s+(an?\s+)?(assessment|test)\b",
        r"\bwhich\s+(shl\s+)?(test|assessment)\b",
    )
)

# Clearly unrelated to talent / assessment workflows
_OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(weather|forecast)\s+(today|tomorrow|this week)\b",
        r"\b(write|compose)\s+(a\s+)?(poem|story|novel|song lyrics)\b",
        r"\b(recipe|cook|bake)\s+for\b",
        r"\b(stock|crypto|bitcoin)\s+(price|prediction|trading)\b",
        r"\b(sports?\s+score|football|cricket|nba)\b",
        r"\b(movie|netflix)\s+recommend",
        r"\b(diagnose|symptoms|medical\s+advice)\b",
        r"\b(homework|essay)\s+(help|due)\b",
        r"\btranslate\s+this\s+(paragraph|text)\s+to\b",
        r"\b(python|javascript)\s+bug\b.*\b(fix|debug)\b",  # general coding help
        r"\bwho\s+won\s+(the\s+)?(election|world\s+cup)\b",
        r"\bhoroscope\b",
        r"\bplay\s+chess\b",
    )
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _any_match(patterns: tuple[re.Pattern[str], ...], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def is_prompt_injection(text: str) -> bool:
    """Return True if the text looks like a prompt-injection attempt."""
    t = _normalize(text)
    if not t:
        return False
    return _any_match(_INJECTION_PATTERNS, t)


def is_legal_request(text: str) -> bool:
    """Return True if the text asks for legal advice or legal interpretation."""
    t = _normalize(text)
    if not t:
        return False
    return _any_match(_LEGAL_PATTERNS, t)


def is_off_topic(text: str) -> bool:
    """
    Return True if the text appears unrelated to hiring / SHL assessments.

    On-topic hiring or assessment language overrides generic off-topic cues.
    """
    t = _normalize(text)
    if not t:
        return False

    on_topic = _any_match(_ON_TOPIC_PATTERNS, t)
    if on_topic:
        return False

    return _any_match(_OFF_TOPIC_PATTERNS, t)
