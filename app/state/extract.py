"""
Keyword / rule-based extraction of ConversationState from chat history.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence, Union

from app.state.models import ConversationState

# --- role phrases (longest first so e.g. "software engineer" beats "engineer") ---
_ROLE_PHRASES: tuple[str, ...] = (
    "machine learning engineer",
    "ml engineer",
    "site reliability engineer",
    "sre engineer",
    "full stack developer",
    "full-stack developer",
    "frontend developer",
    "front-end developer",
    "backend developer",
    "back-end developer",
    "software engineer",
    "data engineer",
    "data scientist",
    "product manager",
    "project manager",
    "program manager",
    "business analyst",
    "sales engineer",
    "devops engineer",
    "cloud engineer",
    "security engineer",
    "qa engineer",
    "test engineer",
    "hr business partner",
    "account manager",
    "customer success",
    "developer",
    "engineer",
    "analyst",
    "architect",
    "designer",
    "consultant",
    "manager",
    "director",
)

# --- seniority: canonical label -> regex (first match in list order wins) ---
_SENIORITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("executive", re.compile(r"\b(ceo|cto|cfo|coo|chief|vp|vice president|executive)\b")),
    ("director", re.compile(r"\b(director|head of)\b")),
    ("lead", re.compile(r"\b(tech lead|team lead|lead engineer|staff engineer|principal)\b")),
    ("senior", re.compile(r"\b(senior|sr\.?|iii|iv)\b")),
    ("mid", re.compile(r"\b(mid|middle|intermediate|experienced|ii\b|l3|l4)\b")),
    ("junior", re.compile(r"\b(junior|jr\.?|entry|graduate|intern|associate|i\b|l1|l2)\b")),
)

# --- technical skills: phrase -> regex (longer phrases first) ---
_SKILL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("machine learning", re.compile(r"\bmachine learning\b|\bml\b")),
    ("deep learning", re.compile(r"\bdeep learning\b")),
    ("react native", re.compile(r"\breact native\b")),
    ("node.js", re.compile(r"\bnode\.?js\b")),
    ("vue.js", re.compile(r"\bvue\.?js\b")),
    ("angular", re.compile(r"\bangular\b")),
    ("react", re.compile(r"\breact\b")),
    ("kubernetes", re.compile(r"\bkubernetes\b|\bk8s\b")),
    ("docker", re.compile(r"\bdocker\b")),
    ("terraform", re.compile(r"\bterraform\b")),
    ("aws", re.compile(r"\baws\b")),
    ("azure", re.compile(r"\bazure\b")),
    ("gcp", re.compile(r"\bgcp\b|\bgoogle cloud\b")),
    ("typescript", re.compile(r"\btypescript\b|\bts\b")),
    ("javascript", re.compile(r"\bjavascript\b|\bjs\b")),
    ("c++", re.compile(r"\bc\+\+\b")),
    ("c#", re.compile(r"\bc#\b")),
    ("java", re.compile(r"\bjava\b")),
    ("kotlin", re.compile(r"\bkotlin\b")),
    ("swift", re.compile(r"\bswift\b")),
    ("go", re.compile(r"\bgo\b|\bgolang\b")),
    ("rust", re.compile(r"\brust\b")),
    ("ruby", re.compile(r"\bruby\b")),
    ("rails", re.compile(r"\brails\b|\bruby on rails\b")),
    ("django", re.compile(r"\bdjango\b")),
    ("flask", re.compile(r"\bflask\b")),
    ("fastapi", re.compile(r"\bfastapi\b")),
    ("spring", re.compile(r"\bspring boot\b|\bspring\b")),
    ("python", re.compile(r"\bpython\b")),
    ("sql", re.compile(r"\bsql\b")),
    ("postgresql", re.compile(r"\bpostgres(ql)?\b")),
    ("mysql", re.compile(r"\bmysql\b")),
    ("mongodb", re.compile(r"\bmongo(db)?\b")),
    ("redis", re.compile(r"\bredis\b")),
    ("kafka", re.compile(r"\bkafka\b")),
    ("spark", re.compile(r"\bspark\b")),
    ("pandas", re.compile(r"\bpandas\b")),
    ("linux", re.compile(r"\blinux\b")),
    ("bash", re.compile(r"\bbash\b|\bshell scripting\b")),
    ("html", re.compile(r"\bhtml\b")),
    ("css", re.compile(r"\bcss\b")),
    (".net", re.compile(r"\.net\b|\bdotnet\b")),
)

_PERSONALITY = re.compile(
    r"personality|behavioral|behavioural|psychometric|culture fit|values assessment|"
    r"opq|big five|big5|hexaco|disc\b|situational judgment|situational judgement|"
    r"integrity test|work style|temperament",
    re.IGNORECASE,
)

_COGNITIVE = re.compile(
    r"cognitive|aptitude|numerical reasoning|verbal reasoning|logical reasoning|"
    r"\biq\b|abstract reasoning|inductive reasoning|deductive reasoning|"
    r"spatial reasoning|problem[- ]solving assessment|critical thinking test",
    re.IGNORECASE,
)

_COMMUNICATION = re.compile(
    r"communication skills|interpersonal skills|presentation skills|public speaking|"
    r"written communication|stakeholder communication|client[- ]facing|"
    r"influencing skills|negotiation skills",
    re.IGNORECASE,
)


def _flatten_messages(
    messages: Sequence[Union[str, Mapping[str, str]]],
) -> str:
    parts: list[str] = []
    for m in messages:
        if isinstance(m, str):
            parts.append(m)
        elif isinstance(m, Mapping):
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            else:
                parts.append(str(c or ""))
        else:
            parts.append(str(m))
    return "\n".join(parts)


def _find_role(text_lower: str) -> str | None:
    for phrase in _ROLE_PHRASES:
        if phrase in text_lower:
            return " ".join(word.capitalize() for word in phrase.split())
    return None


def _find_seniority(text_lower: str) -> str | None:
    for label, pat in _SENIORITY_RULES:
        if pat.search(text_lower):
            return label
    return None


def _find_skills(text_lower: str) -> list[str]:
    found: set[str] = set()
    for label, pat in _SKILL_PATTERNS:
        if pat.search(text_lower):
            found.add(label)
    return sorted(found, key=str.lower)


def extract_conversation_state(
    messages: Sequence[Union[str, Mapping[str, str]]],
) -> ConversationState:
    """
    Parse conversation history (strings or ``{"role": ..., "content": ...}`` dicts)
    and return a :class:`ConversationState` using keyword / regex rules.
    """
    raw = _flatten_messages(messages)
    text_lower = raw.lower()

    return ConversationState(
        role=_find_role(text_lower),
        seniority=_find_seniority(text_lower),
        skills=_find_skills(text_lower),
        personality_needed=bool(_PERSONALITY.search(raw)),
        cognitive_needed=bool(_COGNITIVE.search(raw)),
        communication_needed=bool(_COMMUNICATION.search(raw)),
    )
