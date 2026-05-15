"""
SHL conversational assessment recommender API.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.guardrails.checks import (
    is_legal_request,
    is_off_topic,
    is_prompt_injection,
)
from app.retrieval.hybrid_search import hybrid_search
from app.state.extract import extract_conversation_state
from app.state.models import ConversationState
from app.utils.gemini_client import generate_response

logger = logging.getLogger(__name__)

MAX_RECOMMENDATIONS = 10
RETRIEVAL_TOP_K = 10

_COMPARE_PATTERN = re.compile(
    r"\b(compare|comparison|difference|differences|vs\.?|versus|"
    r"which\s+(one|assessment|test)\s+(is\s+)?better)\b",
    re.IGNORECASE,
)
_END_PATTERN = re.compile(
    r"\b(bye|goodbye|thanks|thank you|that'?s all|done|end conversation)\b",
    re.IGNORECASE,
)
_VAGUE_ONLY = re.compile(
    r"^(hi|hello|hey|help|help me|yo|hiya|good morning|good afternoon)[!.?]*$",
    re.IGNORECASE,
)


# --- API models ---


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]
    end_of_conversation: bool = False


# --- helpers ---


def _latest_user_message(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower().strip()
        content = msg.get("content")
        if role == "user" and isinstance(content, str):
            return content.strip()
    return ""


def _format_test_type(test_type: Any) -> str:
    if isinstance(test_type, dict):
        categories = test_type.get("categories") or []
        if categories:
            return "; ".join(str(c) for c in categories)
        codes = test_type.get("codes") or []
        if codes:
            return ", ".join(str(c) for c in codes)
    if test_type:
        return str(test_type)
    return ""


def _to_recommendations(
    assessments: list[dict[str, Any]],
) -> list[RecommendationItem]:
    items: list[RecommendationItem] = []
    for row in assessments[:MAX_RECOMMENDATIONS]:
        items.append(
            RecommendationItem(
                name=str(row.get("assessment_name", "")),
                url=str(row.get("assessment_url", "")),
                test_type=_format_test_type(row.get("test_type")),
            )
        )
    return items


def _is_comparison_query(text: str) -> bool:
    return bool(_COMPARE_PATTERN.search(text))


def _is_end_of_conversation(text: str) -> bool:
    return bool(_END_PATTERN.search(text))


def _is_vague_query(text: str, state: ConversationState) -> bool:
    t = text.strip()
    if not t:
        return True
    if _VAGUE_ONLY.match(t):
        return True
    if len(t) < 20 and not any(
        [
            state.role,
            state.seniority,
            state.skills,
            state.personality_needed,
            state.cognitive_needed,
            state.communication_needed,
        ]
    ):
        return True
    generic = re.match(
        r"^(please\s+)?(recommend|suggest|find|show)(\s+me)?\s+"
        r"(an?\s+)?(assessment|assessments|test|tests)[!.?]*$",
        t,
        re.IGNORECASE,
    )
    if generic and not state.role and not state.skills:
        return True
    return False


def _build_retrieval_query(user_message: str, state: ConversationState) -> str:
    parts = [user_message]
    if state.role:
        parts.append(state.role)
    if state.seniority:
        parts.append(state.seniority)
    if state.skills:
        parts.extend(state.skills)
    if state.personality_needed:
        parts.append("personality assessment")
    if state.cognitive_needed:
        parts.append("cognitive aptitude assessment")
    if state.communication_needed:
        parts.append("communication skills assessment")
    return " ".join(parts)


def _format_catalog_for_prompt(assessments: list[dict[str, Any]]) -> str:
    if not assessments:
        return "(No assessments retrieved.)"
    blocks: list[str] = []
    for i, row in enumerate(assessments, start=1):
        blocks.append(
            f"{i}. name={row.get('assessment_name', '')}\n"
            f"   url={row.get('assessment_url', '')}\n"
            f"   test_type={_format_test_type(row.get('test_type'))}\n"
            f"   description={row.get('description', '')}"
        )
    return "\n".join(blocks)


def _build_llm_prompt(
    user_message: str,
    state: ConversationState,
    assessments: list[dict[str, Any]],
    *,
    comparison: bool,
) -> str:
    catalog = _format_catalog_for_prompt(assessments)
    state_summary = (
        f"role={state.role or 'unknown'}; "
        f"seniority={state.seniority or 'unknown'}; "
        f"skills={', '.join(state.skills) if state.skills else 'none'}; "
        f"personality_needed={state.personality_needed}; "
        f"cognitive_needed={state.cognitive_needed}; "
        f"communication_needed={state.communication_needed}"
    )

    task = (
        "Compare ONLY the assessments in RETRIEVED CATALOG. "
        "Highlight differences in test_type and description using catalog facts only."
        if comparison
        else "Recommend suitable assessments from RETRIEVED CATALOG and explain why they fit."
    )

    return f"""You are an SHL assessment recommendation assistant.

STRICT RULES:
- Use ONLY assessments listed in RETRIEVED CATALOG below.
- NEVER invent, rename, or add assessments not in that list.
- When naming an assessment, use the exact name from the catalog.
- If nothing fits well, say so and suggest what details the user should provide.

EXTRACTED STATE: {state_summary}

USER MESSAGE: {user_message}

TASK: {task}

RETRIEVED CATALOG:
{catalog}

Write a concise, conversational reply for the user. Do not output JSON."""


def _fallback_reply(
    assessments: list[dict[str, Any]],
    *,
    comparison: bool,
) -> str:
    if not assessments:
        return (
            "I could not find matching SHL assessments in the catalog for your request. "
            "Please share the role, seniority, and skills you are hiring for."
        )
    if comparison:
        lines = ["Here is a comparison based on the SHL catalog entries I found:\n"]
    else:
        lines = ["Here are SHL assessments from the catalog that may fit your needs:\n"]
    for row in assessments:
        name = row.get("assessment_name", "")
        url = row.get("assessment_url", "")
        tt = _format_test_type(row.get("test_type"))
        desc = (row.get("description") or "")[:200]
        lines.append(f"- {name} ({tt})\n  {url}\n  {desc}")
    return "\n".join(lines)


def _clarification_reply() -> str:
    return (
        "To recommend SHL assessments, I need a bit more detail. "
        "What role are you hiring for, what seniority level, and which technical or "
        "behavioral skills should the assessment cover (e.g. Java, communication, "
        "personality, cognitive aptitude)?"
    )


def _guardrail_reply() -> str:
    return (
        "I can only help with SHL talent assessments and hiring-related recommendations. "
        "Please ask about roles, skills, or which assessments to use for your candidates."
    )


def _process_chat(messages: list[dict[str, Any]]) -> ChatResponse:
    user_message = _latest_user_message(messages)
    if not user_message:
        return ChatResponse(
            reply="Please send a user message describing the role and skills you are hiring for.",
            recommendations=[],
            end_of_conversation=False,
        )

    if (
        is_prompt_injection(user_message)
        or is_legal_request(user_message)
        or is_off_topic(user_message)
    ):
        return ChatResponse(
            reply=_guardrail_reply(),
            recommendations=[],
            end_of_conversation=False,
        )

    state = extract_conversation_state(messages)
    end_of_conversation = _is_end_of_conversation(user_message)

    if _is_vague_query(user_message, state):
        return ChatResponse(
            reply=_clarification_reply(),
            recommendations=[],
            end_of_conversation=end_of_conversation,
        )

    comparison = _is_comparison_query(user_message)
    query = _build_retrieval_query(user_message, state)
    retrieved = hybrid_search(query, top_k=RETRIEVAL_TOP_K)
    recommendations = _to_recommendations(retrieved)

    if not retrieved:
        return ChatResponse(
            reply=(
                "I could not find matching SHL assessments in the catalog. "
                "Please refine the role, skills, or assessment type you need."
            ),
            recommendations=[],
            end_of_conversation=end_of_conversation,
        )

    prompt = _build_llm_prompt(
        user_message,
        state,
        retrieved,
        comparison=comparison,
    )
    llm_text = generate_response(prompt)
    if not llm_text or llm_text.startswith("LLM Error:"):
        reply = _fallback_reply(retrieved, comparison=comparison)
    else:
        reply = llm_text.strip()

    return ChatResponse(
        reply=reply,
        recommendations=recommendations,
        end_of_conversation=end_of_conversation,
    )


# --- FastAPI app ---

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational SHL assessment recommendations grounded in catalog retrieval.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    try:
        return _process_chat(body.messages)
    except Exception:
        logger.exception("Chat request failed")
        return ChatResponse(
            reply="Something went wrong while processing your request. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
