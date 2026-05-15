from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    role: Optional[str] = None
    seniority: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    personality_needed: bool = False
    cognitive_needed: bool = False
    communication_needed: bool = False
