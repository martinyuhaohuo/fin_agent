from typing import TypedDict, Optional
from .schemas import Critique, Idea

class IdeaState(TypedDict, total=False):
    raw_idea: str
    current_idea: Idea
    raw_critique: str
    current_critique: Optional[Critique]
    round: int