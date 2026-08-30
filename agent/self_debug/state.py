from typing import TypedDict, Optional
from .schemas import Script, ExecutionFeedback, StepFeedback

class CodeState(TypedDict, total=False):
    raw_script: str
    current_script: Script
    raw_feedback: str
    current_feedback: Optional[ExecutionFeedback | StepFeedback]
    round: int
    ExecutionVerdict: bool
    StepVerdict: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool