import operator
from typing import Annotated, TypedDict, Optional, Literal
from .schemas import Script, ExecutionFeedback, StepFeedback


class CodeErrorRecord(TypedDict):
    round: int
    mode: Literal["code_error"]
    error_summary: str
    fix_suggestion: str


class GoalMissRecord(TypedDict):
    round: int
    mode: Literal["goal_miss"]
    unmet_requirements: str
    feedback: str


class CodeState(TypedDict, total=False):
    raw_script: str
    current_script: Script
    raw_feedback: str
    current_feedback: Optional[ExecutionFeedback | StepFeedback]
    error_history: Annotated[list[CodeErrorRecord | GoalMissRecord], operator.add]
    round: int
    execution_failed: bool
    step_fulfilled: bool
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    work_dir: str