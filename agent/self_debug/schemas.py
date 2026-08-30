from pydantic import BaseModel, Field


class Script(BaseModel):
    """A Python script with explanation."""
    code: str = Field(description="An executable Python script.")
    explanation: str = Field(description="5-10 sentences explaning how the script work.")


class ExecutionFeedback(BaseModel):
    """A feedback on why the last script results in a runtime error."""
    issue: str = Field(description="3-5 sentences describing why the script does not work.")
    fix_plan: str = Field(description="5-10 sentences describing how to address the issue found.")


class StepFeedback(BaseModel):
    """An evaluation on whether the last script achieve the goal of the step."""
    step_verdict: bool = Field(description="False if the script achieves the goal of the step, True if it does not.")
    issue: str = Field(description="3-5 sentences describing why the script does not achieve the goal, in case it does not.")
    fix_plan: str = Field(description="5-10 sentences describing how to address the issue found, in case the goal is not achieved.")


def _type_name(ann) -> str:
    # str(int) -> "<class 'int'>", str(list[str]) -> "list[str]" — clean both.
    s = str(ann).replace("typing.", "")
    if s.startswith("<class '") and s.endswith("'>"):
        return s[len("<class '"):-len("'>")]
    return s


def schema_field_brief(schema) -> str:
    """Render a Pydantic schema as a bullet list of fields + descriptions.

    Used inside *generator* prompts so the writer knows what to cover, while
    still producing free-form prose (no JSON)."""
    lines = []
    for name, field in schema.model_fields.items():
        t = _type_name(field.annotation)
        desc = field.description or ""
        lines.append(f"- {name} ({t}): {desc}")
    return "\n".join(lines)