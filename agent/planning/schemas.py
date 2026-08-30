from pydantic import BaseModel, Field


class Idea(BaseModel):
    """A single concrete research plan."""
    title: str = Field(description="One-line title.")
    summary: str = Field(description="2-3 sentences describing the plan.")
    method: str = Field(description="How you would test or implement it.")
    expected_outcome: str = Field(description="What you would measure or produce.")


class Critique(BaseModel):
    """Structured critique of a single idea."""
    feasibility: str = Field(description="Concrete feasibility issues, if any.")
    novelty: str = Field(description="Whether the idea is novel and why or why not.")
    constraints_respected: bool = Field(description="True if the idea respects the constraints.")
    fixes: list[str] = Field(description="Concrete fixes to make on the next revision.")


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