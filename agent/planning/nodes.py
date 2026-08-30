import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END
from .state import IdeaState
from .schemas import Idea, Critique
from .prompts import MAKER_SYSTEM, HATER_SYSTEM
from .context import IdeaContext


load_dotenv()
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except:
    print("Google API key not found")


PROPOSE_MODEL  = "gemini-3.1-flash-lite-preview"   # cheap, fast → idea_maker
CRITIQUE_MODEL = "gemini-3.1-pro-preview"            # careful, sharp → idea_hater
FORMAT_MODEL   = "gemini-3.1-flash-lite-preview"   # cheap, deterministic → formatter


proposer = ChatGoogleGenerativeAI(
    model=PROPOSE_MODEL,
    temperature=1.0,
    thinking_level="low",
    google_api_key=GOOGLE_API_KEY,
)


critic = ChatGoogleGenerativeAI(
    model=CRITIQUE_MODEL,
    temperature=0.4,
    thinking_level="high",
    google_api_key=GOOGLE_API_KEY,
)


formatter_llm = ChatGoogleGenerativeAI(
    model=FORMAT_MODEL,
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
)


def make_formatter(schema, input_field: str, output_field: str, tag: str):
    """Build a node that converts state[input_field] (str) into a `schema` instance
    and writes it to state[output_field]. Tags the LLM call so the logger
    can attribute it."""
    structured = formatter_llm.with_structured_output(schema)
    sys = SystemMessage(
        f"You are a formatter. Convert the user's text into a {schema.__name__} "
        f"object. Preserve all substantive content. Do not invent new facts. "
        f"If a field is not explicitly stated, infer it conservatively from context."
    )

    def formatter(state, runtime):
        raw = state[input_field]
        obj = structured.invoke([sys, HumanMessage(raw)], config={"tags": [tag]})
        return {output_field: obj}
    return formatter


def idea_maker(state: IdeaState, runtime: Runtime[IdeaContext]) -> IdeaState:
    ctx = runtime.context
    round_n = state.get("round", 0) + 1
    critique = state.get("current_critique")

    if critique is None:
        user = (
            f"Research topic:\n{ctx.research_topic}\n\n"
            f"Data description:\n{ctx.data_description}\n\n"
            f"Constraints:\n{ctx.constraints}\n\n"
            f"Propose a single idea."
        )
    else:
        prev = state["current_idea"]
        user = (
            f"Research topic:\n{ctx.research_topic}\n\n"
            f"Data description:\n{ctx.data_description}\n\n"
            f"Constraints:\n{ctx.constraints}\n\n"
            f"Your previous idea:\n{prev.model_dump_json(indent=2)}\n\n"
            f"idea_hater's critique:\n{critique.model_dump_json(indent=2)}\n\n"
            f"Revise. Address every fix the critic listed. Return one revised idea."
        )

    msg = proposer.invoke(
        [SystemMessage(MAKER_SYSTEM), HumanMessage(user)],
        config={"tags": ["idea_maker"]},
    )
    return {"raw_idea": msg.text, "round": round_n}


def idea_hater(state: IdeaState, runtime: Runtime[IdeaContext]) -> IdeaState:
    ctx = runtime.context
    idea = state["current_idea"]
    user = (
        f"Research topic:\n{ctx.research_topic}\n\n"
        f"Data description:\n{ctx.data_description}\n\n"
        f"Constraints:\n{ctx.constraints}\n\n"
        f"Idea under review:\n{idea.model_dump_json(indent=2)}\n\n"
        f"Critique it."
    )
    msg = critic.invoke(
        [SystemMessage(HATER_SYSTEM), HumanMessage(user)],
        config={"tags": ["idea_hater"]},
    )
    return {"raw_critique": msg.text}


format_idea = make_formatter(
    schema=Idea,
    input_field="raw_idea",
    output_field="current_idea",
    tag="format_idea",
)


format_critique = make_formatter(
    schema=Critique,
    input_field="raw_critique",
    output_field="current_critique",
    tag="format_critique",
)


def route_after_format_critique(state: IdeaState, runtime: Runtime[IdeaContext]) -> str:
    if state["round"] < runtime.context.num_rounds:
        return "idea_maker"
    return END