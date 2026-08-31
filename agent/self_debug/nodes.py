import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, START, END

from .state import CodeState
from .schemas import Script, ExecutionFeedback, StepFeedback
from .context import CodeContext

from .prompts import ENGINEER_SYSTEM, EXECUTION_EVALUATOR_SYSTEM, STEP_EVALUATOR_SYSTEM



load_dotenv()
try:
    GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
except:
    print("Google API key not found")


ENGINEER_MODEL  = "gemini-3.1-flash-lite-preview"
FORMAT_MODEL = "gemini-3.1-flash-lite-preview"
EXECUTION_EVALUATOR_MODEL = "gemini-3.1-flash-lite-preview"
STEP_EVALUATOR = "gemini-3.1-pro-preview"


engineer = ChatGoogleGenerativeAI(
    model=ENGINEER_MODEL,
    temperature=0.4,
    thinking_level="low",
    google_api_key=GOOGLE_API_KEY,
)


execution_evaluator_llm = ChatGoogleGenerativeAI(
    model=EXECUTION_EVALUATOR_MODEL,
    temperature=0.4,
    thinking_level="low",
    google_api_key=GOOGLE_API_KEY,
)


step_evaluator_llm = ChatGoogleGenerativeAI(
    model=STEP_EVALUATOR,
    temperature=0.4,
    thinking_level="high",
    google_api_key=GOOGLE_API_KEY,
)


formatter_llm = ChatGoogleGenerativeAI(
    model=FORMAT_MODEL,
    temperature=0.0,
    google_api_key=GOOGLE_API_KEY,
)


def code_maker(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    ctx = runtime.context
    round_n = state.get("round", 0) + 1
    critique = state.get("current_feedback")
    execution_error = state.get("execution_failed")
    step_error = state.get("step_failed")

    if critique is None:
        user = (
            f"Task:\n{ctx.task}\n\n"
            f"Propose a single Python script with explanation."
        )
    else:
        if execution_error:
            prev = state["current_script"]
            user = (
                f"Task:\n{ctx.task}\n\n"
                f"Your previous script:\n{prev.model_dump_json(indent=2)}\n\n"
                f"execution_evaluator found that your code results in the execution error\n\n"
                f"execution_evaluator's feedback:\n{critique.model_dump_json(indent=2)}\n\n"
                f"Revise. Address every fix listed. Return one revised idea."
            )
        if step_error:
            prev = state["current_script"]
            user = (
                f"Task:\n{ctx.task}\n\n"
                f"Your previous script:\n{prev.model_dump_json(indent=2)}\n\n"
                f"step_evaluator found that your code does not achieve the goal of this step\n\n"
                f"step_evaluator's feedback:\n{critique.model_dump_json(indent=2)}\n\n"
                f"Revise. Address every fix listed. Return one revised idea."
            )

    msg = engineer.invoke(
        [SystemMessage(ENGINEER_SYSTEM), HumanMessage(user)],
        config={"tags": ["code_maker"]},
    )
    return {"raw_script": msg.text, "round": round_n}


def executor(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    ctx = runtime.context
    script = state["current_script"]
    step_n = ctx.step_n
    codebase = Path("codebase")
    codebase.mkdir(exist_ok=True)
    script_path = codebase / f"step_{step_n}.py"
    script_path.write_text(script.code, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout = ctx.time_out
        )

        stdout = result.stdout
        stderr = result.stderr
        returncode = result.returncode
        timed_out = False

    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        stderr = e.stderr or ""
        returncode = None
        timed_out = True

    

    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "timed_out": timed_out}


def execution_evaluator(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    ctx = runtime.context
    timed_out = state["timed_out"]
    returncode = state["returncode"]
    if returncode != 0 or timed_out == True:
        execution_failed = True
    else:
        execution_failed = False
        return {"execution_failed": execution_failed}

    script = state["current_script"]
    error = state["stderr"]
    if timed_out:
        user = (
            f"Task:\n{ctx.task}\n\n"
            f"Script under review:\n{script.model_dump_json(indent=2)}\n\n"
            f"Execution time exceeds then pre-defined limit of {ctx.time_out}seconds\n\n"
            f"Find the issue and suggest solution."
        )
    else:
        user = (
            f"Task:\n{ctx.task}\n\n"
            f"Script under review:\n{script.model_dump_json(indent=2)}\n\n"
            f"Execution error message:\n{error}\n\n"
            f"Find the issue and suggest solution."
        )
    msg = execution_evaluator_llm.invoke(
        [SystemMessage(EXECUTION_EVALUATOR_SYSTEM), HumanMessage(user)],
        config={"tags": ["execution_evaluator"]},
    )
    return {"raw_feedback": msg.text, "execution_failed": execution_failed}


def step_evaluator(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    ctx = runtime.context
    script = state["current_script"]
    output = state["stdout"]
    user = (
        f"Task:\n{ctx.task}\n\n"
        f"Script under review:\n{script.model_dump_json(indent=2)}\n\n"
        f"The output of the script:\n{output}\n\n"
        f"The script is executed successfully, you need to check whether the script achieves the goal of the step."
    )
    msg = step_evaluator_llm.invoke(
        [SystemMessage(STEP_EVALUATOR_SYSTEM), HumanMessage(user)],
        config={"tags": ["step_evaluator"]},
    )
    return {"raw_feedback": msg.text}


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


format_script = make_formatter(
    schema=Script,
    input_field="raw_script",
    output_field="current_script",
    tag="format_script",
)


format_execution_feedback = make_formatter(
    schema=ExecutionFeedback,
    input_field="raw_feedback",
    output_field="current_feedback",
    tag="format_execution_feedback",
)


format_step_feedback = make_formatter(
    schema=StepFeedback,
    input_field="raw_feedback",
    output_field="current_feedback",
    tag="format_step_feedback",
)


def extract_step_verdict(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    step_feedback = state["current_feedback"]
    step_fulfilled = step_feedback.step_fulfilled
    return {"step_fulfilled": step_fulfilled}


def execution_gate(state: CodeState, runtime: Runtime[CodeContext]) -> str:
    if state["round"] >= runtime.context.num_rounds:
        return END
    elif state["execution_failed"]:
        return "format_execution_feedback"
    else:
        return "step_evaluator"


def step_gate(state: CodeState, runtime: Runtime[CodeContext]) -> str:
    if state["step_fulfilled"]:
        return END
    else:
        return "codemaker"
