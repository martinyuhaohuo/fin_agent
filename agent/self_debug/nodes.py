import os
import subprocess
import sys
import json
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
from .tools import snapshot_files, print_error_history

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
    step_pass = state.get("step_fulfilled")
    error_history = state.get("error_history", [])

    base_prompt = (
        f"Task:\n{ctx.task}\n\n"
        f"Constraints:\n{ctx.constraints}\n\n"
    )

    if not error_history:
        user = (
            base_prompt +
            f"Propose a single Python script with explanation."
        )
        
    else:
        prev = state["current_script"]
        base_prompt = (
            base_prompt + f"Your previous script:\n{prev.model_dump_json(indent=2)}\n\n"
            )
        
        if len(error_history) >= 2:
            previous_errors = print_error_history(error_history[:-1])
            previous_errors = "Earlier failure history:\n" + previous_errors + "\n\n"
        else:
            previous_errors = ""

        if execution_error is True:
            user = (
                base_prompt + 
                f"execution_evaluator found that your code results in the execution error\n\n" +
                f"execution_evaluator's feedback:\n{critique.model_dump_json(indent=2)}\n\n" +
                previous_errors +
                f"Revise. Address every fix listed. Return one revised idea."
            )
        elif step_pass is False:
            prev = state["current_script"]
            user = (
                base_prompt + 
                f"step_evaluator found that your code does not achieve the goal of this step\n\n" +
                f"step_evaluator's feedback:\n{critique.model_dump_json(indent=2)}\n\n" +
                previous_errors +
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
    work_dir = Path(state["work_dir"])

    codebase = work_dir / "codebase"
    script_path = codebase / f"step_{step_n}.py"
    script_path.write_text(script.code, encoding="utf-8")
    
    data_dir = work_dir / "data"
    before = snapshot_files(data_dir)

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

    after = snapshot_files(data_dir)
    created_files = sorted(after - before)
    manifest = {
        "step": step_n,
        "created_files": created_files,
    }
    manifest_path = work_dir / "logs" / f"step_{step_n}_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8",)

    return {"stdout": stdout, "stderr": stderr, "returncode": returncode, "timed_out": timed_out}


def execution_evaluator(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    ctx = runtime.context
    step_n = ctx.step_n
    round = state["round"]
    timed_out = state["timed_out"]
    returncode = state["returncode"]
    if returncode != 0 or timed_out == True:
        execution_failed = True
        codebase = Path(state["work_dir"] + "/codebase")
        script_path = codebase / f"step_{step_n}.py"
        failed_script_name = codebase / f"step_{step_n}_failed_{round}.py"
        script_path.rename(failed_script_name)
    else:
        execution_failed = False
        return {"execution_failed": execution_failed}

    script = state["current_script"]
    error = state["stderr"]
    if timed_out:
        user = (
            f"Task:\n{ctx.task}\n\n"
            f"Constraints:\n{ctx.constraints}\n\n"
            f"Script under review:\n{script.model_dump_json(indent=2)}\n\n"
            f"Execution time exceeds then pre-defined limit of {ctx.time_out}seconds\n\n"
            f"Find the issue and suggest solution."
        )
    else:
        user = (
            f"Task:\n{ctx.task}\n\n"
            f"Constraints:\n{ctx.constraints}\n\n"
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
        f"Constraints:\n{ctx.constraints}\n\n"
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


def error_history(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    current_feedback = state["current_feedback"]
    if isinstance(current_feedback, ExecutionFeedback):
        record = {
            "round": state["round"],
            "mode": "code_error",
            "error_summary": current_feedback.error_summary,
            "fix_suggestion": current_feedback.fix_suggestion,
        }
    elif isinstance(current_feedback, StepFeedback):
        record = {
            "round": state["round"],
            "mode": "goal_miss",
            "unmet_requirements": current_feedback.unmet_requirements,
            "feedback": current_feedback.feedback,
        }
    return {"error_history": [record]}


def extract_step_verdict(state: CodeState, runtime: Runtime[CodeContext]) -> CodeState:
    step_feedback = state["current_feedback"]
    round = state["round"]
    step_n = runtime.context.step_n
    step_fulfilled = step_feedback.step_fulfilled
    if step_fulfilled is False: 
        codebase = Path(state["work_dir"] + "/codebase")
        script_path = codebase / f"step_{step_n}.py"
        failed_script_name = codebase / f"step_{step_n}_failed_{round}.py"
        script_path.rename(failed_script_name)
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
    elif state["round"] >= runtime.context.num_rounds:
                return END
    else:
        return "error_history"
