from langgraph.graph import StateGraph, START, END
from .state import CodeState
from .context import CodeContext
from .nodes import code_maker, executor, execution_evaluator, step_evaluator, format_script, format_execution_feedback, format_step_feedback, extract_step_verdict, execution_gate, step_gate


graph = (
    StateGraph(CodeState, context_schema=CodeContext)
    .add_node("code_maker", code_maker)
    .add_node("format_script", format_script)
    .add_node("executor", executor)
    .add_node("execution_evaluator", execution_evaluator)
    .add_node("format_execution_feedback", format_execution_feedback)
    .add_node("step_evaluator", step_evaluator)
    .add_node("format_step_feedback", format_step_feedback)
    .add_node("extract_step_verdict", extract_step_verdict)
    .add_edge(START, "code_maker")
    .add_edge("code_maker", "format_script")
    .add_edge("format_script", "executor")
    .add_edge("executor", "execution_evaluator")
    .add_edge("format_execution_feedback", "code_maker")
    .add_edge("step_evaluator", "format_step_feedback")
    .add_edge("format_step_feedback", "extract_step_verdict")
    .add_conditional_edges(
        "execution_evaluator",
        execution_gate,
        {"format_execution_feedback": "format_execution_feedback", "step_evaluator":"step_evaluator", END: END},
    )
    .add_conditional_edges(
        "extract_step_verdict",
        step_gate,
        {"code_maker": "code_maker", END: END},
    )
    .compile()
)