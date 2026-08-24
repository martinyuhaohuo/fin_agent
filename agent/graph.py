from langgraph.graph import StateGraph, START, END
from .state import IdeaState
from .context import IdeaContext
from .nodes import idea_maker, format_idea, idea_hater, format_critique, route_after_format_critique


graph = (
    StateGraph(IdeaState, context_schema=IdeaContext)
    .add_node("idea_maker", idea_maker)
    .add_node("format_idea", format_idea)
    .add_node("idea_hater", idea_hater)
    .add_node("format_critique", format_critique)
    .add_edge(START, "idea_maker")
    .add_edge("idea_maker", "format_idea")
    .add_edge("format_idea", "idea_hater")
    .add_edge("idea_hater", "format_critique")
    .add_conditional_edges(
        "format_critique",
        route_after_format_critique,
        {"idea_maker": "idea_maker", END: END},
    )
    .compile()
)