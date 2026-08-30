from dataclasses import dataclass


@dataclass
class IdeaContext:
    research_topic: str
    data_description: str
    num_rounds: int = 2
    constraints: str = "No external data sources. Ideas must be testable. Only use Python for programming."