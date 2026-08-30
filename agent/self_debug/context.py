from dataclasses import dataclass


@dataclass
class CodeContext:
    step_n: int
    task: str
    num_rounds: int = 4
    time_out: int = 60