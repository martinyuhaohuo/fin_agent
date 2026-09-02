from dataclasses import dataclass


@dataclass
class CodeContext:
    step_n: int
    task: str
    num_rounds: int = 4
    time_out: int = 60
    constraints: str = """
    If the task outputs data files, they must be saved to the DATA_DIR shown below:
    from pathlib import Path
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    """