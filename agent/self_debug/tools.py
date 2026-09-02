import shutil
from pathlib import Path

def prepare_work_dir(work_dir: str, task_name: str) -> str:
    work_dir = Path(work_dir)
    lab_dir = work_dir / task_name
    if lab_dir.exists():
        shutil.rmtree(lab_dir)
    codebase_dir = lab_dir / "codebase"
    data_dir = lab_dir / "data"
    logs_dir = lab_dir / "logs"

    for directory in [codebase_dir, data_dir, logs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return lab_dir


def snapshot_files(directory: Path) -> set[str]:
    file_set = set()
    for path in directory.rglob("*"):
        if path.is_file():
            file_set.add(str(path.relative_to(directory)))
    return file_set


def print_error_history(history: list[dict]) -> str:
    sections = []
    for record in history:
        if record["mode"] == "code_error":
            section = (
                f"Round {record['round']} — code error\n"
                f"Error summary:\n{record['error_summary']}\n"
                f"Suggested fix:\n{record['fix_suggestion']}"
            )
        elif record["mode"] == "goal_miss":
            section = (
                f"Round {record['round']} — goal miss\n"
                f"Unmet requirements:\n{record['unmet_requirements']}\n"
                f"Reviewer feedback:\n{record['feedback']}"
            )
        sections.append(section)
    return "\n\n".join(sections)
