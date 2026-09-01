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

