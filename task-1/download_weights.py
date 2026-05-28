"""Скачивает gemma-4-E2B-it в task-1/submission/weights/.

Запуск из корня task-1 (локально, один раз перед сборкой Docker):

    conda run -n ml-env python download_weights.py

Содержимое submission/ — то, что упаковывается в zip для контеста.
"""
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "google/gemma-4-E2B-it"
SUBMISSION_DIR = Path(__file__).resolve().parent / "submission"
LOCAL_DIR = SUBMISSION_DIR / "weights"


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"weights downloaded to: {path}")
    print(f"pack for contest: cd {SUBMISSION_DIR} && zip -r ../task-1-submission.zip .")


if __name__ == "__main__":
    main()
