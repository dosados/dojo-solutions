"""Скачивает google/translategemma-4b-it в task-1/submission_translate/weights/.

Запуск из корня task-1:

    conda activate ml-env
    python download_weights_translate.py

На Hugging Face нужно принять лицензию Gemma.
"""
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "google/translategemma-4b-it"
SUBMISSION_DIR = Path(__file__).resolve().parent / "submission_translate"
LOCAL_DIR = SUBMISSION_DIR / "weights"


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"weights downloaded to: {path}")
    print(
        f"pack for contest: cd {SUBMISSION_DIR} && "
        "zip -0 -r ../task-1-translategemma-submission.zip ."
    )


if __name__ == "__main__":
    main()
