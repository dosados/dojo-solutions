"""Скачивает google/translategemma-4b-it в task-2/submission_translate/weights/.

Запуск из корня task-2 (локально, один раз перед сборкой Docker):

    conda activate ml-env
    python download_weights_translate.py

На Hugging Face нужно принять лицензию Gemma для доступа к модели.
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
    print(f"pack for contest: cd {SUBMISSION_DIR} && zip -r ../task-2-translategemma-submission.zip .")


if __name__ == "__main__":
    main()
