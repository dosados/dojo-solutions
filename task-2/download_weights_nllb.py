"""Скачивает facebook/nllb-200-3.3B в task-2/submission_nllb/weights/.

Запуск из корня task-2 (локально, один раз перед Docker / обучением):

    conda activate ml-env
    python download_weights_nllb.py

Если раньше качали 1.3B/600M — удалите submission_nllb/weights/ и скачайте заново.

Модель ~6.5–7 GB (bf16 на диске); QLoRA train — 4-bit; merge для zip — bf16.
"""
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "facebook/nllb-200-3.3B"
SUBMISSION_DIR = Path(__file__).resolve().parent / "submission_nllb"
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
        "zip -r ../task-2-nllb-submission.zip ."
    )


if __name__ == "__main__":
    main()
