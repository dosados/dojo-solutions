#!/usr/bin/env python3
"""QLoRA для NLLB-200 (seq2seq): stage-1 mono → stage-2 parallel RU→AB.

Те же данные, что TranslateGemma (`data/processed/mono`, `data/processed/parallel`).

Примеры (из task-2/):

    conda activate ml-env
    pip install -r requirements-train.txt
    python download_weights_nllb.py

    ./scripts/prepare_training_data.sh   # или prepare_mono + prepare_parallel

    python -u scripts/train_qlora_nllb.py --config configs/nllb/clm_pilot.yaml
    python -u scripts/train_qlora_nllb.py --config configs/nllb/sft_pilot.yaml

    python -u scripts/train_qlora_nllb.py --config configs/nllb/clm_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, TrainerCallback

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from train.checkpoint import (  # noqa: E402
    LowVRAMCheckpointCallback,
    resolve_resume_checkpoint,
)
from train.data import load_mono_pool, load_parallel_pool  # noqa: E402
from train.kbit import prepare_model_for_kbit_training_low_vram  # noqa: E402
from train.load_model import load_quantized_base, load_tokenizer  # noqa: E402
from train.nllb_peft import default_lora_target_modules  # noqa: E402
from train.paths import resolve_weights_dir  # noqa: E402
from train.tokenize_nllb import (  # noqa: E402
    tokenize_nllb_mono_batch,
    tokenize_nllb_mono_row,
    tokenize_nllb_sft_batch,
    tokenize_nllb_sft_row,
)
from train.trainer_nllb import LowVRAMSeq2SeqTrainer  # noqa: E402


class _LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            parts = [f"step {state.global_step}"]
            if "loss" in logs:
                parts.append(f"loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                parts.append(f"eval_loss={logs['eval_loss']:.4f}")
            print("  " + "  ".join(parts), flush=True)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(root: Path, p: str | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def _attach_lora(model, cfg: dict, *, adapter_init: Path | None):
    lora_config = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="SEQ_2_SEQ_LM",
        target_modules=cfg.get("lora_target_modules") or default_lora_target_modules(),
    )

    if adapter_init is not None:
        if not adapter_init.is_dir():
            raise FileNotFoundError(f"adapter_init not found: {adapter_init}")
        print(f"  continue LoRA from {adapter_init}", flush=True)
        return PeftModel.from_pretrained(model, str(adapter_init), is_trainable=True)

    return get_peft_model(model, lora_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="task-2 QLoRA for NLLB-200")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint dir или 'latest'",
    )
    parser.add_argument("--no-auto-resume", action="store_true")
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    root = _TASK_ROOT
    stage = str(cfg.get("stage", "sft")).lower()
    if stage not in ("clm", "sft"):
        raise ValueError(f"stage must be clm or sft, got {stage!r}")

    run_name = cfg["run_name"]
    out_root = _resolve(root, cfg["output_dir"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_weights_dir("nllb", cfg.get("model_dir"))
    adapter_init = _resolve(root, cfg.get("adapter_init"))

    with (out_root / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"stage={stage}  model=nllb  run={run_name}", flush=True)
    print(f"output: {out_root}", flush=True)

    max_len = int(cfg.get("max_seq_length", 512))
    seed = int(cfg.get("seed", 42))

    if stage == "clm":
        train_path = _resolve(root, cfg["train_file"])
        lang_weights = cfg.get("mono_lang_weights")
        train_rows = load_mono_pool(
            train_path,
            max_samples=cfg.get("max_train_samples"),
            seed=seed,
            lang_weights=lang_weights,
            min_chars=int(cfg.get("min_chars", 20)),
            max_chars=int(cfg.get("max_chars", 4096)),
        )
        val_rows: list = []
    else:
        train_path = _resolve(root, cfg["train_file"])
        train_rows = load_parallel_pool(
            train_path,
            max_samples=cfg.get("max_train_samples"),
            seed=seed,
            min_chars=int(cfg.get("min_src_chars", 5)),
            max_chars=int(cfg.get("max_src_chars", 4096)),
            max_sents=int(cfg.get("max_sents", 20)),
        )
        val_rows = []
        val_path = _resolve(root, cfg.get("val_file"))
        max_val = int(cfg.get("max_val_samples", 500))
        if val_path and val_path.is_file() and max_val > 0:
            val_rows = load_parallel_pool(
                val_path,
                max_samples=max_val,
                seed=seed + 99,
                min_chars=int(cfg.get("min_src_chars", 5)),
                max_chars=int(cfg.get("max_src_chars", 4096)),
                max_sents=int(cfg.get("max_sents", 20)),
            )

    print(f"  train rows: {len(train_rows)}", flush=True)
    if val_rows:
        print(f"  val rows:   {len(val_rows)}", flush=True)

    try:
        model = load_quantized_base("nllb", model_dir, load_cfg=cfg)
    except Exception as e:
        if cfg.get("attn_implementation"):
            print(
                f"  warning: attn load failed ({e}); retry without attn_implementation",
                flush=True,
            )
            cfg2 = dict(cfg)
            cfg2.pop("attn_implementation", None)
            model = load_quantized_base("nllb", model_dir, load_cfg=cfg2)
        else:
            raise

    model = prepare_model_for_kbit_training_low_vram(
        model,
        use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        cast_norms_to_fp32=bool(cfg.get("cast_norms_to_fp32", True)),
    )
    tokenizer = load_tokenizer("nllb", model_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = _attach_lora(model, cfg, adapter_init=adapter_init)

    tok_batched = bool(cfg.get("tokenize_batched", True))
    tok_batch_size = int(cfg.get("tokenize_batch_size", 256))

    if stage == "clm":
        remove_cols = ["text", "lang"]
        if tok_batched:

            def _map_fn(batch):
                return tokenize_nllb_mono_batch(
                    batch, tokenizer, max_length=max_len
                )
        else:

            def _map_fn(ex):
                return tokenize_nllb_mono_row(ex, tokenizer, max_length=max_len)
    else:
        remove_cols = ["src", "ref"]
        if tok_batched:

            def _map_fn(batch):
                return tokenize_nllb_sft_batch(batch, tokenizer, max_length=max_len)
        else:

            def _map_fn(ex):
                return tokenize_nllb_sft_row(ex, tokenizer, max_length=max_len)

    map_kwargs: dict = {"desc": "tokenize train"}
    if tok_batched:
        map_kwargs.update({"batched": True, "batch_size": tok_batch_size})

    train_ds = Dataset.from_list(train_rows).map(
        _map_fn,
        remove_columns=remove_cols,
        **map_kwargs,
    )

    eval_ds = None
    if val_rows:
        val_map_kwargs: dict = {"desc": "tokenize val"}
        if tok_batched:
            val_map_kwargs.update({"batched": True, "batch_size": tok_batch_size})
        eval_ds = Dataset.from_list(val_rows).map(
            _map_fn,
            remove_columns=remove_cols,
            **val_map_kwargs,
        )

    save_only = bool(cfg.get("save_only_model", True))
    auto_resume = bool(cfg.get("auto_resume", True)) and not args.no_auto_resume

    max_steps = cfg.get("max_steps")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_root),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(max_steps) if max_steps is not None else -1,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_steps=int(cfg.get("warmup_steps", 50)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 200)) if eval_ds else None,
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        save_only_model=save_only,
        predict_with_generate=False,
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False,
        label_smoothing_factor=float(cfg.get("label_smoothing_factor", 0.0)),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = LowVRAMSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[_LogCallback(), LowVRAMCheckpointCallback()],
    )

    resume_path = None
    if args.resume is not None:
        if args.resume.lower() == "latest":
            resume_path = resolve_resume_checkpoint(out_root, explicit=None, auto_resume=True)
        else:
            resume_path = resolve_resume_checkpoint(
                out_root, explicit=Path(args.resume), auto_resume=False
            )
    elif auto_resume:
        resume_path = resolve_resume_checkpoint(out_root, explicit=None, auto_resume=True)

    if resume_path is not None:
        print("training (resume) …", flush=True)
        trainer.train(resume_from_checkpoint=str(resume_path))
    else:
        print("training …", flush=True)
        trainer.train()

    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    meta = {
        "stage": stage,
        "model_type": "nllb",
        "run_name": run_name,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "adapter_init": str(adapter_init) if adapter_init else None,
        "config": str(args.config.resolve()),
    }
    with (final_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"done. adapter: {final_dir}", flush=True)
    if stage == "clm":
        print(
            "next:\n"
            "  python -u scripts/train_qlora_nllb.py --config configs/nllb/sft_pilot.yaml",
            flush=True,
        )
    else:
        print(
            "next:\n"
            f"  python -u scripts/validate_lora.py --model nllb "
            f"--adapter {final_dir}\n"
            f"  python -u scripts/merge_lora.py --model nllb --adapter {final_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
