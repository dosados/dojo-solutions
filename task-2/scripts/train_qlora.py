#!/usr/bin/env python3
"""QLoRA для task-2: stage-1 CLM (mono ru+ab) → stage-2 SFT (parallel ru→ab).

Поддерживает model_type: gemma | translate

Примеры (из task-2/):

    conda activate ml-env
    pip install -r requirements-train.txt

    # Данные
    python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru
    python -u scripts/prepare_parallel.py

    # Stage 1 — CLM
    python -u scripts/train_qlora.py --config configs/translate/clm_pilot.yaml
    python -u scripts/train_qlora.py --config configs/gemma/clm_pilot.yaml

    # Stage 2 — SFT (adapter_init = stage-1 final)
    python -u scripts/train_qlora.py --config configs/translate/sft_pilot.yaml
    python -u scripts/train_qlora.py --config configs/gemma/sft_pilot.yaml

    # Smoke (10 шагов)
    python -u scripts/train_qlora.py --config configs/translate/clm_smoke.yaml
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
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from train.checkpoint import (  # noqa: E402
    BestEvalCheckpointCallback,
    LowVRAMCheckpointCallback,
    resolve_resume_checkpoint,
)
from train.collator import build_gemma3_collator  # noqa: E402
from train.data import load_mono_pool, load_parallel_pool  # noqa: E402
from train.gemma_peft import (  # noqa: E402
    default_lora_target_modules,
    patch_peft_gemma4_clippable_linear,
)
from train.kbit import prepare_model_for_kbit_training_low_vram  # noqa: E402
from train.load_model import load_processor, load_quantized_base  # noqa: E402
from train.paths import resolve_weights_dir  # noqa: E402
from train.tokenize_clm import tokenize_clm_batch, tokenize_clm_row  # noqa: E402
from train.tokenize_sft import tokenize_translation_batch, tokenize_translation_row  # noqa: E402
from train.trainer import LowVRAMSFTTrainer  # noqa: E402


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
        task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_target_modules") or default_lora_target_modules(),
    )

    if adapter_init is not None:
        if not adapter_init.is_dir():
            raise FileNotFoundError(f"adapter_init not found: {adapter_init}")
        print(f"  continue LoRA from {adapter_init}", flush=True)
        return PeftModel.from_pretrained(model, str(adapter_init), is_trainable=True)

    return get_peft_model(model, lora_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="task-2 QLoRA (CLM or SFT)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint dir или 'latest' (иначе auto_resume из yaml)",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Не продолжать с последнего checkpoint-N в output_dir",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config.resolve())
    root = _TASK_ROOT
    stage = str(cfg.get("stage", "sft")).lower()
    model_type = str(cfg.get("model_type", "translate")).lower()
    if stage not in ("clm", "sft"):
        raise ValueError(f"stage must be clm or sft, got {stage!r}")
    if model_type not in ("gemma", "translate"):
        raise ValueError(f"model_type must be gemma or translate, got {model_type!r}")

    run_name = cfg["run_name"]
    out_root = _resolve(root, cfg["output_dir"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = resolve_weights_dir(model_type, cfg.get("model_dir"))
    adapter_init = _resolve(root, cfg.get("adapter_init"))

    with (out_root / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"stage={stage}  model={model_type}  run={run_name}", flush=True)
    print(f"output: {out_root}", flush=True)

    if model_type == "gemma":
        patch_peft_gemma4_clippable_linear()

    # --- data ---
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

    # --- model ---
    try:
        model = load_quantized_base(model_type, model_dir, load_cfg=cfg)
    except Exception as e:
        # flash_attention_2 / attn_implementation может быть недоступна в окружении
        if cfg.get("attn_implementation"):
            print(
                f"  warning: failed to load with attn_implementation={cfg.get('attn_implementation')!r}: {e}\n"
                "  retry without attn_implementation",
                flush=True,
            )
            cfg2 = dict(cfg)
            cfg2.pop("attn_implementation", None)
            model = load_quantized_base(model_type, model_dir, load_cfg=cfg2)
        else:
            raise
    low_vram = bool(cfg.get("low_vram", True))
    model = prepare_model_for_kbit_training_low_vram(
        model,
        use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        cast_norms_to_fp32=bool(cfg.get("cast_norms_to_fp32", True)),
    )
    processor = load_processor(model_type, model_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = _attach_lora(model, cfg, adapter_init=adapter_init)

    if bool(cfg.get("compile_train", False)):
        mode = str(cfg.get("compile_mode", "default"))
        try:
            model.forward = torch.compile(model.forward, mode=mode, fullgraph=False)
            print(f"  torch.compile enabled (mode={mode})", flush=True)
        except Exception as e:
            print(f"  warning: torch.compile failed: {e}", flush=True)

    # --- tokenize ---
    tok_batched = bool(cfg.get("tokenize_batched", True))
    tok_batch_size = int(cfg.get("tokenize_batch_size", 256))
    tok_num_proc = cfg.get("tokenize_num_proc")
    tok_num_proc = int(tok_num_proc) if tok_num_proc is not None else None

    if stage == "clm":
        if tok_batched:
            def _map_clm(ex):
                return tokenize_clm_batch(
                    ex, processor, model_type=model_type, max_length=max_len
                )
        else:
            def _map_clm(ex):
                return tokenize_clm_row(
                    ex, processor, model_type=model_type, max_length=max_len
                )

        remove_cols = ["text", "lang"]
        completion_only = False
    else:
        if tok_batched:
            def _map_sft(ex):
                return tokenize_translation_batch(
                    ex, processor, model_type=model_type, max_length=max_len
                )
        else:
            def _map_sft(ex):
                return tokenize_translation_row(
                    ex, processor, model_type=model_type, max_length=max_len
                )

        remove_cols = ["src", "ref"]
        completion_only = bool(cfg.get("completion_only_loss", True))

    map_kwargs: dict = {"desc": "tokenize train"}
    if tok_batched:
        map_kwargs.update({"batched": True, "batch_size": tok_batch_size})
    if tok_num_proc and tok_num_proc > 1:
        # Важно: processor может быть непиклируемым; если упадёт — выключить tokenize_num_proc.
        map_kwargs["num_proc"] = tok_num_proc
        print(f"  tokenizing train (batched={tok_batched}, bs={tok_batch_size}, num_proc={tok_num_proc}) …", flush=True)
    else:
        print(f"  tokenizing train (batched={tok_batched}, bs={tok_batch_size}) …", flush=True)

    train_ds = Dataset.from_list(train_rows).map(
        _map_clm if stage == "clm" else _map_sft,
        remove_columns=remove_cols,
        **map_kwargs,
    )
    if stage == "sft" and "prefix_mismatch" in train_ds.column_names:
        n_bad = sum(train_ds["prefix_mismatch"])
        if n_bad:
            print(f"  warning: prefix_mismatch on {n_bad}/{len(train_ds)} train rows", flush=True)
        train_ds = train_ds.remove_columns(["prefix_mismatch"])

    eval_ds = None
    if val_rows:
        print(f"  tokenizing val (batched={tok_batched}, bs={tok_batch_size}) …", flush=True)
        val_map_kwargs: dict = {"desc": "tokenize val"}
        if tok_batched:
            val_map_kwargs.update({"batched": True, "batch_size": tok_batch_size})
        eval_ds = Dataset.from_list(val_rows).map(
            _map_sft,
            remove_columns=remove_cols,
            **val_map_kwargs,
        )
        if "prefix_mismatch" in eval_ds.column_names:
            eval_ds = eval_ds.remove_columns(["prefix_mismatch"])

    save_only = bool(cfg.get("save_only_model", True))
    auto_resume = bool(cfg.get("auto_resume", True)) and not args.no_auto_resume

    max_steps = cfg.get("max_steps")
    training_args = SFTConfig(
        output_dir=str(out_root),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        max_steps=int(max_steps) if max_steps is not None else -1,
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_steps=int(cfg.get("warmup_steps", 50)),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type=str(cfg.get("loss_type", "nll")),
        completion_only_loss=completion_only,
        assistant_only_loss=False,
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 200)) if eval_ds else None,
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        save_only_model=save_only,
        max_length=max_len,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # PeftModel.forward не включает token_type_ids → Trainer иначе вырезает колонку
        remove_unused_columns=False if model_type == "translate" else True,
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False,
    )

    data_collator = None
    if model_type == "translate":
        data_collator = build_gemma3_collator(
            processor,
            completion_only_loss=completion_only,
            pad_to_multiple_of=training_args.pad_to_multiple_of,
        )
        print("  data collator: Gemma3TextLM (+ token_type_ids)", flush=True)
        print("  remove_unused_columns=False (PeftModel скрывает token_type_ids)", flush=True)

    callbacks: list[TrainerCallback] = [_LogCallback(), LowVRAMCheckpointCallback()]
    if eval_ds is not None and bool(cfg.get("save_best_eval", True)):
        best_dir_name = str(cfg.get("best_eval_dir", "best_eval"))
        callbacks.append(
            BestEvalCheckpointCallback(
                out_root,
                metric=str(cfg.get("best_eval_metric", "eval_loss")),
                best_dir_name=best_dir_name,
                greater_is_better=bool(cfg.get("best_eval_greater_is_better", False)),
            )
        )
        print(f"  best eval checkpoint: {out_root / best_dir_name}/ (min {cfg.get('best_eval_metric', 'eval_loss')})", flush=True)

    trainer = LowVRAMSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=None,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    max_logits = cfg.get("max_logits_to_keep")
    if max_logits is None and model_type == "gemma" and stage == "clm":
        max_logits = max_len + 1
    if max_logits is not None:
        trainer.max_logits_to_keep = int(max_logits)
        print(f"  max_logits_to_keep={trainer.max_logits_to_keep}", flush=True)

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
        print(f"training (resume) …", flush=True)
        trainer.train(resume_from_checkpoint=str(resume_path))
    else:
        print("training …", flush=True)
        trainer.train()

    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))

    meta = {
        "stage": stage,
        "model_type": model_type,
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
            "next: SFT with adapter_init pointing to this final/\n"
            f"  python -u scripts/train_qlora.py --config configs/{model_type}/sft_pilot.yaml",
            flush=True,
        )
    else:
        print(
            "next:\n"
            f"  python -u scripts/validate_lora.py --model {model_type} "
            f"--adapter {final_dir}\n"
            f"  python -u scripts/merge_lora.py --model {model_type} --adapter {final_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
