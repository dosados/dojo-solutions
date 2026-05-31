#!/usr/bin/env python3
"""QLoRA SFT на OPUS (предложения). Прогресс — tqdm от Hugging Face Trainer.

Пример:
  cd task-1
  python -u scripts/train_lora.py --config configs/lora_pilot.yaml
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
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from train.data import load_train_pool, load_val_pool  # noqa: E402
from train.gemma_peft import (  # noqa: E402
    default_lora_target_modules,
    patch_peft_gemma4_clippable_linear,
)
from train.kbit import prepare_model_for_kbit_training_low_vram  # noqa: E402
from train.gemma_trainer import LowVRAMSFTTrainer  # noqa: E402
from train.tokenize_sft import tokenize_translation_row  # noqa: E402


class _TqdmCallback(TrainerCallback):
    """Явно не отключаем tqdm в Trainer (по умолчанию включён при logging)."""

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


def _resolve(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def _gpu_free_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(0)
    return free / 2**30


def _load_quantized_model(model_dir: Path):
    """4-bit на одной GPU — без CPU-offload (auto ломает bnb на ~8GB)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training needs CUDA. On CPU-only machine use a cloud GPU."
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    free_gib = _gpu_free_gib()
    if free_gib is not None:
        print(f"  GPU free VRAM: ~{free_gib:.1f} GiB", flush=True)
        if free_gib < 6.0:
            print(
                "  warning: мало VRAM — уменьшите max_seq_length в config (напр. 512–768)",
                flush=True,
            )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("loading base model (4-bit, device=cuda:0) …", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map={"": 0},
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA OOM при загрузке 4-bit модели. "
            "Закройте другие процессы на GPU или в configs/lora_pilot.yaml "
            "поставьте max_seq_length: 512 и gradient_checkpointing: true"
        ) from exc
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA EN→RU on OPUS sentences")
    parser.add_argument(
        "--config",
        type=Path,
        default=_TASK_ROOT / "configs" / "lora_pilot.yaml",
        help="YAML-конфиг (см. configs/)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Продолжить с checkpoint-директории",
    )
    args = parser.parse_args()
    cfg = _load_config(args.config.resolve())
    root = _TASK_ROOT

    run_name = cfg["run_name"]
    out_root = _resolve(root, cfg["output_dir"]) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    model_dir = _resolve(root, cfg["model_dir"])
    train_path = _resolve(root, cfg["train_file"])
    val_path = _resolve(root, cfg["val_file"])
    self_mix = cfg.get("self_mix_file")
    self_mix_path = _resolve(root, self_mix) if self_mix else None

    with (out_root / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"run: {run_name}", flush=True)
    print(f"output: {out_root}", flush=True)
    print(f"base model: {model_dir}", flush=True)
    patch_peft_gemma4_clippable_linear()

    train_rows = load_train_pool(
        train_path,
        max_samples=cfg.get("max_train_samples"),
        seed=int(cfg.get("seed", 42)),
        min_chars=int(cfg.get("min_src_chars", 10)),
        max_chars=int(cfg.get("max_src_chars", 800)),
        self_mix_path=self_mix_path,
        self_mix_max=int(cfg.get("self_mix_max", 0)),
    )
    low_vram = bool(cfg.get("low_vram", False))
    eval_during_train = bool(cfg.get("eval_during_train", not low_vram))
    max_val_samples = int(cfg.get("max_val_samples", 500))
    if not eval_during_train:
        max_val_samples = 0

    val_rows: list = []
    if max_val_samples > 0:
        val_rows = load_val_pool(
            val_path,
            max_samples=max_val_samples,
            seed=int(cfg.get("seed", 42)) + 99,
            min_chars=int(cfg.get("min_src_chars", 10)),
            max_chars=int(cfg.get("max_src_chars", 800)),
        )
    else:
        print("  val during train: off (BLEU после train в validate_lora.py)", flush=True)

    model = _load_quantized_model(model_dir)
    if low_vram:
        print("  low_vram prepare (incremental norm cast) …", flush=True)
        cast_fp32 = bool(cfg.get("cast_norms_to_fp32", True))
        model = prepare_model_for_kbit_training_low_vram(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
            cast_norms_to_fp32=cast_fp32,
        )
    else:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
        model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_dir)

    lora_r = int(cfg.get("lora_r", 16))
    target_modules = cfg.get("lora_target_modules")
    if target_modules is None:
        target_modules = default_lora_target_modules()
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    max_len = int(cfg.get("max_seq_length", 1024))

    def _tokenize_map(example: dict) -> dict:
        return tokenize_translation_row(example, processor, max_length=max_len)

    print("  tokenizing train (chat template, один проход) …", flush=True)
    train_ds = Dataset.from_list(train_rows).map(
        _tokenize_map,
        remove_columns=["src", "ref"],
        desc="tokenize train",
    )
    n_mismatch = sum(train_ds["prefix_mismatch"])
    if n_mismatch:
        print(
            f"  warning: prefix_mismatch на {n_mismatch}/{len(train_ds)} примеров "
            f"(маска через общий префикс токенов)",
            flush=True,
        )
    train_ds = train_ds.remove_columns(["prefix_mismatch"])

    eval_ds = None
    if val_rows:
        print("  tokenizing eval …", flush=True)
        eval_ds = Dataset.from_list(val_rows).map(
            _tokenize_map,
            remove_columns=["src", "ref"],
            desc="tokenize eval",
        )
        eval_ds = eval_ds.remove_columns(["prefix_mismatch"])

    use_gc = bool(cfg.get("gradient_checkpointing", True))
    loss_type = cfg.get("loss_type")
    if loss_type is None:
        loss_type = "nll"
    if low_vram and loss_type == "chunked_nll":
        print(
            "  note: low_vram — chunked_nll часто OOM на 8GB; используем nll + LowVRAMSFTTrainer",
            flush=True,
        )
        loss_type = "nll"
    print(f"  loss_type: {loss_type}", flush=True)
    completion_only = bool(cfg.get("completion_only_loss", True))
    assistant_only = bool(cfg.get("assistant_only_loss", False))

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
        lr_scheduler_type="cosine",
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type=loss_type,
        completion_only_loss=completion_only,
        assistant_only_loss=assistant_only,
        activation_offloading=bool(cfg.get("activation_offloading", low_vram)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 100)) if eval_ds else None,
        save_steps=int(cfg.get("save_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        max_length=max_len,
        packing=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="none",
        disable_tqdm=False,
        dataloader_pin_memory=False if low_vram else torch.cuda.is_available(),
    )
    print(
        f"  SFT: completion_only_loss={training_args.completion_only_loss}, "
        f"assistant_only_loss={training_args.assistant_only_loss}",
        flush=True,
    )

    # TRL SFTTrainer снова вызывает peft.prepare_model_for_kbit_training (пик +~1.5 GiB
    # на cast norm→fp32). На 8GB после low_vram prepare VRAM уже не хватает.
    peft_config_for_trainer = lora_config
    if low_vram and isinstance(model, torch.nn.Module) and not isinstance(model, PeftModel):
        print("  attaching LoRA (skip 2nd kbit prepare in SFTTrainer) …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = get_peft_model(model, lora_config)
        peft_config_for_trainer = None

    trainer_cls = LowVRAMSFTTrainer if low_vram else SFTTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        peft_config=peft_config_for_trainer,
        callbacks=[_TqdmCallback()],
    )

    if args.resume:
        print(f"resume from {args.resume}", flush=True)
        trainer.train(resume_from_checkpoint=str(args.resume))
    else:
        print("training …", flush=True)
        trainer.train()

    final_dir = out_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    meta = {
        "run_name": run_name,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "config": str(args.config.resolve()),
    }
    with (final_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"done. adapter: {final_dir}", flush=True)
    print(
        "next:\n"
        f"  python -u scripts/lora_train_validate.py --config {args.config} --skip-train\n"
        f"  python -u scripts/merge_lora.py --adapter {final_dir} --device cpu --backup\n"
        "  ./scripts/zip-submission.sh",
        flush=True,
    )


if __name__ == "__main__":
    main()
