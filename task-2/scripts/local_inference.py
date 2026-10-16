"""Локальный inference для task-2 (не входит в zip контеста)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

QuantizeMode = Literal["none", "4bit"]

# bf16 Gemma-4-E2B ~7.3 GiB; на GPU ≤ этого порога включаем 4bit при --gpu-only
_GPU_4BIT_TOTAL_GIB = 8.5

GEMMA_MAX_NEW_TOKENS = 1024
TRANSLATE_MAX_NEW_TOKENS = 128
NLLB_MAX_NEW_TOKENS = 128
NLLB_MAX_INPUT_LENGTH = 512

GEMMA_PROMPT = """
Ты профессиональный переводчик. Твоя задача перевести текст с русского на абхазский язык.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
"""


def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _gpu_mem_gib() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info(0)
    return free / 2**30, total / 2**30


def resolve_load_mode(
    *,
    gpu_only: bool,
    quantize: QuantizeMode,
) -> tuple[bool, QuantizeMode]:
    """На ~8GB GPU bf16 целиком не влезает — автоматически 4bit при --gpu-only."""
    if quantize != "none" or not gpu_only:
        return gpu_only, quantize
    _free, total = _gpu_mem_gib()
    if total is not None and total < _GPU_4BIT_TOTAL_GIB:
        print(
            f"  auto: GPU {total:.1f} GiB — bf16 целиком не влезет; "
            f"включаем 4bit (или уберите --gpu-only для CPU offload)",
            flush=True,
        )
        return True, "4bit"
    return gpu_only, quantize


def _model_load_kwargs(*, gpu_only: bool, quantize: QuantizeMode) -> dict:
    if quantize == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("--quantize 4bit: нужна CUDA и pip install bitsandbytes")
        return {
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
            "quantization_config": _bnb_4bit_config(),
            "attn_implementation": "sdpa",
        }
    if gpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-only: CUDA недоступна")
        return {
            "dtype": "auto",
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        }
    return {"dtype": "auto", "device_map": "auto", "attn_implementation": "sdpa"}


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_compile(model, *, enabled: bool):
    if not enabled:
        return model
    if not torch.cuda.is_available():
        print("  compile: skipped (no CUDA)", flush=True)
        return model
    try:
        print("  compile: torch.compile(forward, mode=default)", flush=True)
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  compile: skipped ({exc})", flush=True)
    return model


def _iter_batches(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size <= 1:
        for row in rows:
            yield [row]
        return
    ordered = sorted(rows, key=lambda r: len(r["src"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def _batch_progress(total_rows: int, batch_size: int, *, show_progress: bool):
    n_batches = (total_rows + batch_size - 1) // max(batch_size, 1)
    if show_progress and tqdm is not None:
        return tqdm(total=n_batches, unit="batch", dynamic_ncols=True)
    return None


def _log_load_mode(*, quantize: QuantizeMode, gpu_only: bool) -> None:
    if quantize == "4bit":
        print("  load: 4-bit NF4 on GPU (bitsandbytes)", flush=True)
    elif gpu_only:
        print("  load: full weights on GPU", flush=True)
    else:
        print("  load: device_map=auto (может offload на CPU)", flush=True)


def run_gemma(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int = GEMMA_MAX_NEW_TOKENS,
    batch_size: int = 1,
    use_compile: bool = False,
    warmup: bool = False,
) -> list[dict]:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=use_compile)
    device = _input_device(model)
    pad_token_id = int(tokenizer.pad_token_id)

    def _generate_batch(batch: list[dict]) -> None:
        prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": GEMMA_PROMPT + row["src"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        for row, in_len, out_ids in zip(batch, input_lens, outputs, strict=True):
            response = processor.decode(out_ids[int(in_len) :], skip_special_tokens=False)
            results.append({
                "rid": row["rid"],
                "translation": processor.parse_response(response)["content"],
            })

    results: list[dict] = []
    if warmup and rows and use_compile:
        _generate_batch(rows[: min(batch_size, len(rows))])
        results.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    progress = _batch_progress(len(rows), batch_size, show_progress=show_progress)
    for batch in _iter_batches(rows, batch_size):
        _generate_batch(batch)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return results


def _load_translate_processor(model_dir: str | Path):
    try:
        return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    except (TypeError, AttributeError):
        return AutoProcessor.from_pretrained(model_dir)


def _translate_pad_token_id(processor, model) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    gen = model.generation_config
    pid = gen.pad_token_id
    if pid is not None:
        return int(pid[0] if isinstance(pid, list) else pid)
    eos = gen.eos_token_id
    if eos is None:
        raise RuntimeError("cannot determine pad_token_id for TranslateGemma")
    return int(eos[0] if isinstance(eos, list) else eos)


def _translate_messages(src: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "ru",
                    "target_lang_code": "ab",
                    "text": src,
                }
            ],
        }
    ]


def _prepare_translate_batch(
    processor,
    batch: list[dict],
    *,
    device: torch.device,
    pad_token_id: int,
):
    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    input_lens: list[int] = []

    for row in batch:
        inputs = processor.apply_chat_template(
            _translate_messages(row["src"]),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"][0]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        else:
            mask = mask[0]
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        input_lens.append(int(ids.shape[0]))

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )
    padded = padded.to(device)
    return padded, input_lens


def run_translate(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int = TRANSLATE_MAX_NEW_TOKENS,
    batch_size: int = 40,
    use_compile: bool = False,
    warmup: bool = False,
) -> list[dict]:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = _load_translate_processor(model_dir)

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=use_compile)
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    def _generate_batch(batch: list[dict]) -> None:
        inputs, input_lens = _prepare_translate_batch(
            processor,
            batch,
            device=device,
            pad_token_id=pad_token_id,
        )
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        for row, in_len, out_ids in zip(batch, input_lens, outputs, strict=True):
            results.append({
                "rid": row["rid"],
                "translation": processor.decode(
                    out_ids[in_len:], skip_special_tokens=True
                ).strip(),
            })

    results: list[dict] = []
    if warmup and rows and use_compile:
        _generate_batch(rows[: min(batch_size, len(rows))])
        results.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    progress = _batch_progress(len(rows), batch_size, show_progress=show_progress)
    for batch in _iter_batches(rows, batch_size):
        _generate_batch(batch)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return results


def run_nllb(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int = NLLB_MAX_NEW_TOKENS,
    batch_size: int = 16,
    use_compile: bool = False,
    warmup: bool = False,
) -> list[dict]:
    import sys

    _task_root = Path(__file__).resolve().parents[1]
    if str(_task_root) not in sys.path:
        sys.path.insert(0, str(_task_root))
    from train.nllb_langs import nllb_generate_kwargs, prepare_nllb_tokenizer  # noqa: E402

    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    forced_bos = prepare_nllb_tokenizer(tokenizer)
    gen_kw = nllb_generate_kwargs(forced_bos, max_new_tokens=max_new_tokens, num_beams=4)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=use_compile)
    device = _input_device(model)

    def _generate_batch(batch: list[dict]) -> None:
        texts = [r["src"] for r in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=NLLB_MAX_INPUT_LENGTH,
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kw)
        for row, out_ids in zip(batch, outputs, strict=True):
            results.append({
                "rid": row["rid"],
                "translation": tokenizer.decode(out_ids, skip_special_tokens=True).strip(),
            })

    results: list[dict] = []
    if warmup and rows and use_compile:
        _generate_batch(rows[: min(batch_size, len(rows))])
        results.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    progress = _batch_progress(len(rows), batch_size, show_progress=show_progress)
    for batch in _iter_batches(rows, batch_size):
        _generate_batch(batch)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return results
