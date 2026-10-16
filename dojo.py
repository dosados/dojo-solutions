#!/usr/bin/env python3
"""One CLI for the original translation submissions and research scripts.

The inference adapter only handles paths and input/output formats. Generation
runs through each original submission's main(), with its original defaults.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {
    'gemma': ('google/gemma-4-E2B-it', 'submission'),
    'translate': ('google/translategemma-4b-it', 'submission_translate'),
    'nllb': ('facebook/nllb-200-3.3B', 'submission_nllb'),
}


def submission(task: int, model: str) -> Path:
    if task == 1 and model == 'nllb':
        raise ValueError('NLLB is implemented only for task 2 (RU → Abkhaz).')
    return ROOT / f'task-{task}' / MODELS[model][1]


def load_rows(path: Path) -> list[dict]:
    if path.suffix in ('.pickle', '.pkl'):
        with path.open('rb') as handle:
            rows = pickle.load(handle)  # Only use pickle files from trusted sources.
    elif path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input must be a nonempty list of {rid: int, src: str}.')
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or type(row.get('rid')) is not int or not isinstance(row.get('src'), str):
            raise ValueError('Each input row needs an integer rid and a string src.')
        if row['rid'] in ids:
            raise ValueError(f"Duplicate rid: {row['rid']}")
        ids.add(row['rid'])
    return [{'rid': row['rid'], 'src': row['src']} for row in rows]


def run_original(source: Path, weights: Path, work: Path, extra: list[str]) -> None:
    """Load original code in isolation; keep algorithms and model defaults intact."""
    previous_cwd, previous_argv, previous_path = Path.cwd(), sys.argv[:], sys.path[:]
    try:
        spec = importlib.util.spec_from_file_location('dojo_submission', source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load {source}')
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(source.parent))
        spec.loader.exec_module(module)
        module.MODEL_DIR = str(weights)
        os.chdir(work)
        sys.argv = [str(source), *extra]
        module.main()
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        sys.path[:] = previous_path


def infer(args: argparse.Namespace) -> None:
    sub = submission(args.task, args.model)
    weights = (args.weights or sub / 'weights').expanduser().resolve()
    rows = load_rows(args.input.expanduser().resolve())
    output = args.output.expanduser().resolve()
    extra = args.runtime_args
    if extra[:1] == ['--']:
        extra = extra[1:]
    if extra and (args.task, args.model) != (1, 'gemma'):
        raise ValueError('Extra runtime flags are supported only by task 1 / gemma.')
    # The wrapper owns I/O paths; prevent conflicting output destinations.
    if any(x.split('=')[0] in ('--input', '--output') for x in extra):
        raise ValueError('Pass --input and --output to dojo.py before the -- separator.')
    plan = {'task': args.task, 'model': args.model, 'entrypoint': str(sub / 'solution.py'),
            'weights': str(weights), 'examples': len(rows), 'output': str(output),
            'runtime_args': extra}
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if not (weights / 'config.json').is_file() or not any(weights.glob('*.safetensors')) and not any(weights.glob('*.bin')):
        raise FileNotFoundError(f'Model weights missing/incomplete: {weights}. Run the download command first.')
    with tempfile.TemporaryDirectory(prefix='dojo-infer-') as directory:
        work = Path(directory)
        with (work / 'input.pickle').open('wb') as handle:
            pickle.dump(rows, handle)
        run_original(sub / 'solution.py', weights, work, extra)
        results = json.loads((work / 'output.json').read_text(encoding='utf-8'))
        if (not isinstance(results, list) or len(results) != len(rows)
                or any(not isinstance(r, dict) or type(r.get('rid')) is not int
                       or not isinstance(r.get('translation'), str) for r in results)
                or {r['rid'] for r in results} != {r['rid'] for r in rows}):
            raise ValueError('Submission output does not match the input rid/translation contract.')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved {len(results)} translations to {output}', flush=True)


def download(args: argparse.Namespace) -> None:
    sub = submission(args.task, args.model)
    target = (args.weights or sub / 'weights').expanduser().resolve()
    # Never overwrite a locally merged/fine-tuned model with Hub base weights.
    if target.exists() and any(target.iterdir()):
        raise ValueError(f'Target is not empty: {target}. Choose a new --weights directory.')
    from huggingface_hub import snapshot_download
    repo = MODELS[args.model][0]
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, revision=args.revision, local_dir=str(target))
    (target / 'dojo-source.json').write_text(json.dumps(
        {'repo_id': repo, 'revision': args.revision, 'kind': 'base model'}, indent=2) + '\n')
    print(f'Base model saved to {target}. This is not a contest fine-tuned checkpoint.')


def pack(args: argparse.Namespace) -> None:
    sub = submission(args.task, args.model)
    weights = (args.weights or sub / 'weights').expanduser().resolve()
    if not (weights / 'config.json').is_file() or not (any(weights.glob('*.safetensors')) or any(weights.glob('*.bin'))):
        raise FileNotFoundError(f'Model weights missing/incomplete: {weights}')
    output = args.output.expanduser().resolve()
    if output.is_relative_to(sub) or output.is_relative_to(weights):
        raise ValueError('Archive output must be outside the submission and weights directories.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        # Submission code is kept flat, while optional model copies and cache are omitted.
        for path in sorted(sub.iterdir()):
            if path.is_file() and (path.suffix == '.py' or path.name == 'Dockerfile'):
                archive.write(path, path.name)
        for path in sorted(weights.rglob('*')):
            relative = path.relative_to(weights)
            if path.is_file() and '.cache' not in relative.parts and not path.name.endswith('.incomplete'):
                archive.write(path, str(Path('weights') / relative))
    print(f'Contest archive: {output} ({output.stat().st_size / 2**30:.2f} GiB)')


def script_command(args: argparse.Namespace) -> None:
    task_root = ROOT / f'task-{args.task}'
    if args.command == 'evaluate':
        script = ROOT / 'task-1/scripts/eval_bleu.py'
        flags = ['--hyp', str(args.hyp.resolve()), '--ref', str(args.ref.resolve())]
    else:
        config = args.config.expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(config)
        if args.task == 1:
            if args.model != 'gemma':
                raise ValueError('Task 1 training is implemented only for gemma.')
            name = 'train_lora.py'
        else:
            import yaml
            cfg = yaml.safe_load(config.read_text(encoding='utf-8'))
            # Original NLLB configs have no model_type: their trainer is dedicated.
            configured_model = cfg.get('model_type', 'nllb' if args.model == 'nllb' else 'translate')
            if configured_model != args.model:
                raise ValueError('The config model_type must match --model.')
            name = 'train_qlora_nllb.py' if args.model == 'nllb' else 'train_qlora.py'
        script = task_root / 'scripts' / name
        flags = ['--config', str(config)]
    subprocess.run([sys.executable, '-u', str(script), *flags], cwd=task_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('download', 'infer', 'pack', 'train'):
        command = commands.add_parser(name)
        command.add_argument('--task', type=int, choices=(1, 2), required=True)
        command.add_argument('--model', choices=tuple(MODELS), required=True)
        if name in ('download', 'infer', 'pack'):
            command.add_argument('--weights', type=Path, help='Optional model directory; defaults to task submission/weights')
        if name == 'download':
            command.add_argument('--revision', default='main', help='Use a Hub commit hash for a fixed base revision')
            command.set_defaults(handler=download)
        elif name == 'infer':
            command.add_argument('--input', type=Path, required=True, help='JSON, JSONL, or trusted pickle')
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--dry-run', action='store_true', help='Validate input and show the selected entry point without loading a model')
            command.add_argument('runtime_args', nargs=argparse.REMAINDER)
            command.set_defaults(handler=infer)
        elif name == 'pack':
            command.add_argument('--output', type=Path, required=True)
            command.set_defaults(handler=pack)
        else:
            command.add_argument('--config', type=Path, required=True, help='Config path relative to your current directory')
            command.set_defaults(handler=script_command)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--task', type=int, choices=(1, 2), default=1)
    evaluate.add_argument('--hyp', type=Path, required=True)
    evaluate.add_argument('--ref', type=Path, required=True)
    evaluate.set_defaults(handler=script_command)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'error: {error}\n')
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, f'Subprocess failed: {error.cmd}\n')


if __name__ == '__main__':
    main()
