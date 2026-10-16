"""CPU-only checks for I/O, routing and packaging; no model downloads."""
import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import dojo


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.json'
        self.rows = [{'rid': 7, 'src': 'Hello'}, {'rid': 2, 'src': 'World'}]
        self.input.write_text(json.dumps(self.rows))

    def tearDown(self):
        self.tmp.cleanup()

    def test_supported_routes(self):
        for task, model, folder in [(1, 'gemma', 'submission'), (1, 'translate', 'submission_translate'),
                                    (2, 'gemma', 'submission'), (2, 'translate', 'submission_translate'),
                                    (2, 'nllb', 'submission_nllb')]:
            self.assertTrue((dojo.submission(task, model) / 'solution.py').is_file())
            self.assertEqual(dojo.submission(task, model).name, folder)
        with self.assertRaises(ValueError):
            dojo.submission(1, 'nllb')

    def test_json_jsonl_pickle_contract(self):
        for suffix in ('.json', '.jsonl', '.pickle'):
            path = self.root / ('rows' + suffix)
            rows = [dict(r, ref='Ignored by model') for r in self.rows]
            if suffix == '.pickle':
                path.write_bytes(pickle.dumps(rows))
            elif suffix == '.jsonl':
                path.write_text('\n'.join(json.dumps(r) for r in rows))
            else:
                path.write_text(json.dumps(rows))
            self.assertEqual(dojo.load_rows(path), self.rows)

    def test_invalid_inputs(self):
        for rows in [[], {}, [{'rid': 1, 'src': 42}], [{'rid': True, 'src': 'x'}],
                     [{'rid': 1, 'src': 'x'}, {'rid': 1, 'src': 'y'}]]:
            self.input.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                dojo.load_rows(self.input)

    def test_dry_run_does_not_load_models(self):
        r = subprocess.run([sys.executable, str(dojo.ROOT / 'dojo.py'), 'infer', '--task', '2',
                            '--model', 'translate', '--input', str(self.input),
                            '--output', str(self.root / 'answer.json'), '--dry-run'],
                           text=True, capture_output=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['examples'], 2)
        self.assertFalse((self.root / 'answer.json').exists())

    def test_original_module_and_context_restoration(self):
        source = self.root / 'solution.py'
        source.write_text('''import json, pickle
MODEL_DIR = './weights'
NUM_BEAMS = 3
def main():
    assert NUM_BEAMS == 3
    assert MODEL_DIR.endswith('custom-weights')
    with open('input.pickle', 'rb') as f:
        rows = pickle.load(f)
    with open('output.json', 'w') as f:
        json.dump([{'rid': r['rid'], 'translation': r['src']} for r in rows], f)
''')
        work = self.root / 'work'
        work.mkdir()
        (work / 'input.pickle').write_bytes(pickle.dumps(self.rows))
        previous = Path.cwd(), sys.argv[:], sys.path[:]
        dojo.run_original(source, self.root / 'custom-weights', work, [])
        self.assertEqual((Path.cwd(), sys.argv, sys.path), previous)
        result = json.loads((work / 'output.json').read_text())
        self.assertEqual([r['rid'] for r in result], [7, 2])

    def test_context_restored_on_model_error(self):
        source = self.root / 'bad.py'
        source.write_text('def main():\n    raise RuntimeError("model failed")\n')
        previous = Path.cwd(), sys.argv[:], sys.path[:]
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            dojo.run_original(source, self.root, self.root, [])
        self.assertEqual((Path.cwd(), sys.argv, sys.path), previous)

    def test_download_refuses_nonempty_model_dir(self):
        with self.assertRaisesRegex(ValueError, 'not empty'):
            dojo.download(argparse.Namespace(task=1, model='gemma', weights=self.root, revision='main'))

    def test_pack_layout_and_cache_exclusion(self):
        sub = self.root / 'submission'
        weights = sub / 'weights'
        cache = weights / '.cache'
        cache.mkdir(parents=True)
        (sub / 'solution.py').write_text('pass')
        (sub / 'prompts.py').write_text('PROMPT = "keep source"')
        (sub / 'Dockerfile').write_text('FROM scratch')
        (sub / 'output.json').write_text('[]')
        (weights / 'config.json').write_text('{}')
        (weights / 'model.safetensors').write_bytes(b'model')
        (cache / 'token').write_text('exclude')
        (weights / 'download.incomplete').write_text('exclude')
        output = self.root / 'artifact.zip'
        with patch.object(dojo, 'submission', return_value=sub):
            dojo.pack(argparse.Namespace(task=1, model='gemma', weights=weights, output=output))
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {'solution.py', 'prompts.py', 'Dockerfile',
                                                     'weights/config.json', 'weights/model.safetensors'})

    def test_pack_rejects_recursive_output(self):
        weights = self.root / 'weights'
        weights.mkdir()
        (weights / 'config.json').write_text('{}')
        (weights / 'model.safetensors').write_bytes(b'model')
        with patch.object(dojo, 'submission', return_value=self.root):
            with self.assertRaisesRegex(ValueError, 'outside'):
                dojo.pack(argparse.Namespace(task=1, model='gemma', weights=weights,
                                             output=weights / 'recursive.zip'))

    def test_training_dispatch_preserves_script(self):
        with patch.object(dojo.subprocess, 'run') as run:
            dojo.script_command(argparse.Namespace(command='train', task=2, model='nllb',
                config=dojo.ROOT / 'task-2/configs/nllb/sft_smoke.yaml'))
        args, kwargs = run.call_args
        self.assertTrue(args[0][2].endswith('train_qlora_nllb.py'))
        self.assertEqual(kwargs['cwd'], dojo.ROOT / 'task-2')

    def test_training_config_model_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'model_type'):
            dojo.script_command(argparse.Namespace(command='train', task=2, model='gemma',
                config=dojo.ROOT / 'task-2/configs/translate/sft_smoke.yaml'))


    def test_infer_validates_results_before_publishing(self):
        sub = self.root / 'sub'
        weights = sub / 'weights'
        weights.mkdir(parents=True)
        (weights / 'config.json').write_text('{}')
        (weights / 'model.safetensors').write_bytes(b'model')
        output = self.root / 'result.json'
        args = argparse.Namespace(task=2, model='gemma', weights=weights,
            input=self.input, output=output, runtime_args=[], dry_run=False)
        def valid(source, model, work, extra):
            (work / 'output.json').write_text(json.dumps([
                {'rid': row['rid'], 'translation': 'Translated'} for row in self.rows]))
        with patch.object(dojo, 'submission', return_value=sub), patch.object(dojo, 'run_original', side_effect=valid):
            dojo.infer(args)
        self.assertEqual(len(json.loads(output.read_text())), 2)
        output.unlink()
        def invalid(source, model, work, extra):
            (work / 'output.json').write_text('[{"rid": 999, "translation": "wrong"}]')
        with patch.object(dojo, 'submission', return_value=sub), patch.object(dojo, 'run_original', side_effect=invalid):
            with self.assertRaisesRegex(ValueError, 'contract'):
                dojo.infer(args)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
