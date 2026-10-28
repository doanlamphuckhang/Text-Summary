"""Offline regression tests: execute selected pure helpers, never notebook training."""
import ast
from contextlib import nullcontext, redirect_stdout
import io
import json
import math
from pathlib import Path
import random
import re
import tempfile
import types
import unittest
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PPO = ROOT / 'notebooks/ViT5_base/stage3_vit5_RLUsingPPO.ipynb'
SFT = ROOT / 'notebooks/ViT5_base/stage2_vit5_base.ipynb'


def cells(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))['cells']


def source(path, index):
    return ''.join(cells(path)[index]['source'])


def helpers(code, names, **values):
    tree = ast.parse(code)
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    scope = dict(re=re, json=json, Path=Path, random=random, unicodedata=unicodedata)
    scope.update(values)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook helpers>', 'exec'), scope)
    return scope


class NotebookRegressionTests(unittest.TestCase):
    def test_all_python_cells_parse(self):
        for path in ROOT.glob('notebooks/**/*.ipynb'):
            for i, cell in enumerate(cells(path)):
                if cell['cell_type'] != 'code':
                    continue
                # Only remove IPython shell/magic lines; never execute cell bodies.
                code = '\n'.join('pass' if line.lstrip().startswith(('!', '%')) else line
                                 for line in ''.join(cell['source']).splitlines())
                with self.subTest(path=path.name, cell=i):
                    ast.parse(code)

    def test_unicode_rouge_tokens(self):
        for path in ROOT.glob('notebooks/**/*.ipynb'):
            for cell in cells(path):
                code = ''.join(cell['source'])
                if cell['cell_type'] != 'code' or 'rouge.compute(' not in code:
                    continue
                scope = helpers(code, ['rouge_tokenize'])
                tokenize = scope['rouge_tokenize']
                text = 'BỆNH nhân uống 2,5 mg.'
                self.assertEqual(tokenize(text), ['bệnh', 'nhân', 'uống', '2', '5', 'mg'])
                self.assertEqual(tokenize(text), tokenize(unicodedata.normalize('NFD', text)))
                self.assertNotEqual(tokenize('ma'), tokenize('má'))
                for node in ast.walk(ast.parse(code)):
                    if isinstance(node, ast.Call) and ast.unparse(node.func) == 'rouge.compute':
                        self.assertIn('tokenizer', [kw.arg for kw in node.keywords])

    def test_records_reject_null_and_empty(self):
        scope = helpers(source(SFT, 8), ['normalize_record', 'validate_splits'])
        for row in ({'source': None, 'summary': 'x'}, {'source': 'x', 'summary': ' '},
                    {'source': 42, 'summary': 'x'}):
            with self.assertRaises(ValueError):
                scope['normalize_record'](row)
        row = scope['normalize_record']({'id': 5, 'source': ' a b ', 'summary': 'a'})
        self.assertEqual(row['id'], '5')
        self.assertEqual(row['length_ratio'], .5)

    def test_split_leakage_by_text_id_url_and_empty_split(self):
        scope = helpers(source(SFT, 8), ['normalize_record', 'validate_splits'])
        norm, validate = scope['normalize_record'], scope['validate_splits']
        a = norm({'id': 'a', 'url': 'url-a', 'source': 'Bệnh nhân', 'summary': 'Bệnh'})
        b = norm({'id': 'b', 'url': 'url-b', 'source': 'khác', 'summary': 'khác'})
        validate({'train': [a], 'val': [b]})
        for key, value in [('id', 'a'), ('url', 'url-a'), ('source', ' BỆNH  nhân ')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate({'train': [a], 'val': [{**b, key: value}]})
        with self.assertRaises(ValueError):
            validate({'train': [a], 'val': []})

    def test_augmentation_does_not_change_injection_or_duplicate(self):
        code = source(SFT, 10)
        tree = ast.parse(code)
        synonym_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                            and any(isinstance(t, ast.Name) and t.id == 'MEDICAL_SYNONYMS' for t in n.targets))
        synonyms = ast.literal_eval(synonym_node.value)
        scope = helpers(code, ['extract_numbers', 'synonym_replace_medical', 'back_translate', 'augment_dataset'],
                        MEDICAL_SYNONYMS=synonyms, NUM_RE=re.compile(r'\d+(?:[.,]\d+)*'), SEED=42,
                        CONFIG={'USE_BACK_TRANSLATE': False, 'SYNONYM_REPLACE_PROB': 1.0})
        text = 'Bệnh nhân được tiêm insulin 5 đơn vị.'
        self.assertEqual(scope['synonym_replace_medical'](text, prob=1.0, seed=42), text)
        row = {'id': 'a', 'source': text, 'summary': 'Tiêm insulin 5 đơn vị.'}
        self.assertEqual(scope['augment_dataset']([row], n_aug=2), [row])

    def test_ppo_train_test_reward_parity(self):
        config_tree = ast.parse(source(PPO, 6))
        assignment = next(n for n in config_tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in n.targets))
        config = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(assignment.value.keys, assignment.value.values, strict=True)
                  if not isinstance(v, ast.Name)}
        names = ['number_recall', 'number_hallucination', 'length_ratio_fn', 'length_score',
                 'diversity_score', 'compute_reward_breakdown']
        context = dict(CONFIG=config, EVAL_CONFIG=config,
                       extract_numbers=lambda s: set(re.findall(r'\d+(?:[.,]\d+)*', s or '')),
                       np=types.SimpleNamespace(clip=lambda x, lo, hi: max(lo, min(x, hi))))
        train = helpers(source(PPO, 17), names, **context)
        test = helpers(source(PPO, 24), names, **context)
        for src, pred, gold in [('nghỉ ngơi uống nước', 'nghỉ ngơi', 'uống nước'),
                                ('liều 5 mg', 'liều 10 mg', 'liều 10 mg'),
                                ('liều 5 mg', '5 mg', '5 mg')]:
            self.assertEqual(train['compute_reward_breakdown'](src, pred, gold),
                             test['compute_reward_breakdown'](src, pred, gold))
        self.assertEqual(test['number_hallucination']('5 mg', '10 mg'), 1.0)

    def test_checkpoint_requires_complete_local_files(self):
        scope = helpers(source(PPO, 13), ['validate_stage2_checkpoint'])
        validate = scope['validate_stage2_checkpoint']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.json').write_text('{}')
            (root / 'spiece.model').write_bytes(b'tiny synthetic placeholder')
            (root / 'optimizer.pt').write_bytes(b'not model weights')
            with self.assertRaises(FileNotFoundError):
                validate(root)
            index = root / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'w': 'model-00001.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                validate(root)
            (root / 'model-00001.safetensors').write_bytes(b'tiny synthetic placeholder')
            self.assertEqual(validate(root), root)
            index.write_text(json.dumps({'weight_map': {'w': '../outside.safetensors'}}))
            with self.assertRaises(ValueError):
                validate(root)

    def test_short_ppo_run_selects_baseline_or_final_validation(self):
        """Run the real loop with tiny fake model/trainer, no torch or model imports."""
        code = source(PPO, 19)
        code = code[code.index('training_log     = []'):]

        class Token:
            def squeeze(self, *args):
                return self

            def to(self, *args):
                return self

        class Tokenizer:
            def __call__(self, *args, **kwargs):
                return {'input_ids': Token()}

            def batch_decode(self, responses, **kwargs):
                return ['5 mg' for _ in responses]

        class Progress:
            def __init__(self, items, **kwargs):
                self.items = items

            def __iter__(self):
                return iter(self.items)

            def set_postfix(self, value):
                pass

        config = dict(PPO_BATCH_SIZE=2, PPO_MINI_BATCH=1, RL_EPOCHS=1, EVAL_EVERY_N_STEPS=25,
                      EVAL_VAL_SUBSET=30, MAX_INPUT_LEN=20, KL_PENALTY_MODE='abs',
                      KL_HIGH_THRESHOLD=5, KL_HIGH_STREAK=3, BEST_W_ROUGEL=.5, BEST_W_REWARD=.5)
        for final_score, selected_step in [(40, 0), (60, 2)]:
            with self.subTest(final_score=final_score), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                evaluations = iter([50, final_score])
                saved_steps = []

                def save_actor(path, step=None):
                    path.mkdir(exist_ok=True)
                    saved_steps.append(step)  # noqa: B023 - called inside this iteration
                    return path

                def reward(*args):
                    return dict(reward=.1, number_recall=1., hallucination=0.,
                                length_score=.5, diversity=1., hard_penalty=0.)

                scope = dict(CONFIG=config, OUT_DIR=out, RESULTS_DIR=out, BEST_DIR_S3=out / 'best_model',
                             quick_eval_on_val=lambda **kw: dict(rougeL=next(evaluations), reward_mean=.1),  # noqa: B023
                             save_actor=save_actor, train_rl=[dict(source='5 mg', summary='5 mg')] * 4,
                             compute_reward_breakdown=reward, check_stop_requested=lambda: None,
                             clear_cuda=lambda: None, MODEL_DTYPE='fake', DEVICE='cpu', tqdm=Progress,
                             tokenizer=Tokenizer(), random=random, json=json, math=math,
                             generation_kwargs={}, tensor_to_float=float,
                             np=types.SimpleNamespace(mean=lambda xs: sum(xs) / len(xs)),
                             torch=types.SimpleNamespace(no_grad=nullcontext, tensor=lambda x, **kw: x),
                             ppo_trainer=types.SimpleNamespace(generate=lambda qs, **kw: qs,
                                 step=lambda *args: {'objective/kl': .1, 'ppo/loss/total': .2}))
                with redirect_stdout(io.StringIO()):
                    exec(compile(code, '<PPO loop>', 'exec'), scope)
                self.assertEqual(scope['best_meta']['step'], selected_step)
                self.assertEqual(saved_steps, [0] if selected_step == 0 else [0, 2])
                logs = [json.loads(line) for line in (out / 'ppo_training_steps.jsonl').read_text().splitlines()]
                self.assertEqual([row['step'] for row in logs], [1, 2])
                self.assertIn('val_rougeL', logs[-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
