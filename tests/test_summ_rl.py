"""Offline tests for the RL core using a tiny random T5 (no downloads, no pretrained weights)."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summ_rl.grpo import (GroupRLConfig, GroupRLTrainer, assert_fp32_trainable, completion_labels,
                          evaluate_model, group_advantages, policy_loss_terms, shift_right)
from summ_rl.rewards import RewardConfig, reward_breakdown
from summ_rl.text import rouge_scores, tokens

VOCAB, PAD, EOS = 64, 0, 1


class Encoded(dict):
    def to(self, device):
        return Encoded({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """Words are 'w<id>'; real words are hashed into ids 2..VOCAB-1."""
    pad_token_id, eos_token_id = PAD, EOS

    def _ids(self, text, max_length):
        ids = [2 + (sum(map(ord, w)) % (VOCAB - 2)) for w in str(text).split()][:max_length - 1]
        return ids + [EOS]

    def __call__(self, texts, max_length=64, truncation=True, padding=True, return_tensors='pt'):
        rows = [self._ids(t, max_length) for t in texts]
        width = max(map(len, rows))
        ids = torch.tensor([r + [PAD] * (width - len(r)) for r in rows])
        return Encoded(input_ids=ids, attention_mask=(ids != PAD).long())

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [' '.join(f'w{int(t)}' for t in seq if int(t) > EOS) for seq in sequences]


def tiny_model():
    config = T5Config(vocab_size=VOCAB, d_model=32, d_ff=64, d_kv=16, num_layers=2, num_heads=2,
                      decoder_start_token_id=PAD, pad_token_id=PAD, eos_token_id=EOS, dropout_rate=0.1)
    return T5ForConditionalGeneration(config)


ROWS = [dict(source=' '.join(f'từ{j}x{i}' for j in range(30)), summary=' '.join(f'từ{j}x{i}' for j in range(7)))
        for i in range(12)]


class MathTests(unittest.TestCase):
    def test_rouge_is_unicode_aware(self):
        self.assertAlmostEqual(rouge_scores('Bệnh nhân uống thuốc', 'bệnh nhân uống thuốc')['rougeL'], 1.0)
        self.assertLess(rouge_scores('ma', 'má')['rouge1'], 1.0)
        self.assertEqual(tokens('2,5 mg'), ['2', '5', 'mg'])

    def test_reward_prefers_faithful_summary(self):
        source = 'Bệnh nhân dùng 5 mg thuốc mỗi ngày. ' * 30
        gold = 'Bệnh nhân dùng 5 mg thuốc mỗi ngày kéo dài nhiều tuần liên tục theo chỉ định.'
        good = reward_breakdown(source, gold, gold)
        fake = reward_breakdown(source, gold.replace('5 mg', '50 mg'), gold)
        empty = reward_breakdown(source, 'ok', gold)
        self.assertGreater(good['reward'], fake['reward'])
        self.assertEqual(fake['hallucination'], 1.0)
        self.assertEqual(empty['reward'], RewardConfig().degenerate_reward)

    def test_advantages_skip_flat_groups(self):
        rewards = torch.tensor([[1., 1., 1., 1.], [0., 1., 0., 1.]])
        adv, active = group_advantages(rewards, 0.02)
        self.assertEqual(active.tolist(), [False, True])
        self.assertEqual(adv[0].abs().sum().item(), 0.)
        self.assertAlmostEqual(adv[1].mean().item(), 0., places=5)

    def test_completion_labels_stop_after_first_eos(self):
        seq = torch.tensor([[0, 5, 6, EOS, PAD, PAD], [0, 5, 6, 7, 8, 9]])
        labels = completion_labels(seq, EOS)
        self.assertEqual(labels[0].tolist(), [5, 6, EOS, -100, -100])
        self.assertEqual(labels[1].tolist(), [5, 6, 7, 8, 9])
        self.assertEqual(shift_right(labels[:1], PAD, PAD)[0].tolist(), [0, 5, 6, EOS, PAD])

    def test_clipping_blocks_gradient_outside_trust_region(self):
        mask = torch.ones(2, 3, dtype=torch.bool)
        base = torch.zeros(2, 3)
        logp = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]], requires_grad=True)
        loss, _ = policy_loss_terms(logp, base, base, torch.tensor([1., -1.]), mask, 0.2, 0.0)
        loss.backward()
        # Positive advantage with ratio>1.2 and negative advantage with ratio<0.8 are both clipped.
        self.assertTrue(torch.all(logp.grad == 0))
        logp = torch.zeros(2, 3, requires_grad=True)
        loss, _ = policy_loss_terms(logp, base, base, torch.tensor([1., -1.]), mask, 0.2, 0.0)
        loss.backward()
        self.assertTrue(torch.all(logp.grad[0] < 0) and torch.all(logp.grad[1] > 0))

    def test_bf16_weights_lose_small_updates_but_fp32_keeps_them(self):
        """Regression for the previous Stage 3: bf16 weights + lr 1e-6 barely move."""
        results = {}
        for dtype in (torch.bfloat16, torch.float32):
            weight = torch.nn.Parameter(torch.full((1000,), 0.05, dtype=dtype))
            initial = weight.detach().clone()
            optimizer = torch.optim.AdamW([weight], lr=1e-6, weight_decay=0)
            for _ in range(20):
                weight.grad = torch.ones_like(weight)
                optimizer.step()
            results[dtype] = float((weight.detach().float() - initial.float()).abs().max())
        self.assertEqual(results[torch.bfloat16], 0.0)
        self.assertGreater(results[torch.float32], 1e-5)


class TrainerTests(unittest.TestCase):
    def make(self, **overrides):
        torch.manual_seed(0)
        model, ref = tiny_model(), tiny_model()
        ref.load_state_dict(model.state_dict())
        cfg = GroupRLConfig(group_size=4, prompts_per_step=3, max_input_len=40, max_target_len=16, max_new_tokens=12,
                            min_new_tokens=2, use_amp=False, warmup_steps=1, **overrides)
        return model, GroupRLTrainer(model, ref, FakeTokenizer(), torch.device('cpu'), cfg)

    def test_rejects_low_precision_trainable_weights(self):
        model = tiny_model().to(torch.bfloat16)
        with self.assertRaises(TypeError):
            assert_fp32_trainable(model)

    def test_step_updates_weights_and_reports_finite_stats(self):
        model, trainer = self.make(lr=1e-3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        stats, _ = trainer.step(ROWS[:3])
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))
        for key in ('reward', 'kl', 'loss', 'grad_norm', 'clip_fraction'):
            self.assertTrue(torch.isfinite(torch.tensor(float(stats[key]))), key)

    def test_policy_improves_a_verifiable_reward(self):
        """With a reward that counts one target token, RL must raise it well above chance."""
        model, trainer = self.make(lr=3e-3, kl_coef=0.0, adapt_kl=False, sft_coef=0.0, temperature=1.0)
        target = 'w7'
        trainer.reward_cfg = RewardConfig(min_words=1)
        import summ_rl.grpo as grpo
        original = grpo.reward_breakdown
        grpo.reward_breakdown = lambda s, p, g, c=None: dict(
            reward=p.split().count(target) / max(1, len(p.split())), rougeL=0., number_recall=0.,
            hallucination=0., length_ratio=1., distinct3=0.)
        try:
            first = [trainer.step(ROWS[:3])[0]['reward'] for _ in range(3)]
            for _ in range(60):
                trainer.step(ROWS[:3])
            last = [trainer.step(ROWS[:3])[0]['reward'] for _ in range(3)]
        finally:
            grpo.reward_breakdown = original
        self.assertGreater(sum(last) / 3, 3 * (sum(first) / 3) + 0.05)

    def test_train_keeps_baseline_and_logs(self):
        model, trainer = self.make(lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            history, best = trainer.train(ROWS[:6], ROWS[6:10], steps=2, out_dir=tmp, eval_every=1,
                                          eval_kwargs=dict(num_beams=2, batch_size=2), log=lambda *_: None)
            self.assertEqual(len(history), 2)
            self.assertTrue((Path(tmp) / 'best_model' / 'best_meta.json').exists())
            self.assertEqual(len((Path(tmp) / 'rl_steps.jsonl').read_text().splitlines()), 2)
            self.assertIn(best['selection'], ('SFT baseline', 'RL'))
            metrics, _ = evaluate_model(model, FakeTokenizer(), ROWS[6:8], torch.device('cpu'), num_beams=1,
                                        max_input_len=40, max_new_tokens=8, min_new_tokens=1, use_amp=False)
            self.assertEqual(metrics['n'], 2)

    def test_resume_reproduces_uninterrupted_run(self):
        kwargs = dict(eval_every=100, eval_kwargs=dict(num_beams=1, batch_size=4), log=lambda *_: None, save_every=2)
        with tempfile.TemporaryDirectory() as tmp:
            full_model, full = self.make(lr=1e-3)
            full.train(ROWS[:6], ROWS[6:10], steps=4, out_dir=Path(tmp) / 'a', **kwargs)
            resumed_model, resumed = self.make(lr=1e-3)
            history, _ = resumed.train(ROWS[:6], ROWS[6:10], steps=4, out_dir=Path(tmp) / 'a', resume=True, **kwargs)
            self.assertEqual([h['step'] for h in history], [1, 2, 3, 4])
            self.assertEqual(len((Path(tmp) / 'a' / 'rl_steps.jsonl').read_text().splitlines()), 4)
            for key, value in full_model.state_dict().items():
                self.assertTrue(torch.allclose(value, resumed_model.state_dict()[key], atol=1e-6), key)
            with self.assertRaises(FileExistsError):
                self.make(lr=1e-3)[1].train(ROWS[:6], ROWS[6:10], steps=1, out_dir=Path(tmp) / 'a', **kwargs)

    def test_stop_request_saves_state_and_resume_matches_uninterrupted_run(self):
        kwargs = dict(eval_every=100, eval_kwargs=dict(num_beams=1, batch_size=4), log=lambda *_: None, save_every=100)
        with tempfile.TemporaryDirectory() as tmp:
            full_model, full = self.make(lr=1e-3)
            full.train(ROWS[:6], ROWS[6:10], steps=3, out_dir=Path(tmp) / 'full', **kwargs)
            calls = iter([False, False, True])
            stopped_model, stopped = self.make(lr=1e-3)
            history, _ = stopped.train(ROWS[:6], ROWS[6:10], steps=3, out_dir=Path(tmp) / 'cut',
                                       should_stop=lambda: next(calls), **kwargs)
            self.assertEqual(len(history), 2)
            self.assertTrue((Path(tmp) / 'cut' / 'resume_state.pt').exists())
            resumed_model, resumed = self.make(lr=1e-3)
            history, _ = resumed.train(ROWS[:6], ROWS[6:10], steps=3, out_dir=Path(tmp) / 'cut', resume=True, **kwargs)
            self.assertEqual([h['step'] for h in history], [1, 2, 3])
            for key, value in full_model.state_dict().items():
                self.assertTrue(torch.allclose(value, resumed_model.state_dict()[key], atol=1e-6), key)

    def test_resume_without_saved_state_explains_itself(self):
        kwargs = dict(eval_every=100, eval_kwargs=dict(num_beams=1, batch_size=4), log=lambda *_: None, save_every=100)
        with tempfile.TemporaryDirectory() as tmp:
            self.make(lr=1e-3)[1].train(ROWS[:6], ROWS[6:10], steps=1, out_dir=tmp, **kwargs)
            with self.assertRaisesRegex(FileExistsError, 'resume_state.pt is missing'):
                self.make(lr=1e-3)[1].train(ROWS[:6], ROWS[6:10], steps=2, out_dir=tmp, resume=True, **kwargs)

    def test_eval_kwargs_can_override_trainer_defaults(self):
        _, trainer = self.make(lr=1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            history, _ = trainer.train(ROWS[:6], ROWS[6:8], steps=1, out_dir=tmp, log=lambda *_: None,
                                       eval_kwargs=dict(num_beams=1, max_new_tokens=4, min_new_tokens=1))
            self.assertIn('val_reward', history[0])
            self.assertTrue((Path(tmp) / 'rl_config.json').exists())
        with self.assertRaises(ValueError):
            trainer.train([], ROWS[:2], steps=1, out_dir='unused')


class ConfigTests(unittest.TestCase):
    def test_invalid_configs_fail_fast(self):
        for bad in (dict(group_size=1), dict(lr=0), dict(min_new_tokens=500), dict(kl_coef_min=2.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                GroupRLConfig(**bad)
        for bad in (dict(min_ratio=.3), dict(w_halluc=-1), dict(rouge_weights=(1., 0.))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                RewardConfig(**bad)

    def test_text_helpers_accept_none(self):
        self.assertEqual(tokens(None), [])
        self.assertEqual(reward_breakdown('nguồn', None, 'gold')['reward'], RewardConfig().degenerate_reward)


if __name__ == '__main__':
    unittest.main()
