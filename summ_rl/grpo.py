"""Group-relative PPO (GRPO-style) for encoder-decoder summarizers.

Design choices that address the weaknesses of the previous Stage 3:
* Trainable weights are kept in float32 and only the forward pass uses bf16 autocast. Updates of
  size ~lr=1e-6 vanish when parameters themselves are bf16 (8 mantissa bits).
* K candidates are sampled per source and compared inside the group (advantage = standardized
  reward). There is no value head or critic, so there is nothing extra to fit on ~800 documents.
* Each sampled batch is reused for several clipped-PPO updates, with the per-token KL to the
  frozen SFT reference (k3 estimator) adaptively controlled, plus a small supervised NLL anchor.
* Dropout is off during sampling and updating, so the importance ratio is exactly 1 on the first
  update instead of being corrupted by dropout noise.
* The checkpoint is selected on a validation set, never below the SFT baseline, and rejected when
  number hallucination rises beyond a tolerance.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import torch
from transformers.modeling_outputs import BaseModelOutput

from .rewards import RewardConfig, reward_breakdown


@dataclass
class GroupRLConfig:
    group_size: int = 8
    prompts_per_step: int = 4
    ppo_epochs: int = 2
    lr: float = 1e-5
    warmup_steps: int = 5
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    adapt_kl: bool = True
    target_kl: float = 0.05
    kl_coef_min: float = 0.005
    kl_coef_max: float = 1.0
    kl_abort: float = 2.0
    sft_coef: float = 0.1
    temperature: float = 0.9
    adv_std_floor: float = 0.02
    max_input_len: int = 1024
    max_target_len: int = 512
    max_new_tokens: int = 320
    min_new_tokens: int = 48
    use_amp: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError('group_size must be >= 2: advantages are standardized inside each group')
        for name in ('prompts_per_step', 'ppo_epochs', 'max_input_len', 'max_target_len', 'max_new_tokens'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be >= 1')
        if not 0 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError('Need 0 <= min_new_tokens <= max_new_tokens')
        if self.lr <= 0 or self.temperature <= 0 or self.clip_eps <= 0:
            raise ValueError('lr, temperature and clip_eps must be positive')
        if not 0 <= self.kl_coef_min <= self.kl_coef_max:
            raise ValueError('Need 0 <= kl_coef_min <= kl_coef_max')
        if self.kl_coef < 0 or self.sft_coef < 0 or self.target_kl <= 0:
            raise ValueError('kl_coef and sft_coef must be >= 0 and target_kl > 0')


def assert_fp32_trainable(model):
    bad = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad and p.dtype != torch.float32})
    if bad:
        raise TypeError(f'Trainable parameters must be float32 (found {bad}); tiny RL updates are lost in '
                        'bf16/fp16 weights. Load the model in float32 and use autocast for speed.')


def shift_right(labels, start_id, pad_id):
    shifted = labels.new_full(labels.shape, pad_id)
    shifted[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
    shifted[:, 0] = start_id
    return shifted


def completion_labels(sequences, eos_id):
    """Labels for sampled sequences: tokens up to and including the first EOS, -100 afterwards."""
    labels = sequences[:, 1:].clone()
    is_eos = labels == eos_id
    first_eos = is_eos.float().argmax(1)
    first = torch.where(is_eos.any(1), first_eos, torch.full_like(first_eos, labels.size(1) - 1))
    keep = torch.arange(labels.size(1), device=labels.device)[None] <= first[:, None]
    return labels.masked_fill(~keep, -100)


def group_advantages(rewards, std_floor):
    """rewards: [P, K]. Returns standardized advantages and a mask of groups that carry signal."""
    mean, std = rewards.mean(1, keepdim=True), rewards.std(1, unbiased=False, keepdim=True)
    active = (std.squeeze(1) >= std_floor)
    advantages = (rewards - mean) / (std + 1e-6)
    return advantages * active[:, None], active


def policy_loss_terms(logp, old_logp, ref_logp, advantages, mask, clip_eps, kl_coef):
    """Per-token clipped surrogate + k3 KL, summed over tokens. advantages: [N]. mask: [N, T] bool."""
    mask = mask.float()
    ratio = torch.exp(logp - old_logp)
    adv = advantages[:, None]
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    delta = ref_logp - logp
    kl = torch.exp(delta) - delta - 1
    loss = ((-surrogate + kl_coef * kl) * mask).sum()
    clipped = ((ratio - 1).abs() > clip_eps).float() * mask
    return loss, dict(kl=(kl * mask).sum().detach(), clipped=clipped.sum().detach(), tokens=mask.sum().detach())


def decoder_logprobs(model, hidden, enc_mask, decoder_input_ids, labels, temperature):
    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden), attention_mask=enc_mask,
                decoder_input_ids=decoder_input_ids)
    logits = out.logits.float() / temperature
    logp = torch.log_softmax(logits, -1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return logp * (labels != -100)


def encode(model, ids, mask):
    return model.get_encoder()(input_ids=ids, attention_mask=mask).last_hidden_state


@torch.no_grad()
def evaluate_model(model, tokenizer, rows, device, reward_cfg=None, max_input_len=1024, max_new_tokens=320,
                   min_new_tokens=48, num_beams=4, no_repeat_ngram_size=3, batch_size=8, use_amp=True):
    """Beam-search every row and return (aggregate metrics, predictions). Metrics are percentages
    except `reward` and `length_ratio`."""
    if not rows:
        raise ValueError('evaluate_model needs at least one row')
    reward_cfg = reward_cfg or RewardConfig()
    beam_kwargs = dict(num_beams=num_beams, early_stopping=True) if num_beams > 1 else dict(num_beams=1)
    was_training = model.training
    model.eval()
    amp = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp and device.type == 'cuda' else nullcontext()
    preds = []
    try:
        for i in range(0, len(rows), batch_size):
            enc = tokenizer([r['source'] for r in rows[i:i + batch_size]], max_length=max_input_len,
                            truncation=True, padding=True, return_tensors='pt').to(device)
            with amp:
                ids = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                                     max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                                     no_repeat_ngram_size=no_repeat_ngram_size, **beam_kwargs)
            preds.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    finally:
        if was_training:
            model.train()
    parts = [reward_breakdown(r['source'], p, r['summary'], reward_cfg) for r, p in zip(rows, preds, strict=True)]

    def mean(key):
        return sum(x[key] for x in parts) / len(parts)

    ratios = [x['length_ratio'] for x in parts]
    return dict(n=len(parts), reward=mean('reward'), rouge1=100 * mean('rouge1'), rouge2=100 * mean('rouge2'),
                rougeL=100 * mean('rougeL'), number_recall=100 * mean('number_recall'),
                number_hallucination=100 * mean('hallucination'),
                length_compliance=100 * sum(reward_cfg.min_ratio <= x <= reward_cfg.max_ratio for x in ratios) / len(ratios),
                length_ratio=sum(ratios) / len(ratios)), preds


class GroupRLTrainer:
    def __init__(self, model, ref_model, tokenizer, device, cfg=None, reward_cfg=None):
        self.cfg, self.reward_cfg = cfg or GroupRLConfig(), reward_cfg or RewardConfig()
        self.model, self.ref_model, self.tokenizer, self.device = model, ref_model, tokenizer, device
        assert_fp32_trainable(model)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                           lr=self.cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.kl_coef, self.updates = self.cfg.kl_coef, 0
        self.pad_id, self.eos_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        self.start_id = model.config.decoder_start_token_id
        torch.manual_seed(self.cfg.seed)

    def _amp(self):
        if self.cfg.use_amp and self.device.type == 'cuda':
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def _set_lr(self):
        scale = min(1.0, (self.updates + 1) / max(1, self.cfg.warmup_steps))
        for group in self.optimizer.param_groups:
            group['lr'] = self.cfg.lr * scale

    def step(self, rows):
        cfg, K, P = self.cfg, self.cfg.group_size, len(rows)
        self.model.eval()  # dropout off: importance ratios are exactly 1 before the first update
        enc = self.tokenizer([r['source'] for r in rows], max_length=cfg.max_input_len, truncation=True,
                             padding=True, return_tensors='pt').to(self.device)
        ids, mask = enc['input_ids'], enc['attention_mask']
        gold = self.tokenizer([r['summary'] for r in rows], max_length=cfg.max_target_len, truncation=True,
                              padding=True, return_tensors='pt')['input_ids'].to(self.device)
        gold_labels = gold.masked_fill(gold == self.pad_id, -100)
        gold_inputs = shift_right(gold_labels, self.start_id, self.pad_id)

        with torch.no_grad(), self._amp():
            sequences = self.model.generate(
                input_ids=ids, attention_mask=mask, do_sample=True, temperature=cfg.temperature, top_k=0, top_p=1.0,
                max_new_tokens=cfg.max_new_tokens, min_new_tokens=cfg.min_new_tokens, num_return_sequences=K,
                pad_token_id=self.pad_id, eos_token_id=self.eos_id)
        preds = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        parts = [[reward_breakdown(rows[i]['source'], preds[i * K + k], rows[i]['summary'], self.reward_cfg)
                  for k in range(K)] for i in range(P)]
        rewards = torch.tensor([[x['reward'] for x in group] for group in parts], device=self.device)
        advantages, active = group_advantages(rewards, cfg.adv_std_floor)
        labels = completion_labels(sequences, self.eos_id)
        decoder_inputs = sequences[:, :-1]
        flat_adv = advantages.reshape(-1)
        token_mask = labels != -100

        old_logp, ref_logp = torch.zeros(labels.shape, device=self.device), torch.zeros(labels.shape, device=self.device)
        with torch.no_grad(), self._amp():
            for i in range(P):
                if not active[i]:
                    continue
                sl = slice(i * K, (i + 1) * K)
                rep_mask = mask[i:i + 1].repeat_interleave(K, 0)
                h = encode(self.model, ids[i:i + 1], mask[i:i + 1]).repeat_interleave(K, 0)
                old_logp[sl] = decoder_logprobs(self.model, h, rep_mask, decoder_inputs[sl], labels[sl], cfg.temperature)
                h_ref = encode(self.ref_model, ids[i:i + 1], mask[i:i + 1]).repeat_interleave(K, 0)
                ref_logp[sl] = decoder_logprobs(self.ref_model, h_ref, rep_mask, decoder_inputs[sl], labels[sl], cfg.temperature)

        active_rows = active.repeat_interleave(K)
        total_tokens = max(1, int(token_mask[active_rows].sum()))
        sums = dict(kl=0., clipped=0., tokens=0., grad_norm=0., loss=0.)
        for _ in range(cfg.ppo_epochs):
            self._set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            for i in range(P):
                if not active[i] and cfg.sft_coef <= 0:
                    continue  # no RL signal and no supervised anchor: skip the encoder pass
                sl = slice(i * K, (i + 1) * K)
                loss = 0.
                with self._amp():
                    h = encode(self.model, ids[i:i + 1], mask[i:i + 1])
                    if active[i]:
                        logp = decoder_logprobs(self.model, h.repeat_interleave(K, 0), mask[i:i + 1].repeat_interleave(K, 0),
                                                decoder_inputs[sl], labels[sl], cfg.temperature)
                        term, info = policy_loss_terms(logp, old_logp[sl], ref_logp[sl], flat_adv[sl], token_mask[sl],
                                                       cfg.clip_eps, self.kl_coef)
                        loss = loss + term / total_tokens
                        for key in ('kl', 'clipped', 'tokens'):
                            sums[key] += float(info[key])
                    if cfg.sft_coef > 0:
                        glp = decoder_logprobs(self.model, h, mask[i:i + 1], gold_inputs[i:i + 1], gold_labels[i:i + 1], 1.0)
                        nll = -glp.sum() / max(1, int((gold_labels[i:i + 1] != -100).sum()))
                        loss = loss + cfg.sft_coef * nll / P
                if torch.is_tensor(loss):
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite RL loss; stopping to protect the checkpoint')
                    loss.backward()
                    sums['loss'] += float(loss.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError('Non-finite gradient norm; stopping to protect the checkpoint')
            sums['grad_norm'] += float(grad_norm)
            self.optimizer.step()
            self.updates += 1

        tokens = max(1., sums['tokens'])
        kl = sums['kl'] / tokens
        if cfg.adapt_kl and active.any():
            if kl > 2 * cfg.target_kl:
                self.kl_coef = min(cfg.kl_coef_max, self.kl_coef * 1.5)
            elif kl < cfg.target_kl / 2:
                self.kl_coef = max(cfg.kl_coef_min, self.kl_coef / 1.5)
        flat = [x for group in parts for x in group]

        def mean(key):
            return sum(x[key] for x in flat) / len(flat)

        lengths = labels.ne(-100).sum(1).float()
        return dict(reward=mean('reward'), reward_std_in_group=float(rewards.std(1, unbiased=False).mean()),
                    rouge_l=mean('rougeL'), number_recall=mean('number_recall'), hallucination=mean('hallucination'),
                    length_ratio=mean('length_ratio'), distinct3=mean('distinct3'), kl=kl, kl_coef=self.kl_coef,
                    clip_fraction=sums['clipped'] / tokens, active_groups=float(active.float().mean()),
                    grad_norm=sums['grad_norm'] / cfg.ppo_epochs, loss=sums['loss'] / cfg.ppo_epochs,
                    avg_tokens=float(lengths.mean()), diverged=bool(kl > cfg.kl_abort)), preds[0]

    def save(self, directory, meta=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory))
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(str(directory))
        if meta is not None:
            (directory / 'best_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return directory

    def save_state(self, path, **extra):
        """Atomic full training state (weights, optimizer, schedule, data order) for Colab disconnects."""
        path = Path(path)
        tmp = path.with_suffix('.tmp')
        torch.save(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
                        kl_coef=self.kl_coef, updates=self.updates, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, **extra), tmp)
        tmp.replace(path)

    def load_state(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state.pop('model'))
        self.optimizer.load_state_dict(state.pop('optimizer'))
        self.kl_coef, self.updates = state.pop('kl_coef'), state.pop('updates')
        torch.set_rng_state(state.pop('torch_rng').cpu())
        cuda_rng = state.pop('cuda_rng')
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng])
        return state

    def train(self, train_rows, val_rows, steps, out_dir, eval_every=20, eval_kwargs=None, max_halluc_increase=1.0,
              log=print, should_stop=None, save_every=25, resume=False):
        """Run RL; returns (history, best_meta). The SFT baseline is evaluated and saved first.

        With resume=True an interrupted run continues from `resume_state.pt` in out_dir (written every
        `save_every` steps and whenever `should_stop` asks to stop); the same data order, optimizer state
        and best checkpoint are restored. `best_model/` on disk is the authority for the best score."""
        if not train_rows or not val_rows:
            raise ValueError('train_rows and val_rows must both be non-empty')
        if min(steps, eval_every, save_every) < 1:
            raise ValueError('steps, eval_every and save_every must be >= 1')
        cfg, out_dir = self.cfg, Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_kwargs = {**dict(max_input_len=cfg.max_input_len, max_new_tokens=cfg.max_new_tokens,
                              min_new_tokens=cfg.min_new_tokens, use_amp=cfg.use_amp), **(eval_kwargs or {})}
        state_path, log_path, best_dir = out_dir / 'resume_state.pt', out_dir / 'rl_steps.jsonl', out_dir / 'best_model'
        first_step, rng, queue, history = 1, random.Random(cfg.seed), [], []
        if resume and state_path.exists():
            saved = self.load_state(state_path)
            base, best, queue, history = saved['base'], saved['best'], saved['queue'], saved['history']
            rng.setstate(saved['rng'])
            first_step = saved['step'] + 1
            # best_model/ may have been replaced after the last state save; never overwrite it with a worse model.
            on_disk = read_best_meta(best_dir)
            if on_disk is not None and on_disk.get('reward', -math.inf) > best['reward']:
                best = on_disk
            lines = [json.dumps(h, ensure_ascii=False, allow_nan=False) + '\n' for h in history]
            log_path.write_text(''.join(lines), encoding='utf-8')
            log(f'resumed after step {saved["step"]} (best so far: step {best["step"]}, reward {best["reward"]:.4f})')
        else:
            if log_path.exists():
                if resume:
                    raise FileExistsError(f'resume=True but {state_path.name} is missing (the run stopped before its '
                                          'first state save); start again in a fresh out_dir')
                raise FileExistsError('out_dir already holds a run; use a fresh directory or resume=True')
            base, _ = evaluate_model(self.model, self.tokenizer, val_rows, self.device, self.reward_cfg, **eval_kwargs)
            best = dict(step=0, selection='SFT baseline', **base)
            self.save(best_dir, best)
            log(f'baseline: {json.dumps(base)}')
        (out_dir / 'rl_config.json').write_text(
            json.dumps(dict(rl=asdict(cfg), reward=asdict(self.reward_cfg), steps=steps, eval_every=eval_every,
                            max_halluc_increase=max_halluc_increase), indent=2), encoding='utf-8')

        def save_progress(done):
            self.save_state(state_path, base=base, best=best, queue=queue, history=history, rng=rng.getstate(), step=done)

        for step in range(first_step, steps + 1):
            if should_stop and should_stop():
                save_progress(step - 1)
                log(f'stop requested; state saved after step {step - 1} (resume=True continues from there)')
                break
            while len(queue) < cfg.prompts_per_step:
                epoch_rows = list(train_rows)
                rng.shuffle(epoch_rows)
                queue.extend(epoch_rows)
            batch, queue = queue[:cfg.prompts_per_step], queue[cfg.prompts_per_step:]
            started = time.time()
            stats, _ = self.step(batch)
            stats['step'], stats['seconds'] = step, time.time() - started
            if step % eval_every == 0 or step == steps:
                metrics, _ = evaluate_model(self.model, self.tokenizer, val_rows, self.device, self.reward_cfg, **eval_kwargs)
                stats.update({f'val_{k}': v for k, v in metrics.items()})
                safe = metrics['number_hallucination'] <= base['number_hallucination'] + max_halluc_increase
                if safe and metrics['reward'] > best['reward']:
                    best = dict(step=step, selection='RL', **metrics)
                    self.save(best_dir, best)
                    log(f'step {step}: new best val reward {metrics["reward"]:.4f}')
            history.append(stats)
            with log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(stats, ensure_ascii=False, allow_nan=False) + '\n')
            log(f'step {step}: reward={stats["reward"]:+.3f} kl={stats["kl"]:.4f} coef={stats["kl_coef"]:.3f} '
                f'active={stats["active_groups"]:.2f} clip={stats["clip_fraction"]:.2f} halluc={stats["hallucination"]:.2f} '
                f'{stats["seconds"]:.1f}s/step, ETA {(steps - step) * stats["seconds"] / 60:.0f} min')
            if stats['diverged']:
                log('KL exceeded kl_abort; stopping and keeping the best checkpoint')
                break
            if step % save_every == 0 and step < steps:
                save_progress(step)
        return history, best


def read_best_meta(directory):
    """Metadata of a saved best checkpoint, or None when it is missing or unreadable."""
    try:
        return json.loads((Path(directory) / 'best_meta.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
