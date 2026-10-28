"""Composite reward for faithful, well-sized summaries.

Dense terms (ROUGE against the reference, number recall, length, non-repetition) give the policy a
gradient even when sequences are fluent; the number-hallucination terms keep it grounded in the source.
Number checks compare digit strings only. They do not verify units, entities or medical truth.
"""
from dataclasses import dataclass

from .text import distinct_ngram_ratio, numbers, rouge_scores, tokens


@dataclass
class RewardConfig:
    w_quality: float = 0.50
    w_number: float = 0.15
    w_length: float = 0.15
    w_distinct: float = 0.10
    w_halluc: float = 0.25
    halluc_hard_threshold: float = 0.5
    halluc_hard_penalty: float = 0.5
    min_ratio: float = 0.18
    max_ratio: float = 0.27
    min_words: int = 8
    degenerate_reward: float = -1.0
    rouge_weights: tuple = (0.3, 0.3, 0.4)  # ROUGE-1, ROUGE-2, ROUGE-L

    def __post_init__(self):
        if not 0 < self.min_ratio < self.max_ratio:
            raise ValueError(f'Need 0 < min_ratio < max_ratio (got {self.min_ratio}, {self.max_ratio})')
        if self.min_words < 0:
            raise ValueError('min_words must be >= 0')
        if len(self.rouge_weights) != 3:
            raise ValueError('rouge_weights needs three values: ROUGE-1, ROUGE-2, ROUGE-L')
        weights = (self.w_quality, self.w_number, self.w_length, self.w_distinct, self.w_halluc,
                   self.halluc_hard_penalty, *self.rouge_weights)
        if any(w < 0 for w in weights):
            raise ValueError('Reward weights and penalties must be non-negative')


def length_ratio(source, pred):
    return len(str(pred).split()) / max(len(str(source).split()), 1)


def length_score(ratio, cfg):
    if cfg.min_ratio <= ratio <= cfg.max_ratio:
        return 1.0
    edge = cfg.min_ratio if ratio < cfg.min_ratio else cfg.max_ratio
    return max(0.0, 1.0 - abs(ratio - edge) / edge)


def number_recall(pred, gold):
    target = numbers(gold)
    return len(target & numbers(pred)) / len(target) if target else 1.0


def number_hallucination(source, pred):
    produced = numbers(pred)
    return len(produced - numbers(source)) / len(produced) if produced else 0.0


def reward_breakdown(source, pred, gold, cfg=None):
    cfg = cfg or RewardConfig()
    pred = str(pred or '')
    if len(pred.split()) < cfg.min_words:
        # Too short to judge: fixed penalty, all component scores zero (hallucination 0 keeps averages honest).
        return dict(reward=float(cfg.degenerate_reward), quality=0., rouge1=0., rouge2=0., rougeL=0., number_recall=0.,
                    hallucination=0., length_score=0., distinct3=0., length_ratio=length_ratio(source, pred))
    scores = rouge_scores(pred, gold)
    w1, w2, wl = cfg.rouge_weights
    quality = w1 * scores['rouge1'] + w2 * scores['rouge2'] + wl * scores['rougeL']
    recall = number_recall(pred, gold)
    halluc = number_hallucination(source, pred)
    ratio = length_ratio(source, pred)
    length = length_score(ratio, cfg)
    distinct = distinct_ngram_ratio(tokens(pred), 3)
    total = (cfg.w_quality * quality + cfg.w_number * recall + cfg.w_length * length
             + cfg.w_distinct * distinct - cfg.w_halluc * halluc)
    if halluc > cfg.halluc_hard_threshold:
        total -= cfg.halluc_hard_penalty
    return dict(reward=float(total), quality=quality, number_recall=recall, hallucination=halluc,
                length_score=length, distinct3=distinct, length_ratio=ratio, **scores)
