"""Measure real tokenizer lengths of a training split (run where the tokenizer is available, e.g. Colab).

Reports how many sources exceed the model input window and the tokens-per-word ratio, so that
`build_training_data.py --max-source-words` can be set from evidence instead of an assumption.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

SPLITS = ('train', 'val', 'test')


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def percentile(values, q):
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def length_report(rows_by_split, count_tokens, max_input=1024, max_target=512, safety=0.97):
    """Per-split token statistics plus a --max-source-words suggestion pooled over every split.

    The suggestion uses the 95th percentile of tokens per word, so about 95% of sources fit the window
    (the mean would let roughly half of the token-heavy articles overflow)."""
    splits, ratios = [], []
    for split, rows in rows_by_split.items():
        if not rows:
            splits.append(dict(split=split, n=0))
            continue
        src = [count_tokens(r['source']) for r in rows]
        tgt = [count_tokens(r['summary']) for r in rows]
        per_word = [s / max(1, len(r['source'].split())) for s, r in zip(src, rows, strict=True)]
        ratios.extend(per_word)
        splits.append(dict(split=split, n=len(rows), tokens_per_word=round(statistics.mean(per_word), 3),
                           source_median=statistics.median(src), source_max=max(src),
                           source_over_limit=sum(s > max_input for s in src),
                           target_median=statistics.median(tgt), target_max=max(tgt),
                           target_over_limit=sum(t > max_target for t in tgt)))
    if not ratios:
        raise ValueError('No rows found in any split')
    p95 = percentile(ratios, 95)
    return dict(splits=splits, tokens_per_word_mean=round(statistics.mean(ratios), 3),
                tokens_per_word_p95=round(p95, 3), suggested_max_source_words=int(max_input / p95 * safety))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='local checkpoint dir or hub id, e.g. VietAI/vit5-base')
    parser.add_argument('--data', type=Path, default=Path('data/health_training_v1'))
    parser.add_argument('--max-input', type=int, default=1024)
    parser.add_argument('--max-target', type=int, default=512)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    rows = {split: load_rows(args.data / split / 'dataset.jsonl') for split in SPLITS
            if (args.data / split / 'dataset.jsonl').exists()}
    report = length_report(rows, lambda text: len(tokenizer(text)['input_ids']), args.max_input, args.max_target)
    for line in report['splits']:
        print(json.dumps(line))
    print(f'tokens/word: mean {report["tokens_per_word_mean"]}, p95 {report["tokens_per_word_p95"]}')
    print(f'suggested --max-source-words = {report["suggested_max_source_words"]}')


if __name__ == '__main__':
    main()
