"""Measure real tokenizer lengths of a training split (run where the tokenizer is available, e.g. Colab).

Reports how many sources exceed the model input window and the tokens-per-word ratio, so that
`build_training_data.py --max-source-words` can be set from evidence instead of an assumption.
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='local checkpoint dir or hub id, e.g. VietAI/vit5-base')
    parser.add_argument('--data', type=Path, default=Path('data/health_training_v1'))
    parser.add_argument('--max-input', type=int, default=1024)
    parser.add_argument('--max-target', type=int, default=512)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
    for split in ('train', 'val', 'test'):
        rows = [json.loads(line) for line in (args.data / split / 'dataset.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        src = [len(tokenizer(r['source'])['input_ids']) for r in rows]
        tgt = [len(tokenizer(r['summary'])['input_ids']) for r in rows]
        per_word = statistics.mean(s / r['src_word_count'] for s, r in zip(src, rows))
        print(json.dumps(dict(split=split, n=len(rows), tokens_per_word=round(per_word, 3),
                              source_median=statistics.median(src), source_max=max(src),
                              source_over_limit=sum(s > args.max_input for s in src),
                              target_median=statistics.median(tgt), target_max=max(tgt),
                              target_over_limit=sum(t > args.max_target for t in tgt))))
    print(f'suggested --max-source-words = {int(args.max_input / per_word * 0.97)}')


if __name__ == '__main__':
    main()
