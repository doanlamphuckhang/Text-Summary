"""Export human-approved annotations as reviewed train/val/test files.

Reads a prepared directory plus the JSONL exported by the review page. Only entries with
approved=true, a named reviewer, a matching source hash, a 20-25% length ratio and no numbers
absent from the source are exported. Nothing else is promoted; the prepared directory is not modified.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from prepare_health_data import clean_text, load_jsonl, save_jsonl

NUMBERS = re.compile(r'\d+(?:[.,]\d+)*')
SPLITS = ('train', 'val', 'test')


def export(prepared, annotations_path, output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory; existing datasets are never overwritten')
    rows = {r['id']: r for split in SPLITS for r in load_jsonl(prepared / 'draft_splits' / split / 'dataset.jsonl')}
    annotations, seen = [], set()
    for item in load_jsonl(annotations_path):
        if item['id'] in seen:
            raise ValueError('Duplicate annotation ID: ' + item['id'])
        seen.add(item['id'])
        if item['id'] not in rows:
            raise ValueError('Annotation ID is not in prepared corpus: ' + item['id'])
        annotations.append(item)
    ready = {split: [] for split in SPLITS}
    skipped = Counter()
    for item in annotations:
        row = rows[item['id']]
        if item.get('approved') is not True:
            skipped['not_approved'] += 1
            continue
        if item.get('source_sha256') != row['prepared_source_sha256']:
            raise ValueError('Annotation source hash mismatch: ' + row['id'])
        reviewer = item.get('reviewer')
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError('Approved annotation must name its reviewer: ' + row['id'])
        if item.get('split', row['split']) != row['split']:
            raise ValueError('Annotation changes the frozen split: ' + row['id'])
        summary = clean_text(item['summary'])
        ratio = len(summary.split()) / row['src_word_count']
        if not .20 <= ratio <= .25:
            raise ValueError('Approved summary must meet the 20-25% length target: ' + row['id'])
        if set(NUMBERS.findall(summary)) - set(NUMBERS.findall(row['source'])):
            raise ValueError('Approved summary contains numbers absent from source: ' + row['id'])
        ready[row['split']].append({**row, 'summary': summary, 'tgt_word_count': len(summary.split()),
                                    'length_ratio': ratio, 'label_type': 'human_reviewed',
                                    'label_status': 'human_reviewed', 'quality_tag': 'human_reviewed',
                                    'reviewer': reviewer.strip(), 'review_notes': item.get('notes', '')})
    for split in SPLITS:
        save_jsonl(output / 'reviewed_splits' / split / 'dataset.jsonl', ready[split])
    report = dict(prepared=str(prepared), annotations=str(annotations_path),
                  reviewed_counts={k: len(v) for k, v in ready.items()},
                  prepared_counts=dict(Counter(r['split'] for r in rows.values())),
                  skipped=dict(skipped), all_splits_nonempty=all(ready.values()))
    (output / 'export_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.prepared, args.annotations, args.output)
