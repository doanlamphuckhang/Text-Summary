"""Independent artifact-level invariants; no downloads or model execution.

Checks are explicit (not `assert`), so they also run under `python -O`. Every failed invariant is
listed; the exit code is 0 only when all pass.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SPLITS = ('train', 'val', 'test')


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify(output):
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    clean = read(output / 'clean_articles.jsonl')
    manifest = read(output / 'split_manifest.jsonl')
    frozen = {r['id']: r for r in read(Path(report['frozen_manifest']))}
    rows = [r for split in SPLITS for r in read(output / 'draft_splits' / split / 'dataset.jsonl')]
    queue = read(output / 'review_queue.jsonl')
    rejected = read(output / 'rejected.jsonl')
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(rows) == len(clean) == len(manifest) == len(queue) == report['retained'],
          'record counts differ between drafts, clean articles, manifest, review queue and report')
    check(len({r['id'] for r in rows}) == len(rows), 'duplicate id in draft splits')
    check(len({r['content_key'] for r in rows}) == len(rows), 'duplicate content in draft splits')
    check(len(rows) + len(rejected) == report['input_records'], 'retained + rejected != input records')
    check({r['id'] for r in rows} == {r['id'] for r in manifest}, 'draft ids differ from the split manifest')
    check(Counter(r['split'] for r in rows) == Counter(report['split_counts']), 'split counts differ from report')
    by_id, groups = {r['id']: r for r in rows}, {}
    for r in rows:
        rid = r['id']
        check(hashlib.sha256(r['source'].encode()).hexdigest() == r['prepared_source_sha256'], f'{rid}: source hash mismatch')
        check(r['src_word_count'] == len(r['source'].split()), f'{rid}: wrong source word count')
        check(r['tgt_word_count'] == len(r['summary'].split()), f'{rid}: wrong summary word count')
        check(r['label_status'] == 'pending_human_review' and r['quality_tag'] == 'unreviewed',
              f'{rid}: draft label marked as reviewed')
        check(r['summary'] == ' '.join(e['text'] for e in r['evidence_spans']), f'{rid}: summary differs from evidence')
        check(all(r['source'][e['start']:e['end']] == e['text'] for e in r['evidence_spans']),
              f'{rid}: evidence offsets do not match source')
        if rid in frozen:
            check(r['split'] == frozen[rid]['split'], f'{rid}: frozen split changed')
            check(r['prepared_source_sha256'] == frozen[rid]['source_sha256'], f'{rid}: frozen source changed')
        groups.setdefault(r['group_id'], set()).add(r['split'])
    check(all(len(s) == 1 for s in groups.values()), 'a duplicate group crosses splits')
    for link in read(output / 'group_links.jsonl'):
        if link['a'] in by_id and link['b'] in by_id:
            check(by_id[link['a']]['split'] == by_id[link['b']]['split'],
                  f'linked records {link["a"]} and {link["b"]} are in different splits')
    check(all(r['approved'] is False and r['reviewer'] == '' for r in queue), 'review queue contains pre-approved items')
    for path, expected in report['input_hashes'].items():
        actual = hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()
        check(actual == expected, f'input file changed since preparation: {path}')
    return dict(passed=not errors, records=len(rows), groups=len(groups),
                frozen_records_preserved=sum(r['id'] in frozen for r in rows),
                human_reviewed=0, training_ready=False, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='prepared directory, e.g. data/health_multisource_prepared_v1')
    args = parser.parse_args()
    result = verify(args.directory)
    # Never overwrite an earlier validation record.
    with (args.directory / 'validation.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
