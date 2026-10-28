"""Verify saved crawl artifacts offline without downloading or modifying articles."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from crawl_cafef_health import article_url, health_decision, load_rows, normalized, shingles


def verify(directory):
    articles = load_rows(directory / 'articles.jsonl')
    audit = load_rows(directory / 'crawl_audit.jsonl')
    if not articles or not audit:
        raise ValueError('Missing articles or crawl audit')
    errors = []
    for field in ('id', 'url', 'source_sha256'):
        if len({r[field] for r in articles}) != len(articles):
            errors.append('Duplicate ' + field)
    if len({r['requested_url'] for r in audit}) != len(audit):
        errors.append('Duplicate audited URL')
    accepted_urls = {r['canonical_url'] for r in audit if r['accepted']}
    if accepted_urls != {r['url'] for r in articles}:
        errors.append('Accepted audit URLs differ from dataset URLs')
    if len(audit) > 100:
        errors.append('Exceeded authorized 100 article attempts')
    prior = []
    for row in articles:
        if row['category'] != 'suc-khoe' or article_url(row['url']) != row['url']:
            errors.append(row['id'] + ': unexpected category/URL')
        if row['src_word_count'] != len(row['source'].split()):
            errors.append(row['id'] + ': incorrect word count')
        digest = hashlib.sha256(normalized(row['source']).casefold().encode()).hexdigest()
        if digest != row['source_sha256']:
            errors.append(row['id'] + ': content hash mismatch')
        if not health_decision(row)[0]:
            errors.append(row['id'] + ': fails health filter')
        if not row['pub_date']:
            errors.append(row['id'] + ': missing publication date')
        else:
            try:
                datetime.fromisoformat(row['pub_date'])
            except ValueError:
                errors.append(row['id'] + ': invalid publication date')
        if '\ufffd' in row['source'] or '<script' in row['source'].casefold():
            errors.append(row['id'] + ': encoding or HTML residue')
        if 'summary' in row:
            errors.append(row['id'] + ': unexpected summary label')
        tokens = shingles(row['source'])
        for other_id, previous in prior:
            union = tokens | previous
            if union and len(tokens & previous) / len(union) >= .85:
                errors.append(row['id'] + ': near duplicate of ' + other_id)
        prior.append((row['id'], tokens))
    return dict(verified_at=datetime.now(timezone.utc).isoformat(), passed=not errors,
                articles=len(articles), audited_attempts=len(audit),
                rejected=dict(Counter(r['reason'] for r in audit if not r['accepted'])),
                total_source_words=sum(r['src_word_count'] for r in articles),
                dataset_bytes=(directory / 'articles.jsonl').stat().st_size,
                errors=errors,
                scope='JSON integrity, provenance, filters and duplicate checks; not medical fact verification or summary evaluation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('data/cafef_health'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        # Never overwrite an existing report or user data.
        with args.report.open('x', encoding='utf-8') as output:
            output.write(payload + '\n')
    print(payload)
    raise SystemExit(0 if result['passed'] else 1)
