"""Offline preprocessing, grouped splits and explicitly unreviewed extractive drafts.

No model downloads, training, automatic label approval or changes to input data.
An approved annotation file can export reviewed pairs on the same split manifest.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import unicodedata

from crawl_cafef_health import health_decision

NUMBERS = re.compile(r'\d+(?:[.,]\d+)*')
STOPWORDS = set('và là của có các một những được cho trong với này đó khi để không cũng người từ về ở đã sẽ như thì đến hơn theo nhưng rất việc nên vì bằng vào ra trên còn lại mỗi nhiều'.split())


def clean_text(value, paragraphs=False):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected nonempty text')
    text = unicodedata.normalize('NFC', value).replace('\u200b', '').replace('\ufeff', '')
    if any(ord(c) < 32 and c not in '\r\n\t' for c in text) or '\ufffd' in text:
        raise ValueError('Invalid control/replacement characters')
    if re.search(r'</?(?:script|iframe|div|p)\b', text, re.I):
        raise ValueError('HTML remains in text')
    if paragraphs:
        return '\n\n'.join(' '.join(p.split()) for p in re.split(r'\n\s*\n', text) if p.strip())
    return ' '.join(text.split())


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def tokens(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text).casefold())


def ngrams(text, n=5):
    words = tokens(text)
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a or b else 0.


def make_groups(rows):
    """Connected components prevent transitive near-duplicates crossing splits."""
    rows = sorted(rows, key=lambda r: r['id'])
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    body = [ngrams(r['source']) for r in rows]
    title = [set(tokens(r['title'])) for r in rows]
    links = []
    for i in range(len(rows)):
        for k in range(i):
            similarity = jaccard(body[i], body[k])
            same = rows[i]['url'] == rows[k]['url'] or rows[i]['content_key'] == rows[k]['content_key']
            similar_title = len(title[i]) >= 5 and len(title[k]) >= 5 and jaccard(title[i], title[k]) >= .75
            if same or similarity >= .45 or (similar_title and similarity >= .15):
                parent[root(i)] = root(k)
                links.append(dict(a=rows[i]['id'], b=rows[k]['id'], body_jaccard=round(similarity, 4),
                                  reason='exact' if same else 'body' if similarity >= .45 else 'title_and_body'))
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(root(i), []).append(row)
    return list(groups.values()), links


def split_groups(groups, seed=42):
    if len(groups) < 3:
        raise ValueError('At least three independent groups needed for three splits')
    ordered = sorted(groups, key=lambda group: min(r['id'] for r in group))
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    total = sum(map(len, ordered))
    targets = dict(train=.8 * total, val=.1 * total, test=.1 * total)
    output = {key: [] for key in targets}
    for group in ordered:
        split = min(output, key=lambda key: len(output[key]) / targets[key])
        group_id = digest('|'.join(sorted(r['id'] for r in group)))[:16]
        output[split].extend({**row, 'group_id': group_id, 'split': split} for row in group)
    return output


def sentences(source):
    # Split at sentence-ending punctuation, never inside decimal numbers.
    return [clean_text(part) for paragraph in source.split('\n\n')
            for part in re.split(r'(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ“"(])', paragraph) if part.strip()]


def extractive_draft(source, lower=.20, upper=.25):
    candidates = sentences(source)
    counts = [len(s.split()) for s in candidates]
    word_limit = math.floor(len(source.split()) * upper)
    min_words = math.ceil(len(source.split()) * lower)
    bags = [set(tokens(s)) - STOPWORDS for s in candidates]
    freq = Counter(word for bag in bags for word in bag)
    selected, used = [], 0
    while used < min_words:
        possible = [i for i in range(len(candidates)) if i not in selected
                    and counts[i] >= 8 and used + counts[i] <= word_limit]
        if not possible:
            break

        def score(i):
            salience = sum(math.log1p(freq[word]) for word in bags[i]) / math.sqrt(max(counts[i], 1))
            redundancy = max((jaccard(bags[i], bags[j]) for j in selected), default=0.)
            return salience * (1 - .7 * redundancy) + .2 / (i + 1)

        chosen = max(possible, key=score)
        selected.append(chosen)
        used += counts[chosen]
    selected.sort()
    summary = ' '.join(candidates[i] for i in selected)
    return summary, [candidates[i] for i in selected]


def load_jsonl(path):
    with path.open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def distribution(values):
    return dict(min=min(values), median=statistics.median(values), mean=round(statistics.mean(values), 4), max=max(values)) if values else {}


def prepare(input_path, output, annotations=None):
    if output.exists():
        raise FileExistsError('Use a fresh output directory; existing datasets are never overwritten')
    if shutil.disk_usage(output.parent).free < 100 * 1024 * 1024:
        raise RuntimeError('Need at least 100 MiB free')
    original_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    raw = load_jsonl(input_path)
    if len(raw) > 1000:
        raise ValueError('This small-corpus pairwise preprocessor is limited to 1000 records')
    rows, rejected, seen = [], [], set()
    for row in raw:
        try:
            identity = row['id']
            if identity in seen:
                raise ValueError('Duplicate input ID')
            seen.add(identity)
            source, title = clean_text(row['source'], True), clean_text(row['title'])
            sapo = clean_text(row['sapo']) if isinstance(row.get('sapo'), str) and row['sapo'].strip() else ''
            date = datetime.fromisoformat(row['pub_date'])
            if date.tzinfo is None:
                raise ValueError('Publication date must include timezone')
            normalized_row = {**row, 'source': source, 'title': title, 'sapo': sapo,
                              'src_word_count': len(source.split()), 'prepared_source_sha256': digest(source),
                              'content_key': digest(' '.join(source.casefold().split()))}
            keep, reason, _ = health_decision(normalized_row)
            if not keep:
                raise ValueError(reason)
            rows.append(normalized_row)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(dict(id=row.get('id'), reason=str(error)))
    groups, links = make_groups(rows)
    splits = split_groups(groups)
    all_rows = [row for split in splits.values() for row in split]
    annotations_by_id = {}
    if annotations:
        for item in load_jsonl(annotations):
            if item['id'] in annotations_by_id:
                raise ValueError('Duplicate annotation ID')
            annotations_by_id[item['id']] = item
        if set(annotations_by_id) - {r['id'] for r in all_rows}:
            raise ValueError('Annotation ID is not in prepared corpus')
    prepared, review, ready = [], [], {split: [] for split in splits}
    flags = Counter()
    for row in all_rows:
        summary, evidence = extractive_draft(row['source'])
        count = len(summary.split())
        ratio = count / row['src_word_count']
        warnings = []
        if not .20 <= ratio <= .25:
            warnings.append('draft_length_outside_20_25_percent')
        if not summary:
            warnings.append('empty_draft')
        if any(s not in row['source'] for s in evidence):
            raise AssertionError('Draft evidence must be verbatim from source')
        missing_sapo_numbers = sorted(set(NUMBERS.findall(row['sapo'])) - set(NUMBERS.findall(row['source'])))
        if missing_sapo_numbers:
            warnings.append('sapo_numbers_not_found_in_body')
        flags.update(warnings)
        draft = {**row, 'summary': summary, 'tgt_word_count': count, 'length_ratio': ratio,
                 'label_type': 'extractive_pseudo', 'label_status': 'pending_human_review',
                 'quality_tag': 'unreviewed', 'evidence_sentences': evidence, 'review_flags': warnings,
                 'sapo_numbers_not_found_in_body': missing_sapo_numbers}
        prepared.append(draft)
        review.append(dict(id=row['id'], split=row['split'], source_sha256=row['prepared_source_sha256'],
                           title=row['title'], url=row['url'], source=row['source'],
                           summary=summary, approved=False, reviewer='', notes='', review_flags=warnings))
        annotation = annotations_by_id.get(row['id'])
        if annotation and annotation.get('approved') is True:
            if annotation.get('source_sha256') != row['prepared_source_sha256']:
                raise ValueError('Annotation source hash mismatch: ' + row['id'])
            if not isinstance(annotation.get('reviewer'), str) or not annotation['reviewer'].strip():
                raise ValueError('Approved annotation must name its reviewer')
            approved_summary = clean_text(annotation['summary'])
            approved_ratio = len(approved_summary.split()) / row['src_word_count']
            if not .20 <= approved_ratio <= .25:
                raise ValueError('Approved summary must meet agreed 20-25% length target: ' + row['id'])
            if set(NUMBERS.findall(approved_summary)) - set(NUMBERS.findall(row['source'])):
                raise ValueError('Approved summary contains numbers absent from source: ' + row['id'])
            ready[row['split']].append({**row, 'summary': approved_summary, 'length_ratio': approved_ratio,
                'tgt_word_count': len(approved_summary.split()), 'quality_tag': 'human_reviewed',
                'label_status': 'human_reviewed', 'reviewer': annotation['reviewer']})
    for a in splits:
        for b in splits:
            if a != b:
                assert not {r['group_id'] for r in splits[a]} & {r['group_id'] for r in splits[b]}
    output.mkdir()
    save_jsonl(output / 'clean_articles.jsonl', rows)
    save_jsonl(output / 'rejected.jsonl', rejected)
    save_jsonl(output / 'review_queue.jsonl', review)
    save_jsonl(output / 'group_links.jsonl', links)
    save_jsonl(output / 'split_manifest.jsonl', [dict(id=r['id'], split=r['split'], group_id=r['group_id'],
                source_sha256=r['prepared_source_sha256']) for r in all_rows])
    for split in splits:
        save_jsonl(output / 'draft_splits' / split / 'dataset.jsonl', [r for r in prepared if r['split'] == split])
        if annotations:
            save_jsonl(output / 'reviewed_splits' / split / 'dataset.jsonl', ready[split])
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), input_sha256=original_hash,
        input_records=len(raw), retained_records=len(rows), rejected_records=len(rejected),
        groups=len(groups), near_duplicate_links=len(links), seed=42,
        split_strategy='seeded grouped 80/10/10 approximate; grouping is heuristic, not semantic proof',
        split_counts={k: len(v) for k, v in splits.items()},
        reviewed_counts={k: len(v) for k, v in ready.items()}, training_ready=all(ready.values()),
        source_words=distribution([r['src_word_count'] for r in rows]),
        sapo_words=distribution([len(r['sapo'].split()) for r in rows]),
        sapo_ratio=distribution([len(r['sapo'].split()) / r['src_word_count'] for r in rows]),
        draft_ratio=distribution([r['length_ratio'] for r in prepared]), review_flags=dict(flags),
        date_years=dict(sorted(Counter(r['pub_date'][:4] for r in rows).items())),
        topic_evidence=dict(Counter(term for r in rows for term in r.get('filter_evidence', [])).most_common()),
        tokenizer_checked=False, baseline_commit='0195b5bd81753bb2b4a4b2e61acecc41bd908dee',
        method='document-local frequency extractive draft; no corpus-fitted statistics or pretrained models',
        limitations=['No medically validated gold labels', 'Word counts are not model token counts',
                     'Single outlet and nonrandom crawl sample', 'No training or test score claims'])
    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == original_hash
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/cafef_health/articles.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--annotations', type=Path)
    args = parser.parse_args()
    prepare(args.input, args.output, args.annotations)
