"""Auditable offline cleaning, duplicate grouping and frozen evaluation splits.

Extractive drafts remain unreviewed. This does not certify medical correctness.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from prepare_health_data import clean_text, digest, tokens, STOPWORDS, load_jsonl, save_jsonl, distribution
from crawl_health_multisource import quality

def hashed_ngrams(text):
    words = tokens(text)
    return {hashlib.blake2b(' '.join(words[i:i+5]).encode(), digest_size=8).digest()
            for i in range(len(words)-4)}

def group_rows(rows):
    """Exact candidate counts through an inverted index; avoids quadratic set unions."""
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    sizes, titles, index, links = [], [], defaultdict(list), []
    for i, row in enumerate(rows):
        grams = hashed_ngrams(row['source'])
        title = set(tokens(row['title']))
        counts = Counter(j for gram in grams for j in index.get(gram, ()))
        for j, overlap in counts.items():
            score = overlap / (len(grams) + sizes[j] - overlap)
            title_score = len(title & titles[j]) / max(1, len(title | titles[j]))
            if score >= .35 or (title_score >= .65 and score >= .12):
                parent[root(i)] = root(j)
                links.append(dict(a=rows[j]['id'], b=row['id'], body_jaccard=score, title_jaccard=title_score))
        for gram in grams:
            index[gram].append(i)
        sizes.append(len(grams))
        titles.append(title)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    return list(groups.values()), links

def sentence_spans(source):
    """Retain exact offsets, decimal points, abbreviations and original punctuation."""
    spans = []
    for paragraph in re.finditer(r'[^\n]+', source):
        start = paragraph.start()
        for boundary in re.finditer(r'(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ“"(])', paragraph.group()):
            end = paragraph.start() + boundary.start()
            prefix = source[start:end]
            if re.search(r'(?:BS|TS|ThS|PGS|GS|Dr|Mr|Mrs|TP|TPHCM)\.$', prefix) or prefix.count('"') % 2:
                continue
            if end > start:
                spans.append((start, end))
            start = paragraph.start() + boundary.end()
        if paragraph.end() > start:
            spans.append((start, paragraph.end()))
    return spans

def draft(row):
    source = row['source']
    spans = sentence_spans(source)
    sentences = [source[a:b] for a,b in spans]
    bags = [set(tokens(s)) - STOPWORDS for s in sentences]
    frequency = Counter(t for bag in bags for t in bag)
    title = set(tokens(row['title'])) - STOPWORDS
    selected, used = [], 0
    minimum, maximum = math.ceil(.2 * row['src_word_count']), math.floor(.25 * row['src_word_count'])
    while used < minimum:
        possible = [i for i,s in enumerate(sentences) if i not in selected and len(s.split()) >= 8 and used + len(s.split()) <= maximum]
        if not possible:
            break
        def score(i):
            redundancy = max((len(bags[i] & bags[j])/max(1,len(bags[i]|bags[j])) for j in selected), default=0)
            salience = sum(math.log1p(frequency[t]) for t in bags[i]) / math.sqrt(len(sentences[i].split()))
            return salience*(1-.7*redundancy) + 2*len(bags[i]&title)/max(1,len(title)) + .5/(i+1)
        i = max(possible, key=score)
        selected.append(i)
        used += len(sentences[i].split())
    evidence = [dict(start=spans[i][0], end=spans[i][1], text=sentences[i]) for i in sorted(selected)]
    return ' '.join(e['text'] for e in evidence), evidence

def prepare(inputs, frozen, output):
    if output.exists():
        raise FileExistsError('Fresh version directory required')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError('Need 1 GiB free')
    input_hashes = {str(p): digest(p.read_text(encoding='utf-8')) for p in inputs}
    raw = [r for p in inputs for r in load_jsonl(p)]
    if len(raw) > 1000:
        raise ValueError('Scope limited to 1000 articles')
    old = {r['id']: r for r in load_jsonl(frozen)}
    rows, rejected, seen, urls, contents = [], [], set(), set(), {}
    for r in raw:
        try:
            r = dict(r)
            r['source'] = clean_text(r['source'], True)
            r['title'] = clean_text(r['title'])
            r['src_word_count'] = len(r['source'].split())
            r['prepared_source_sha256'] = digest(r['source'])
            r['content_key'] = digest(' '.join(tokens(r['source'])))
            authors = []
            for author in r.get('authors', []):
                words = clean_text(author).split()
                half = len(words) // 2
                if len(words) % 2 == 0 and words[:half] == words[half:]:
                    words = words[:half]
                value = ' '.join(words)
                if value not in authors:
                    authors.append(value)
            r['authors'] = authors
            if r['id'] in seen or r['url'] in urls or r['content_key'] in contents:
                raise ValueError('exact_duplicate')
            seen.add(r['id'])
            urls.add(r['url'])
            contents[r['content_key']] = r['id']
            date = datetime.fromisoformat(r['pub_date'])
            if date.tzinfo is None or date > datetime.now(timezone.utc):
                raise ValueError('invalid_date')
            keep, why, _ = quality(r)
            if not keep:
                raise ValueError(why)
            if r['id'] in old and old[r['id']]['source_sha256'] != r['prepared_source_sha256']:
                raise ValueError('frozen_source_changed')
            rows.append(r)
        except (ValueError, KeyError, TypeError) as error:
            rejected.append(dict(id=r.get('id'), url=r.get('url'), reason=str(error)))
    groups, links = group_rows(rows)
    near = {link['b'] for link in links if link['body_jaccard'] >= .85}
    rejected.extend(dict(id=r['id'], url=r['url'], reason='near_duplicate_body_85_percent') for r in rows if r['id'] in near)
    groups = [[r for r in group if r['id'] not in near] for group in groups]
    groups = [group for group in groups if group]
    # Conservatively quarantine groups that would join two existing evaluation partitions.
    counts = Counter()
    accepted = []
    pending = []
    for group in groups:
        previous = {old[r['id']]['split'] for r in group if r['id'] in old}
        if len(previous) > 1:
            rejected.extend(dict(id=r['id'], url=r['url'], reason='frozen_split_conflict') for r in group)
            continue
        group_id = digest('|'.join(sorted(r['id'] for r in group)))[:16]
        if previous:
            split = next(iter(previous))
            accepted.extend({**r, 'group_id': group_id, 'split': split} for r in group)
            counts[split] += len(group)
        else:
            pending.append((group_id, group))
    for group_id, group in sorted(pending):
        split = min(('train','val','test'), key=lambda s: counts[s] / {'train':.8,'val':.1,'test':.1}[s])
        accepted.extend({**r, 'group_id': group_id, 'split': split} for r in group)
        counts[split] += len(group)
    # Near copies remain in raw input and audit, outside the prepared corpus.
    drafts, queue, warnings = [], [], Counter()
    for r in accepted:
        summary, evidence = draft(r)
        ratio = len(summary.split()) / r['src_word_count']
        flags = []
        if not .2 <= ratio <= .25:
            flags.append('draft_length')
        if not r.get('authors'):
            flags.append('author_unavailable')
        if not r.get('external_reference_urls'):
            flags.append('no_external_reference_link')
        if re.search(r'\b(?:bí quyết|thần dược|sống thọ|trường thọ|117 tuổi|chữa khỏi)\b', r['title'], re.I):
            flags.append('strong_health_claim_needs_review')
        if '?' in r['title']:
            flags.append('question_headline_check_answer_coverage')
        if any(re.match(r'(?:Điều này|Theo đó|Do đó|Tuy nhiên|Ông|Bà)\b', e['text']) for e in evidence):
            flags.append('check_context_and_attribution')
        caveats = [r['source'][a:b] for a,b in sentence_spans(r['source'])
            if re.search(r'\b(?:tuy nhiên|không nên|không được|chống chỉ định|chỉ áp dụng|không tự ý|cần tham khảo|hạn chế của nghiên cứu)\b', r['source'][a:b], re.I)]
        omitted_caveats = [s for s in caveats if s not in summary]
        if omitted_caveats:
            flags.append('omitted_caveat_check_before_approval')
        sapo_numbers = set(re.findall(r'\d+(?:[.,]\d+)*', r.get('sapo','')))
        if sapo_numbers - set(re.findall(r'\d+(?:[.,]\d+)*', r['source'])):
            flags.append('sapo_number_not_in_body')
        warnings.update(flags)
        item = {**r, 'summary': summary, 'evidence_spans': evidence, 'tgt_word_count':len(summary.split()),
            'length_ratio':ratio, 'label_type':'extractive_pseudo', 'label_status':'pending_human_review',
            'quality_tag':'unreviewed', 'review_flags':flags, 'omitted_caveat_candidates':omitted_caveats}
        drafts.append(item)
        queue.append({**item, 'approved':False, 'reviewer':'', 'notes':''})
    output.mkdir()
    save_jsonl(output/'clean_articles.jsonl', accepted)
    save_jsonl(output/'rejected.jsonl', rejected)
    save_jsonl(output/'group_links.jsonl', links)
    save_jsonl(output/'review_queue.jsonl', queue)
    save_jsonl(output/'split_manifest.jsonl', [dict(id=r['id'],split=r['split'],group_id=r['group_id'],source_sha256=r['prepared_source_sha256']) for r in accepted])
    for split in ('train','val','test'):
        save_jsonl(output/'draft_splits'/split/'dataset.jsonl', [r for r in drafts if r['split']==split])
    report = dict(input_records=len(raw), retained=len(accepted), rejected=len(rejected),
        publishers=dict(Counter(r['publisher'] for r in accepted)), split_counts=dict(counts),
        publisher_by_split={s:dict(Counter(r['publisher'] for r in accepted if r['split']==s)) for s in ('train','val','test')},
        publication_years=dict(Counter(r['pub_date'][:4] for r in accepted)),
        topic_keyword_evidence=dict(Counter(t for r in accepted for t in r.get('filter_evidence', []))),
        independent_groups=len({r['group_id'] for r in accepted}), duplicate_links=len(links), near_copy_records=len(near),
        flags=dict(warnings), words=distribution([r['src_word_count'] for r in accepted]),
        draft_ratio=distribution([r['length_ratio'] for r in drafts]), input_hashes=input_hashes,
        frozen_manifest=str(frozen), training_ready=False, human_reviewed=0, tokenizer_checked=False,
        limitations=['Heuristic health and duplicate detection; not medical fact verification',
            'Extractive draft labels require human review of completeness, attribution and safety',
            'News sampling is not population representative; copyright/reuse licenses are not granted by robots.txt'])
    # Explicit checks (not `assert`) so they still run under `python -O`.
    if input_hashes != {str(p): digest(p.read_text(encoding='utf-8')) for p in inputs}:
        raise RuntimeError('An input file changed during preparation')
    split_of_group = {}
    for r in drafts:
        if r['summary'] != ' '.join(e['text'] for e in r['evidence_spans']):
            raise RuntimeError(f"{r['id']}: summary differs from its evidence spans")
        if not all(r['source'][e['start']:e['end']] == e['text'] for e in r['evidence_spans']):
            raise RuntimeError(f"{r['id']}: evidence offsets do not match the source")
        if r['id'] in old and r['split'] != old[r['id']]['split']:
            raise RuntimeError(f"{r['id']}: frozen split changed")
        if split_of_group.setdefault(r['group_id'], r['split']) != r['split']:
            raise RuntimeError(f"group {r['group_id']} crosses splits")
    report['validation_passed'] = True
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+',
                        default=[Path('data/cafef_health/articles.jsonl'),
                                 Path('data/health_multisource_raw_v1/articles.jsonl')])
    parser.add_argument('--frozen', type=Path, default=Path('data/cafef_health_prepared_v1/split_manifest.jsonl'),
                        help='Existing split manifest whose assignments must be preserved')
    args = parser.parse_args()
    prepare(args.inputs, args.frozen, args.output)
