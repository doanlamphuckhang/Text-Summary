"""Build notebook-ready train/val/test files from a prepared directory.

Steps: strip bylines/credits, align the source with the model's input window (cut at a sentence
boundary), redraft the extractive label with a stronger scorer, then gate on length and quality.
Splits and group IDs are inherited from the prepared manifest; nothing is re-split.

Labels remain `extractive_pseudo` / `pending_human_review`. Scores on them are not evidence of real
summary quality. The sapo is never used for scoring, only as an independent weak agreement proxy.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summ_rl.text import numbers, rouge_scores, tokens  # noqa: E402
from prepare_health_data import STOPWORDS, load_jsonl, save_jsonl  # noqa: E402
from prepare_health_multisource import sentence_spans  # noqa: E402

CREDIT = re.compile(r'^(?:Nguồn|Ảnh(?: minh họa)?|Video|Đồ họa|Biên tập|Tác giả|Bài|Clip)\s*[:\-–]', re.I)
DANGLING = re.compile(r'^(?:Điều này|Do đó|Vì vậy|Ngoài ra|Bên cạnh đó|Tuy nhiên|Theo đó|Như vậy|Chính vì|Cũng theo|Đồng thời|Trong khi đó)\b')
SPLITS = ('train', 'val', 'test')


def clean_paragraphs(source):
    kept = []
    for paragraph in source.split('\n\n'):
        text = paragraph.strip()
        words = text.split()
        if not text or CREDIT.match(text):
            continue
        if len(words) <= 3 and not re.search(r'[.!?…:]$', text):
            continue  # byline or stray heading
        kept.append(text)
    return '\n\n'.join(kept)


def fit_window(source, max_words):
    """Cut `source` after the last whole sentence that keeps it within max_words."""
    if len(source.split()) <= max_words:
        return source, False
    end, used = 0, 0
    for start, stop in sentence_spans(source):
        used += len(source[start:stop].split())
        if used > max_words:
            break
        end = stop
    return (source[:end] if end else source), True


def score_sentences(spans, source, title, cfg):
    sentences = [source[a:b] for a, b in spans]
    bags = [set(tokens(s)) - STOPWORDS for s in sentences]
    frequency = Counter(t for bag in bags for t in bag)
    title_bag = set(tokens(title)) - STOPWORDS
    has_numbers = bool(numbers(source))
    scores = []
    for i, (sentence, bag) in enumerate(zip(sentences, bags)):
        words = max(1, len(sentence.split()))
        salience = sum(math.log1p(frequency[t]) for t in bag) / math.sqrt(words)
        score = salience
        score += cfg['title'] * len(bag & title_bag) / max(1, len(title_bag))
        score += cfg['lead'] / math.sqrt(i + 1)
        if has_numbers and numbers(sentence):
            score += cfg['numbers']
        if DANGLING.match(sentence):
            score -= cfg['dangling']
        if sentence.rstrip().endswith('?'):
            score -= cfg['question']
        scores.append(score)
    return sentences, bags, scores


def redraft(source, title, cfg, lower=.20, upper=.25):
    spans = sentence_spans(source)
    sentences, bags, base = score_sentences(spans, source, title, cfg)
    total = len(source.split())
    low, high = math.ceil(lower * total), math.floor(upper * total)
    selected, used = [], 0
    while used < low:
        options = [i for i in range(len(sentences)) if i not in selected and len(sentences[i].split()) >= 6
                   and used + len(sentences[i].split()) <= high]
        if not options:
            break

        def mmr(i):
            overlap = max((len(bags[i] & bags[j]) / max(1, len(bags[i] | bags[j])) for j in selected), default=0.)
            return base[i] * (1 - cfg['redundancy'] * overlap)

        best = max(options, key=mmr)
        selected.append(best)
        used += len(sentences[best].split())
    evidence = [dict(start=spans[i][0], end=spans[i][1], text=sentences[i]) for i in sorted(selected)]
    return ' '.join(e['text'] for e in evidence), evidence


OLD = dict(title=0., lead=0., numbers=0., dangling=0., question=0., redundancy=.7)
NEW = dict(title=2., lead=1., numbers=.4, dangling=1., question=1., redundancy=.7)


def build_row(row, max_words, cfg):
    cleaned = clean_paragraphs(row['source'])
    source, truncated = fit_window(cleaned, max_words)
    summary, evidence = redraft(source, row['title'], cfg)
    words = len(source.split())
    ratio = len(summary.split()) / words if words else 0.
    flags = [f for f, hit in [('truncated_to_input_window', truncated), ('draft_length', not .2 <= ratio <= .25),
                              ('short_source', words < 250)] if hit]
    return {**{k: row[k] for k in ('id', 'url', 'title', 'publisher', 'pub_date', 'group_id', 'split', 'sapo') if k in row},
            'category': row.get('category', 'suc-khoe'), 'source': source, 'summary': summary,
            'evidence_spans': evidence, 'src_word_count': words, 'tgt_word_count': len(summary.split()),
            'length_ratio': ratio, 'src_numbers': sorted(numbers(source)), 'tgt_numbers': sorted(numbers(summary)),
            'label_type': 'extractive_pseudo', 'label_status': 'pending_human_review', 'quality_tag': 'pseudo',
            'review_flags': row.get('review_flags', []) + flags, 'original_src_word_count': row['src_word_count']}


def agreement(rows):
    """Mean ROUGE-1/L F of the draft against the publisher sapo (never used for scoring)."""
    scored = [rouge_scores(r['summary'], r['sapo']) for r in rows if r.get('sapo')]
    return dict(rouge1=round(100 * statistics.mean(s['rouge1'] for s in scored), 2),
                rougeL=round(100 * statistics.mean(s['rougeL'] for s in scored), 2), n=len(scored))


def build(prepared, output, max_words):
    if output.exists():
        raise FileExistsError('Use a fresh output directory; existing datasets are never overwritten')
    source_rows = [r for split in SPLITS for r in load_jsonl(prepared / 'draft_splits' / split / 'dataset.jsonl')]
    old_view = [build_row(r, 10 ** 9, OLD) for r in source_rows]
    rows = [build_row(r, max_words, NEW) for r in source_rows]
    kept, dropped = [], Counter()
    for row in rows:
        if 'draft_length' in row['review_flags'] or 'short_source' in row['review_flags'] or not row['summary']:
            dropped['bad_draft_length_or_short_source'] += 1
        elif row['category'] and row['label_type'] != 'extractive_pseudo':
            dropped['label_type'] += 1
        else:
            kept.append(row)
    ids = {r['id'] for r in kept}
    for split in SPLITS:
        save_jsonl(output / split / 'dataset.jsonl', [r for r in kept if r['split'] == split])
    groups = {}
    for r in kept:
        groups.setdefault(r['group_id'], set()).add(r['split'])
    assert all(len(s) == 1 for s in groups.values()), 'a duplicate group crosses splits'
    report = dict(prepared=str(prepared), max_source_words=max_words, input_records=len(source_rows),
                  kept=len(kept), dropped=dict(dropped),
                  split_counts=dict(Counter(r['split'] for r in kept)),
                  truncated_fraction=round(sum('truncated_to_input_window' in r['review_flags'] for r in kept) / len(kept), 3),
                  src_words=dict(mean=round(statistics.mean(r['src_word_count'] for r in kept), 1),
                                 median=statistics.median(r['src_word_count'] for r in kept)),
                  length_ratio=dict(mean=round(statistics.mean(r['length_ratio'] for r in kept), 4)),
                  sapo_agreement_old_draft_full_source=agreement([r for r in old_view if r['id'] in ids]),
                  sapo_agreement_new_draft=agreement(kept),
                  note='Agreement with the publisher sapo is a weak proxy, not summary quality. Labels are unreviewed; '
                       'confirm the word-to-token ratio with check_token_lengths.py before relying on max_source_words.')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('data/health_multisource_prepared_v1'))
    parser.add_argument('--output', type=Path, default=Path('data/health_training_v1'))
    parser.add_argument('--max-source-words', type=int, default=600,
                        help='about 1024 ViT5 tokens for Vietnamese at ~1.7 tokens/word; verify with a tokenizer')
    args = parser.parse_args()
    build(args.prepared, args.output, args.max_source_words)
