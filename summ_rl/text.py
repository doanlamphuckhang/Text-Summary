"""Unicode-aware tokenization and light-weight ROUGE used by the RL reward and evaluation."""
import re
import unicodedata

NUM_RE = re.compile(r'\d+(?:[.,]\d+)*')


def tokens(text):
    """NFC + casefold, split into letter/digit runs (keeps Vietnamese diacritics)."""
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', str(text)).casefold())


def numbers(text):
    return set(NUM_RE.findall(str(text or '')))


def _ngrams(toks, n):
    counts = {}
    for i in range(len(toks) - n + 1):
        key = tuple(toks[i:i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _f1(overlap, pred_total, ref_total):
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    p, r = overlap / pred_total, overlap / ref_total
    return 2 * p * r / (p + r)


def rouge_n(pred, ref, n):
    a, b = _ngrams(pred, n), _ngrams(ref, n)
    overlap = sum(min(c, b.get(k, 0)) for k, c in a.items())
    return _f1(overlap, sum(a.values()), sum(b.values()))


def lcs_length(a, b):
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(pred, ref):
    return _f1(lcs_length(pred, ref), len(pred), len(ref))


def rouge_scores(pred_text, ref_text):
    pred, ref = tokens(pred_text), tokens(ref_text)
    return dict(rouge1=rouge_n(pred, ref, 1), rouge2=rouge_n(pred, ref, 2), rougeL=rouge_l(pred, ref))


def distinct_ngram_ratio(toks, n=3):
    grams = _ngrams(toks, n)
    total = sum(grams.values())
    return len(grams) / total if total else 0.0
