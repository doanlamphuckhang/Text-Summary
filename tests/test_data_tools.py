"""Offline tests for the token-length report, the review page builder and the multisource verifier."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_health_review import build as build_review
from check_token_lengths import length_report, percentile
from prepare_health_multisource import digest, prepare
from verify_health_multisource import verify


def fixture_corpus(path):
    rows = []
    for i in range(3):
        body = '\n\n'.join('Bác sĩ giải thích sức khỏe và dinh dưỡng cần được theo dõi. '
                           + ' '.join(f'chủđề{i}từ{j}mục{k}' for j in range(70)) + '.' for k in range(5))
        rows.append(dict(id=str(i), url=f'https://example.org/{i}', source=body, title=f'Sức khỏe {i}',
                         pub_date='2025-01-01T00:00:00+07:00', publisher='fixture', sapo=''))
    source = path / 'input.jsonl'
    source.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    frozen = path / 'frozen.jsonl'
    frozen.write_text(json.dumps(dict(id='0', split='test', source_sha256=digest(rows[0]['source']))) + '\n',
                      encoding='utf-8')
    return source, frozen


class TokenLengthTests(unittest.TestCase):
    def test_suggestion_pools_every_split_and_uses_p95(self):
        # One token per character: train rows are token-light, test rows token-heavy.
        rows = {'train': [dict(source='a b c d', summary='a')] * 19,
                'test': [dict(source='abcd efgh', summary='a')]}
        report = length_report(rows, len, max_input=100, safety=1.0)
        self.assertEqual([s['split'] for s in report['splits']], ['train', 'test'])
        # 19 train rows at 7 chars / 4 words = 1.75 and one outlier at 4.5: p95 ignores the single outlier.
        self.assertEqual(report['tokens_per_word_p95'], 1.75)
        self.assertEqual(report['suggested_max_source_words'], int(100 / 1.75))

    def test_empty_split_is_reported_not_crashed(self):
        report = length_report({'train': [dict(source='a b', summary='a')], 'val': []}, len, max_input=10)
        self.assertEqual(report['splits'][1], dict(split='val', n=0))
        with self.assertRaises(ValueError):
            length_report({'train': []}, len)

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([3, 1, 2, 4], 50), 2)
        self.assertEqual(percentile([5], 95), 5)


class PreparedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        source, frozen = fixture_corpus(root)
        self.prepared = root / 'prepared'
        prepare([source], frozen, self.prepared)

    def test_verifier_passes_then_catches_tampering(self):
        self.assertEqual(verify(self.prepared)['errors'], [])
        path = self.prepared / 'draft_splits' / 'test' / 'dataset.jsonl'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['summary'] += ' thêm chữ'
        path.write_text(json.dumps(row, ensure_ascii=False) + '\n', encoding='utf-8')
        result = verify(self.prepared)
        self.assertFalse(result['passed'])
        self.assertTrue(any('summary' in e for e in result['errors']))

    def test_review_page_embeds_escaped_payload_and_never_overwrites(self):
        info = build_review(self.prepared)
        page = (self.prepared / 'review.html').read_text(encoding='utf-8')
        self.assertEqual(info['review_records'], 3)
        payload = page.split('<script id="data" type="application/json">', 1)[1].split('</script>', 1)[0]
        self.assertNotIn('<', payload)
        self.assertEqual(len(json.loads(payload)), 3)
        with self.assertRaises(FileExistsError):
            build_review(self.prepared)


if __name__ == '__main__':
    unittest.main()
