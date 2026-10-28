from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_health_data import clean_text, digest, extractive_draft, make_groups, prepare, split_groups


class PreprocessingTests(unittest.TestCase):
    def test_normalization_keeps_numbers_and_paragraphs(self):
        text = 'Be\u0323̂nh nhân dùng 2,5 mg.\n\n  Huyết áp 120/80. '
        result = clean_text(text, paragraphs=True)
        self.assertIn('2,5 mg', result)
        self.assertIn('\n\n', result)
        self.assertIn('120/80', result)
        with self.assertRaises(ValueError):
            clean_text(None)
        with self.assertRaises(ValueError):
            clean_text('bad\ufffdtext')

    def test_draft_contains_only_original_sentences_and_bounded_length(self):
        source = ' '.join(f'Bác sĩ nghiên cứu dinh dưỡng và bệnh nhân nhóm {i} cần theo dõi sức khỏe.' for i in range(15))
        summary, evidence = extractive_draft(source)
        self.assertTrue(summary)
        self.assertTrue(all(sentence in source for sentence in evidence))
        self.assertEqual(summary, ' '.join(evidence))
        self.assertLessEqual(len(summary.split()) / len(source.split()), .25)
        self.assertGreaterEqual(len(summary.split()) / len(source.split()), .20)

    def test_exact_duplicates_stay_grouped(self):
        rows = [dict(id=str(i), title='title ' + str(i), url='url' + str(i),
                     source='word ' * 40 if i < 2 else 'different' + str(i),
                     content_key='same' if i < 2 else str(i)) for i in range(5)]
        groups, links = make_groups(rows)
        splits = split_groups(groups)
        assignment = {r['id']: split for split, items in splits.items() for r in items}
        self.assertEqual(assignment['0'], assignment['1'])
        self.assertTrue(links)
        self.assertEqual(splits, split_groups(make_groups(list(reversed(rows)))[0]))

    def test_prepare_never_promotes_drafts_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / 'articles.jsonl'
            rows = []
            for i in range(6):
                sentence = 'Bác sĩ hỗ trợ bệnh nhân qua dinh dưỡng ' + ' '.join(f'topic{i}word{j}' for j in range(22)) + '.'
                body = ' '.join([sentence] * 12)
                rows.append(dict(id=str(i), source=body, title='Chăm sóc ' + str(i),
                                 url='url' + str(i), pub_date='2026-01-01T00:00:00+07:00',
                                 category='suc-khoe', sapo='Bác sĩ hướng dẫn.', filter_evidence=[]))
            path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
            before = path.read_bytes()
            out = folder / 'prepared'
            with redirect_stdout(io.StringIO()):
                report = prepare(path, out)
            self.assertFalse(report['training_ready'])
            self.assertEqual(sum(report['split_counts'].values()), 6)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((out / 'reviewed_splits').exists())
            with self.assertRaises(FileExistsError):
                prepare(path, out)
            queue = [json.loads(line) for line in (out / 'review_queue.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all(not item['approved'] for item in queue))
            queue[0].update(approved=True, reviewer='test', source_sha256='wrong')
            annotation_path = folder / 'annotations.jsonl'
            annotation_path.write_text(json.dumps(queue[0]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                prepare(path, folder / 'should_not_export', annotation_path)
            self.assertFalse((folder / 'should_not_export').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
