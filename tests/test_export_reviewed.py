import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_reviewed_splits import export


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


class ExportReviewedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.prepared = self.root / 'prepared'
        source = ' '.join(f'từ{i}' for i in range(98)) + ' liều 5 mg'
        self.row = dict(id='a', split='train', source=source, src_word_count=len(source.split()),
                        prepared_source_sha256='h', summary='draft')
        for split in ('train', 'val', 'test'):
            write(self.prepared / 'draft_splits' / split / 'dataset.jsonl', [self.row] if split == 'train' else [])
        self.good = ' '.join(f'từ{i}' for i in range(21)) + ' liều 5'
        self.annotation = dict(id='a', source_sha256='h', summary=self.good, approved=True, reviewer='Lan', split='train')

    def tearDown(self):
        self.tmp.cleanup()

    def run_export(self, annotation, name='out'):
        path = self.root / 'annotations.jsonl'
        write(path, [annotation])
        return export(self.prepared, path, self.root / name)

    def test_exports_valid_approved_annotation(self):
        report = self.run_export(self.annotation)
        self.assertEqual(report['reviewed_counts'], {'train': 1, 'val': 0, 'test': 0})
        self.assertFalse(report['all_splits_nonempty'])
        out = (self.root / 'out/reviewed_splits/train/dataset.jsonl').read_text(encoding='utf-8')
        self.assertEqual(json.loads(out)['label_status'], 'human_reviewed')

    def test_unapproved_is_skipped_not_promoted(self):
        report = self.run_export({**self.annotation, 'approved': False})
        self.assertEqual(report['reviewed_counts']['train'], 0)
        self.assertEqual(report['skipped'], {'not_approved': 1})

    def test_rejects_invalid_approvals(self):
        cases = [{'source_sha256': 'wrong'}, {'reviewer': ' '}, {'split': 'test'},
                 {'summary': 'quá ngắn'}, {'summary': self.good + ' 99'}, {'id': 'missing'}]
        for i, change in enumerate(cases):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_export({**self.annotation, **change}, name=f'bad{i}')
            self.assertFalse((self.root / f'bad{i}').exists())

    def test_never_overwrites(self):
        self.run_export(self.annotation)
        with self.assertRaises(FileExistsError):
            self.run_export(self.annotation)


if __name__ == '__main__':
    unittest.main()
