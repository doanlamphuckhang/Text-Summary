import sys
from pathlib import Path
import tempfile
import unittest
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_health_multisource import sentence_spans, draft, group_rows, prepare, digest

class PreparationTests(unittest.TestCase):
    def test_abbreviation_decimal_and_offsets(self):
        source = 'BS. Nguyễn Văn An khuyên không tự ý tăng liều lên 2.5 mg. Người bệnh cần hỏi bác sĩ.\n\nĐây là đoạn khác.'
        pieces = [source[a:b] for a,b in sentence_spans(source)]
        self.assertEqual(len(pieces), 3)
        self.assertTrue(pieces[0].startswith('BS. Nguyễn'))
        self.assertIn('2.5 mg.', pieces[0])

    def test_cross_source_group(self):
        body = ' '.join('token' + str(i) for i in range(100))
        rows = [dict(id='a',source=body,title='Một nghiên cứu về điều trị bệnh'),
                dict(id='b',source=body + ' bổ sung một câu khác',title='Tin mới nghiên cứu'),
                dict(id='c',source='khác hoàn toàn không dùng những từ trên',title='Khác')]
        groups, links = group_rows(rows)
        self.assertEqual(sorted(map(len, groups)), [1,2])
        self.assertEqual(links[0]['a'], 'a')

    def test_draft_preserves_evidence(self):
        source='\n\n'.join(f'Bác sĩ khuyến cáo nhóm {i} không tự ý dùng thuốc vì có thể gây hại cho sức khỏe.' for i in range(20))
        summary,evidence=draft(dict(source=source,title='Không tự ý dùng thuốc',src_word_count=len(source.split())))
        self.assertTrue(summary)
        self.assertTrue(all(source[e['start']:e['end']]==e['text'] for e in evidence))
        self.assertTrue(.2 <= len(summary.split())/len(source.split()) <= .25)

    def test_preserves_frozen_split_and_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            rows=[]
            for i in range(3):
                body='\n\n'.join(f'Bác sĩ giải thích sức khỏe và dinh dưỡng cần được theo dõi. ' + ' '.join(f'chủđề{i}từ{j}mục{k}' for j in range(70)) + '.' for k in range(5))
                rows.append(dict(id=str(i),url='https://example.org/'+str(i),source=body,title='Sức khỏe '+str(i),pub_date='2025-01-01T00:00:00+07:00',publisher='fixture',sapo=''))
            source=path/'input.jsonl'
            source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            before=source.read_bytes()
            frozen=path/'frozen.jsonl'
            frozen.write_text(json.dumps(dict(id='0',split='test',source_sha256=digest(rows[0]['source'])))+'\n',encoding='utf-8')
            report=prepare([source],frozen,path/'prepared')
            self.assertFalse(report['training_ready'])
            self.assertEqual(source.read_bytes(),before)
            manifest=[json.loads(s) for s in (path/'prepared/split_manifest.jsonl').read_text().splitlines()]
            self.assertEqual(next(r['split'] for r in manifest if r['id']=='0'),'test')

if __name__ == '__main__':
    unittest.main()
