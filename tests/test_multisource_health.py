import sys
from pathlib import Path
import gzip
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from crawl_health_multisource import article_url, safe_url, discover, parse_article, quality, Page

class MultiSourceTests(unittest.TestCase):
    def test_omitted_paragraph_end_tags(self):
        root = Page('<div data-role=content><p>Một đoạn thứ nhất.<p>Đoạn thứ hai.<h2>Tiêu đề phụ</h2><p>Đoạn cuối.</div>').root
        paragraphs = [n.text() for n in root.walk() if n.tag == 'p']
        self.assertEqual(paragraphs, ['Một đoạn thứ nhất.', 'Đoạn thứ hai.', 'Đoạn cuối.'])

    def test_scope(self):
        self.assertIsNone(article_url('https://evil.example/a-123456789012.htm', 'tuoitre.vn'))
        with self.assertRaises(ValueError):
            safe_url('https://tuoitre.vn@evil.example/x', 'tuoitre.vn')
        self.assertEqual(discover('<nav><a href="/a-123456789012.htm">a</a></nav>', 'tuoitre.vn'), [])

    def test_real_samples(self):
        if not Path('outputs/source-probes/3df1965f04fd.html').exists():
            self.skipTest('Optional local crawl fixtures unavailable')
        fixtures = [ ('3df1965f04fd.html', 'https://suckhoedoisong.vn/nguoi-tap-luyen-nen-an-thuc-pham-nao-vua-giau-protein-vua-co-chat-xo-16926101414142831.htm'),
                    ('34ee2b67dab6.html', 'https://tuoitre.vn/ai-can-chu-dong-tranh-mui-hoa-sua-100261014175954656.htm') ]
        for name, url in fixtures:
            data = (Path('outputs/source-probes') / name).read_bytes()
            if data.startswith(b'\x1f\x8b'):
                data = gzip.decompress(data)
            row = parse_article(data.decode('utf-8-sig'), url, 'sample')
            self.assertTrue(quality(row)[0], quality(row))
            self.assertGreater(len(row['authors']), 0)
            self.assertNotIn('ĐỌC THÊM', row['source'])
            row['source'] += '\nHotline: 123456'
            self.assertEqual(quality(row)[1], 'commercial_content')

if __name__ == '__main__':
    unittest.main()
