import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/crawl_cafef_health.py'
spec = importlib.util.spec_from_file_location('cafef_crawler', PATH)
crawler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawler)


class CrawlerTests(unittest.TestCase):
    def test_listing_excludes_unrelated_sidebar(self):
        html = '''<div data-cd-key="siteid188:newsinzone:zone188125">
        <a href="/bai-suc-khoe-20230318132336279.chn" title="Sức khỏe">đọc</a></div>
        <aside><a href="/chung-khoan-20230318132336280.chn">Chứng khoán</a></aside>'''
        self.assertEqual(len(crawler.discover(html)), 1)
        self.assertIn('bai-suc-khoe', crawler.discover(html)[0][0])

    def test_url_boundary_and_tracking_removal(self):
        self.assertIsNone(crawler.article_url('https://other.example/bai-20230318132336279.chn'))
        self.assertIsNone(crawler.article_url('javascript:alert(1)'))
        self.assertIsNone(crawler.article_url('/song.chn'))
        self.assertEqual(crawler.article_url('/bai-20230318132336279.chn?utm_source=a#x'),
                         'https://cafef.vn/bai-20230318132336279.chn')

    def test_body_excludes_related_and_caption(self):
        html = '''<meta property="og:title" content="Sức khỏe &amp; dinh dưỡng">
        <meta property="og:description" content="Publisher sapo">
        <span itemprop="datePublished" datetime="2023-03-18T16:25:18+07:00"></span>
        <div class="detail-content afcbc-body"><p>Bác sĩ khuyên <b>ngủ đủ</b> giấc.</p>
        <div class="tindnd"><p>Tin cổ phiếu không liên quan</p></div>
        <div class="VCSortableInPreviewMode"><p>Chú thích ảnh</p></div>
        <p>Ăn uống<br>điều độ.</p><script>bad()</script></div><p>Footer</p>'''
        record = crawler.parse_article(html, 'https://cafef.vn/bai-20230318132336279.chn')
        self.assertEqual(record['source'], 'Bác sĩ khuyên ngủ đủ giấc.\n\nĂn uống điều độ.')
        self.assertEqual(record['title'], 'Sức khỏe & dinh dưỡng')
        self.assertNotIn('summary', record)
        self.assertEqual(record['sapo'], 'Publisher sapo')

    def test_health_filter_rejects_finance_and_short_articles(self):
        body = 'Bác sĩ chia sẻ về dinh dưỡng và bệnh nhân. ' * 40
        record = dict(title='Bác sĩ chia sẻ', source=body, src_word_count=len(body.split()))
        self.assertTrue(crawler.health_decision(record)[0])
        self.assertFalse(crawler.health_decision({**record, 'title': 'Cổ phiếu bệnh viện tăng giá'})[0])
        self.assertFalse(crawler.health_decision({**record, 'src_word_count': 20})[0])
        self.assertFalse(crawler.health_decision({**record, 'source': 'Tin du lịch ' * 100})[0])

    def test_missing_body_is_error(self):
        with self.assertRaises(ValueError):
            crawler.parse_article('<html>error page</html>', 'https://cafef.vn/bai-20230318132336279.chn')


if __name__ == '__main__':
    unittest.main(verbosity=2)
