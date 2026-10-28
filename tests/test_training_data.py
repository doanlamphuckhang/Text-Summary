from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_training_data import NEW, build_row, clean_paragraphs, fit_window, redraft


def article(n=30):
    return '\n\n'.join(f'Bác sĩ khuyến cáo nhóm {i} không tự ý dùng thuốc khi chưa có chỉ định, liều {i} mg mỗi ngày là đủ.'
                       for i in range(n))


class TrainingDataTests(unittest.TestCase):
    def test_credits_and_bylines_are_removed(self):
        text = 'Câu thân bài đầy đủ có nhiều từ để giữ lại.\n\nẢnh: Shutterstock\n\nNguyễn Văn An\n\nNguồn: Báo X'
        self.assertEqual(clean_paragraphs(text), 'Câu thân bài đầy đủ có nhiều từ để giữ lại.')

    def test_window_cuts_on_sentence_boundary(self):
        source = article()
        cut, truncated = fit_window(source, 100)
        self.assertTrue(truncated)
        self.assertLessEqual(len(cut.split()), 100)
        self.assertTrue(cut.endswith('đủ.'))
        self.assertTrue(source.startswith(cut))
        self.assertEqual(fit_window(source, 10 ** 6), (source, False))

    def test_window_cuts_inside_an_overlong_first_sentence(self):
        source = ' '.join(f'từ{i}' for i in range(50)) + '.\n\nCâu sau.'
        cut, truncated = fit_window(source, 10)
        self.assertTrue(truncated)
        self.assertEqual(cut, ' '.join(f'từ{i}' for i in range(10)))
        self.assertTrue(source.startswith(cut))

    def test_redraft_is_verbatim_and_in_range(self):
        source = article()
        summary, evidence = redraft(source, 'Không tự ý dùng thuốc', NEW)
        self.assertTrue(all(source[e['start']:e['end']] == e['text'] for e in evidence))
        self.assertEqual(summary, ' '.join(e['text'] for e in evidence))
        self.assertTrue(.2 <= len(summary.split()) / len(source.split()) <= .25)

    def test_dangling_sentence_loses_to_self_contained_one(self):
        filler = ' '.join(['Người bệnh cần theo dõi huyết áp mỗi ngày và ghi lại kết quả đo đều đặn.'] * 4)
        source = ('Tuy nhiên người bệnh cần theo dõi huyết áp mỗi ngày và ghi lại kết quả đo đều đặn nhé. '
                  'Người bệnh cần theo dõi huyết áp mỗi ngày và ghi lại kết quả đo đều đặn nhé. ' + filler)
        row = dict(id='x', source=source, title='Huyết áp', src_word_count=len(source.split()), sapo='')
        built = build_row(row, 10 ** 6, NEW)
        self.assertNotIn('Tuy nhiên', built['summary'][:12])


if __name__ == '__main__':
    unittest.main()
