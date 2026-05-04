# Format / Trình bày — VnExpress Health Summarization

Các file Markdown trong thư mục này dùng để **đọc và trình bày**
(không thay thế JSONL trong `data/train`, `data/val`, `data/test` —
các file đó vẫn là nguồn dữ liệu cho việc huấn luyện và đánh giá mô hình).

Mỗi file split bắt đầu bằng một trang tổng quan: phân bố category,
phân loại chất lượng và các chỉ số trung bình; sau đó là từng mẫu
với bảng metadata, bài gốc và bản tóm tắt extractive.

## Danh sách file

| File | Số mẫu | Mục đích |
|---|---:|---|
| `sample_15.full_text.md` | 15 | Preview nhanh 15 mẫu (5 mỗi split) |
| `test.full_text.md` | 105 | Toàn bộ tập kiểm thử |
| `train.full_text.md` | 780 | Toàn bộ tập huấn luyện |
| `val.full_text.md` | 94 | Toàn bộ tập validation |

## Phương pháp tóm tắt

Tóm tắt sinh bằng **extractive có ràng buộc** (`scripts/build_summary_dataset.py`):

1. Tách câu cho văn bản tiếng Việt, tránh cắt giữa số liệu và viết tắt.
2. Greedy set-cover: chọn tập câu nhỏ nhất bảo toàn 100% số liệu duy nhất.
3. Bổ sung câu nếu thiếu hụt độ dài; cắt câu dư khi vượt 25%.
4. Lọc các mẫu có `number_accuracy < 1.0` hoặc `length_ratio` ngoài [0.18, 0.30].
5. Chia 80/10/10 stratified theo `(category, quality_tag)`.
