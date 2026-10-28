# Bài viết sức khỏe CafeF

Nguồn: https://cafef.vn/song/suc-khoe.chn, zone ID `188125`.
URL `/suc-khoe.chn` trong cấu hình cũ trả HTTP 404 khi kiểm tra ngày 14/10/2026.
Phân trang dùng endpoint công khai `/timelinelist/188125/{page}.chn`, xác định từ mã JavaScript của trang.

## Phạm vi

- Tối đa 100 trang bài viết được thử tải trong đợt thử nghiệm được người dùng duyệt; số bài giữ lại có thể thấp hơn sau lọc.
- Chỉ lấy liên kết từ vùng danh sách thuộc chuyên mục sức khỏe. Không lấy liên kết sidebar chung và không tải ảnh/video.
- Giữ bài 250–2.500 từ, có ít nhất 3 cụm từ thuộc danh sách sức khỏe trong nội dung chính. Loại tiêu đề về cổ phiếu, chứng khoán, bất động sản, doanh thu, lợi nhuận hoặc giá vàng.
- Loại quảng cáo, bài liên quan, chú thích ảnh, script và phần ngoài nội dung bài. Loại trùng ID, hash nội dung và gần trùng có Jaccard 5-gram >= 0,85.
- Đây là bộ lọc chủ đề có quy tắc, chưa phải kiểm chứng tính đúng đắn của thông tin y tế. Cần rà soát nhãn trước khi train.
- Chuyên mục có bài lưu trữ từ nhiều năm; ngày crawl không phải ngày đăng. Không coi dữ liệu này là tập tin mới nhất hoặc mẫu đại diện ngẫu nhiên.

## Các file

- `articles.jsonl`: mỗi dòng là một bài đã qua lọc, UTF-8. Các trường chính: `id`, `url`, `title`, `pub_date`, `source`, `sapo`, `category`, `src_word_count`, `source_sha256`, `crawled_at`, `filter_evidence`.
- `crawl_audit.jsonl`: URL đã thử, lý do giữ/loại và lỗi nếu có. Với bài bị loại chỉ lưu metadata phục vụ kiểm tra, không lưu toàn văn vào dataset.
- `stats_*.json`: số lượng, thời gian đăng, số byte tải, lý do dừng và thống kê loại theo từng lần chạy.
- `robots_*.txt`: robots.txt đọc tại thời điểm crawl.

`source` chỉ gồm thân bài đã làm sạch. `sapo` là mô tả của nhà xuất bản lấy từ metadata, lưu riêng. **Chưa có trường `summary` hay nhãn gold do con người xác nhận.** Crawler không tạo nhãn giả, không ghi đè bộ train/val/test cũ và không chạy training.

## Crawler và chạy lại

Mã nguồn: `scripts/crawl_cafef_health.py`. Chỉ dùng thư viện chuẩn Python, không cần cài package hoặc tải model.

```powershell
python scripts/crawl_cafef_health.py --output data/cafef_health --max-articles 100 --max-pages 10 --delay 1.5
```

Khi agent chạy trên máy này phải bọc lệnh bằng runner đã cài theo AGENTS.md. Giới hạn đợt hiện tại: 256 MiB job commit, 64 MiB tổng dữ liệu nhận, 4 MiB mỗi phản hồi; request tuần tự, cách ít nhất 1,5 giây. Dừng khi gặp 401/403/429/5xx thay vì tự thử lại. Không vượt robot rules hoặc chuyển sang domain khác khi redirect.

Chạy lại trên cùng thư mục dùng audit và ID đã lưu để tránh tải lại; giới hạn 100 tính cả URL đã thử trước đó. Không tự xóa các bài bị loại hay log. Nếu lần ghi bị gián đoạn tạo JSONL không hợp lệ, chương trình báo lỗi để kiểm tra file thay vì tự sửa/mất dữ liệu.

Kiểm thử offline:

```powershell
python -m unittest discover -s tests -p test_cafef_crawler.py -v
```

## Trước khi dùng để train

Rà soát nội dung và tạo nhãn tóm tắt riêng; sau đó mới chia train/validation/test theo nhóm bài trùng/gần trùng. Đừng trộn âm thầm với dữ liệu VnExpress cũ hay xem `sapo` là gold chất lượng đã được kiểm chứng.
