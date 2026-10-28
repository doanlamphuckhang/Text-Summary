# Dữ liệu sức khỏe đã tiền xử lý — v1

Ngày xử lý: 15/10/2026. Nguồn: `data/cafef_health/articles.jsonl`.

**Đã chuẩn hóa và chia tập; chưa có nhãn được người duyệt, chưa sẵn sàng cho training/đánh giá chính thức.** Không train model, tải tokenizer/weights, cài package hoặc crawl thêm.

| Hạng mục | Kết quả |
|---|---:|
| Bài đầu vào / giữ lại | 99 / 99 |
| Train / validation / test dự kiến | 79 / 10 / 10 |
| Nhóm theo bộ phát hiện gần trùng | 99 |
| Cặp gần trùng phát hiện được | 0 |
| Nhãn được người duyệt | 0 |
| Thân bài trung bình | 858 đơn vị phân cách bằng khoảng trắng |
| Sapo trung bình | 25,7 đơn vị; 3,1% thân bài |
| Bản nháp extractive | 99; dài 20–24,74% thân bài |
| Năm đăng | 51 bài năm 2023; 45 năm 2024; 3 năm 2026 |

Số từ không phải số token của ViT5/BARTpho. Chưa đo truncation vì chưa có tokenizer checkpoint cục bộ. Không âm thầm cắt văn bản để đạt giới hạn 1024 token; cần đo bằng đúng tokenizer trước khi train.

## File và nhãn

- `clean_articles.jsonl`: bài chuẩn hóa Unicode NFC, khoảng trắng, giữ đoạn và số liệu. Dữ liệu crawl gốc không thay đổi.
- `split_manifest.jsonl`: ID, nhóm, split và hash nguồn chuẩn hóa. Giữ manifest khi đánh giá cùng phiên bản dữ liệu.
- `group_links.jsonl`: cặp gần trùng theo 5-gram nội dung/tiêu đề/URL; hiện rỗng. Không bảo đảm phát hiện mọi bản diễn đạt lại.
- `draft_splits/{train,val,test}/dataset.jsonl`: cặp `source`/`summary` **nhãn giả extractive**, chỉ phục vụ kiểm tra định dạng/quy trình. Không dùng điểm trên các tập này làm bằng chứng chất lượng tóm tắt thực tế.
- `review_queue.jsonl`: nguồn và bản nháp để duyệt, mặc định `approved=false`, `reviewer` rỗng.
- `report.json`: thống kê, hash input, seed và trạng thái sẵn sàng.
- `rejected.jsonl`: dữ liệu lỗi; hiện rỗng.

Giữ mục tiêu tóm tắt 20–25% theo cấu hình project cũ. Sapo quá ngắn để thay thế mục tiêu này nên chỉ giữ tham khảo, không nối vào input hoặc tự coi là gold. Việc chuẩn bị cặp văn bản–tóm tắt và token hóa riêng đầu vào/nhãn được mô tả trong [hướng dẫn Hugging Face](https://huggingface.co/docs/transformers/main/tasks/summarization).

Thuật toán nháp chọn câu theo tần suất từ trong chính bài, giảm ưu tiên câu lặp ý và trả lại thứ tự gốc. Không học thống kê từ validation/test. Mỗi câu lưu trong `evidence_sentences`. Giữ nguyên câu không bảo đảm đủ ý, đủ ngữ cảnh hay đúng y khoa. Cần kiểm tra chủ thể, số liệu, đơn vị, phát biểu về tác dụng thực phẩm/chữa bệnh và điều kiện áp dụng. Không ép giữ mọi số trong nguồn.

Mẫu `cafef_188240108203107069` có chữ số trong sapo không tìm thấy trong thân bài theo bộ tách số. Đây là cờ rà soát, không tự kết luận sapo sai.

## Duyệt nhãn

1. Sao chép `review_queue.jsonl` thành file annotation riêng, giữ `id` và `source_sha256`.
2. Đọc nguồn, sửa `summary` đủ ý, đúng ngữ cảnh, dài 20–25%. Chỉ đặt `approved=true` sau khi duyệt và điền `reviewer`; dùng `notes` ghi chú. Duyệt validation/test độc lập với output model.
3. Chạy `scripts/prepare_health_data.py` với cùng input, `--annotations` trỏ vào file đã duyệt và `--output` là thư mục mới. Agent phải dùng runner theo AGENTS.md.

Script từ chối hash sai, reviewer rỗng, summary sai khoảng độ dài hoặc chứa chuỗi số vắng trong nguồn. Đây là kiểm tra cấu trúc, không thay việc duyệt nội dung. Chỉ mẫu được duyệt mới xuất vào `reviewed_splits`. `training_ready` báo có mẫu đã duyệt ở cả ba tập, không chứng nhận số lượng đủ.

Cùng input và seed 42 giữ nguyên chia tập. Khi thêm dữ liệu, split có thể thay đổi: phải lập phiên bản mới, đối chiếu manifest và khóa test. Không so điểm hai tập test khác nhau như cùng một benchmark.

## Có cần crawl thêm?

**99 bài đủ thử pipeline, chưa đủ để huấn luyện/đánh giá đáng tin cậy.** Chỉ có 79 mẫu train và 10 mẫu test; mỗi bài chiếm 10% test. Hiện nhãn chưa được duyệt nên chất lượng nhãn là trở ngại trước cả số lượng.

Đề xuất thực hành cho project, không phải ngưỡng tối thiểu bảo đảm chất lượng:

- Bước tiếp theo: khoảng **1.000 cặp sạch có nhãn được kiểm tra**, chia khoảng 800/100/100; ưu tiên nhãn được người duyệt cho validation/test.
- Nếu baseline còn yếu hoặc cần phủ nhiều chủ đề: tiến tới **2.000–3.000 cặp**, ít nhất 100–200 mẫu test đa dạng. Quyết định theo learning curve validation, không liên tục điều chỉnh theo test.
- Bổ sung bài mới và đa dạng dinh dưỡng, bệnh lý, vận động, sức khỏe tinh thần, phòng bệnh. Bộ hiện tại chủ yếu là bài cũ từ một báo, chưa đại diện rộng. Thống kê `topic_evidence` chỉ là từ khóa xuất hiện, không phải nhãn chủ đề được duyệt.
- Fine-tune checkpoint pretrained bằng SFT trước; chưa đề xuất train từ đầu hoặc ưu tiên PPO với 99 nhãn giả.

Nhiều bài hơn nhưng nhãn thiếu ý/sai ngữ cảnh không giải quyết được chất lượng. Đợt này chỉ xử lý dữ liệu đã có; crawl vượt giới hạn 100 bài ban đầu cần người dùng xác nhận phạm vi mới.
