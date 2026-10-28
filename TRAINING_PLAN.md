# Kế hoạch chia notebook, thời gian train và RL v2

Số giờ dưới đây là **ước tính từ khối lượng tính toán** (ViT5-base 220M tham số, BARTpho-word khoảng 1,8 lần), **chưa đo trên GPU Colab**. Notebook RL v2 in thời gian mỗi bước và ETA sau bước đầu; hãy tin số đó hơn bảng này.

## 1. Thứ tự chạy và thời gian ước tính

| # | Notebook | Việc làm | A100 (bf16) | L4 (bf16) | T4 |
|---|---|---|---|---|---|
| 1 | `ViT5_base/stage1_vit5_base` | SFT trên 30k VietNews, 3 epoch (5.625 bước, batch 16) | 1–1,5 giờ | 3–5 giờ | không khả thi (T5 lỗi fp16, fp32 rất chậm) |
| 2 | `BARTPho/stage1_bartpho_word` | Như trên cho BARTpho-word | 2–3 giờ | 6–9 giờ | không khả thi |
| 3 | `ViT5_base/stage2_vit5_base` | SFT y tế, ~793 mẫu, 6 epoch, rerank 5 ứng viên | 30–50 phút | 1–2 giờ | 3–5 giờ |
| 4 | `BARTPho/stage2_bartpho_word` | Như trên cho BARTpho | 50–80 phút | 2–3 giờ | 5–8 giờ |
| 5 | `ViT5_Checkpoint/stage2_vit5_checkpoint` | SFT từ checkpoint ViT5 đã có | 30–50 phút | 1–2 giờ | 3–5 giờ |
| 6 | `ViT5_base/stage3_vit5_RL_v2_grpo` | **RL mới**, 150 bước (+ 6 lần eval trên val) | 2–3 giờ | 4–6 giờ | rất chậm, dùng profile nhỏ |
| 7 | `gradio_comparision/*` | So sánh Stage 2 và Stage 3 | vài phút | vài phút | vài phút |

Tổng đường chính cho bài nộp (bước 3, 6, 7 nếu đã có Stage 1): **khoảng 3–4 giờ trên A100**. Bước 1 và 2 chỉ chạy lại nếu cần train lại từ đầu; checkpoint Stage 1 đã có trên Drive.

Cách chia để khỏi mất tiến độ khi Colab ngắt: mỗi notebook là một phiên riêng, ghi ra Drive, và bước sau chỉ đọc đường dẫn checkpoint của bước trước. Bước 6 ghi `best_model` ngay từ bước 0 (SFT) nên ngắt giữa chừng vẫn còn model dùng được.

## 2. Vì sao RL cũ yếu và RL v2 sửa gì

1. **Trọng số bf16 với LR 1e-6.** bf16 chỉ có 8 bit mantissa; với trọng số cỡ 0,05 thì bước nhỏ nhất biểu diễn được khoảng 2,4e-4, còn mỗi cập nhật AdamW chỉ cỡ LR = 1e-6, nên bị làm tròn về 0. Test `test_bf16_weights_lose_small_updates_but_fp32_keeps_them` tái hiện đúng hiện tượng này. Đây là nguyên nhân khả dĩ nhất, suy ra từ mã nguồn chứ chưa kiểm chứng trên log chạy cũ.
2. **Quá ít tín hiệu.** ~147 cập nhật, mỗi batch dùng một lần, một mẫu cho mỗi prompt, critic học trên ~800 mẫu.
3. **Reward không có thông tin nội dung**, chỉ luật số liệu/độ dài/đa dạng nên không có hướng cải thiện chất lượng.
4. **Dropout bật khi lấy mẫu và cập nhật**, làm tỉ số xác suất PPO nhiễu.
5. **Chọn best trên 30 mẫu val**, nhiễu và dễ chọn nhầm.

RL v2 (`summ_rl/`): 8 ứng viên mỗi prompt, lợi thế chuẩn hóa trong nhóm, clipped PPO hai lần mỗi batch, KL tới SFT thích nghi, neo NLL trên tham chiếu, reward gồm ROUGE + số liệu + độ dài + chống lặp − hallucination số, fp32 master weights, chọn best trên toàn bộ val và từ chối model làm tăng hallucination số quá 1 điểm phần trăm. Tinh chỉnh nhanh: `lr` 5e-6 đến 2e-5, `group_size` 4 đến 16, `temperature` 0,8 đến 1,0, `kl_coef` 0,02 đến 0,1.

Kiểm thử có chạy thật trên model T5 nhỏ ngẫu nhiên: cập nhật trọng số hữu hạn, và với reward kiểm chứng được (đếm một token mục tiêu) thì chính sách tăng reward rõ rệt. **Chưa chạy trên ViT5 thật, nên chưa có số liệu ROUGE nào cho RL v2.**

## 3. Dữ liệu

Chạy `scripts/build_training_data.py` (xem README). Thay đổi so với nhãn nháp trước đó:

- Bỏ dòng ghi nguồn/ảnh/tác giả và dòng tiêu đề lẻ khỏi thân bài.
- Cắt thân bài ở ranh giới câu về 600 từ, vì bài trung bình ~858 từ vượt cửa sổ 1.024 token. Trước đây nhãn có thể chứa ý ở phần model không nhìn thấy, khuyến khích bịa. 90% mẫu bị cắt. **600 từ là giả định ~1,7 token/từ**; chạy `scripts/check_token_lengths.py --tokenizer <checkpoint>` trên Colab rồi chạy lại với `--max-source-words` đúng.
- Chấm câu có thưởng tiêu đề, vị trí đầu bài, câu có số liệu, và phạt câu mở đầu bằng liên từ (`Tuy nhiên`, `Do đó`...) hoặc câu hỏi, giảm câu thiếu ngữ cảnh. Vẫn là nguyên văn có vị trí ký tự để truy ngược.
- Loại 2 mẫu có độ dài nháp ngoài 20–25% hoặc nguồn quá ngắn. Còn 993 mẫu: train 793, val 100, test 100, không nhóm nào nằm ở hai tập.

Độ khớp ROUGE-1 của nháp với sapo của nhà báo tăng từ 17,1 lên 21,6 (ROUGE-L 11,8 lên 15,2). Sapo **không** dùng để chấm điểm nên không vòng lặp, nhưng sapo vốn tóm đầu bài, nên phần tăng này có thể chủ yếu do thưởng vị trí đầu bài. Đây chỉ là chỉ báo yếu, không phải chất lượng nhãn.

**Điểm nghẽn thật sự vẫn là nhãn.** Toàn bộ nhãn là `extractive_pseudo`, chưa có người duyệt. Muốn kết luận "RL tốt hơn" một cách đáng tin cần duyệt tối thiểu val và test (dùng `build_health_review.py` rồi `export_reviewed_splits.py`) và dùng thư mục `reviewed_splits` để đánh giá.

## 4. Việc cần làm trên Colab

1. Upload `data/health_training_v1` lên Drive (JSONL không có trên GitHub vì bản quyền bài báo).
2. Với Stage 2, đặt `DATA_ROOT` trỏ tới thư mục đó (cùng schema `id/source/summary`).
3. Chạy `check_token_lengths.py`, chỉnh `--max-source-words` nếu cần và dựng lại dữ liệu.
4. Chạy notebook RL v2; sau 3 bước đọc ETA, nếu quá lâu thì giảm `STEPS` hoặc `group_size`.
