# Rà soát và cải thiện pipeline tóm tắt tiếng Việt

Ngày: 14/10/2026. Bản gốc: `0195b5bd81753bb2b4a4b2e61acecc41bd908dee`.
Repo được clone vào thẳng `D:\Text Summary`; không tạo thư mục trung gian mang MSSV.
Không đổi tên repo trên GitHub, không push hoặc commit thay người dùng.

## Những lỗi đã sửa

1. **Mount Drive trong ViT5 Stage 2:** bỏ `shutil.rmtree('/content/drive')` và thao tác unmount trước khi mount. Trước đây exception khi unmount bị bỏ qua, sau đó code vẫn cố xóa cây thư mục Drive.
2. **Đọc dữ liệu:** ViT5-base Stage 2, BARTpho Stage 2 và PPO từ chối source/summary null, không phải chuỗi hoặc chỉ chứa khoảng trắng. Không còn biến `None` thành chuỗi `"None"`. Kiểm tra split rỗng và giao nhau theo source chuẩn hóa Unicode/khoảng trắng/chữ hoa, ID hoặc URL. Phát hiện lỗi thì dừng, không tự sửa hay chia lại test set. Đây là kiểm tra trùng chính xác sau chuẩn hóa, chưa phải phát hiện gần trùng về ngữ nghĩa.
3. **Augmentation ViT5-base:** chỉ giữ biến thể chính tả `vaccine` → `vắc xin`. Bỏ các phép thay phụ thuộc ngữ cảnh như `tiêm` → `chích ngừa`, bỏ biến thể trùng/không đổi. Mặc định không dùng lại file augmentation cũ chưa được rà soát. Bật back-translation chưa triển khai sẽ báo lỗi rõ ràng. Kiểm tra giao split sau augmentation.
4. **ROUGE:** các cell tính ROUGE dùng tokenizer giữ dấu tiếng Việt, chuẩn hóa NFC và casefold. Công bố ROUGE-1/2/L, không công bố ROUGE-Lsum với cách xuống dòng từng từ của code cũ. Đây là tokenizer theo đơn vị chữ/số ngăn cách bằng dấu câu, không phải bộ tách từ tiếng Việt. **Điểm mới không được so trực tiếp với điểm cũ**; cần đánh giá lại các mô hình cùng cấu hình. Hugging Face Evaluate hỗ trợ truyền callable qua `tokenizer`: [mã nguồn ROUGE](https://github.com/huggingface/evaluate/blob/main/metrics/rouge/rouge.py).
5. **Sinh ứng viên:** bỏ `custom_generate` và `trust_remote_code` khỏi reranking ViT5-base/BARTpho. Dùng group beam search có sẵn trong [Transformers 4.46.3](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/generation/utils.py), đúng phiên bản repo chỉ định.
6. **Checkpoint PPO:** không đoán trọng số bằng cách chọn file lớn nhất, không lấy config/tokenizer của base model để tự sửa checkpoint. Yêu cầu checkpoint ViT5 đầy đủ; chấp nhận trọng số đơn hoặc shard index đủ file, từ chối shard path thoát thư mục. Đây là kiểm tra cấu trúc file, không xác nhận nội dung tensor hợp lệ. Transformers vẫn phải load và kiểm tra kiến trúc thật.
7. **Chọn kết quả PPO:** mỗi lần chạy có thư mục riêng. Đánh giá và lưu SFT ban đầu làm mốc, chỉ thay bằng PPO khi composite validation tốt hơn. Đếm bước theo số cập nhật đã hoàn tất và đánh giá bước cuối kể cả run ngắn hơn chu kỳ evaluation. Test trong cùng session dùng đường dẫn checkpoint thực sự được chọn và cấu hình hiện tại. Chạy test độc lập vẫn cần chỉ định đường dẫn của run muốn đánh giá.
8. **Reward PPO nhất quán:** train/validation/test cùng cách xử lý mẫu không có số và số không xuất hiện trong nguồn. Test không còn miễn phạt cho số chỉ có trong gold. Đây là metric kiểm tra chuỗi số, không đo toàn bộ tính đúng đắn về thực thể, đơn vị hay nội dung.
9. **Chẩn đoán PPO:** ghi log JSONL sau mỗi bước hoàn tất, dừng khi reward/KL/loss không hữu hạn và kiểm tra stop-file của runner ở đầu mỗi batch. Run có thể vẫn bị runner buộc dừng nếu đang ở giữa batch dài. Checkpoint xuất ra là trọng số suy diễn; chưa có cơ chế resume PPO đầy đủ optimizer/value head/RNG.
10. **Demo comparison:** mặc định tắt cài dependency tự động ở notebook ViT5/BARTpho. Các notebook cũ khác vẫn cần rà soát các cell setup trước khi chạy; đợt này chưa thay requirements hay môi trường đã cài.

Output và execution count của cell đã sửa được xóa để không hiển thị kết quả cũ như thể được tạo bởi code mới. Các cell khác, tài liệu nhóm cũ, metadata và định dạng JSON ngoài phần chỉnh sửa được giữ lại.

## Kiểm chứng đã thực hiện

- 8 kiểm thử offline trong `tests/test_notebook_regressions.py` đã đạt.
- Parse các Python cell của toàn bộ notebook, bỏ qua dòng lệnh IPython khi parse.
- Kiểm thử helper thật trích từ notebook: dữ liệu lỗi/trùng split, Unicode NFC/NFD và dấu tiếng Việt, augmentation, checkpoint thiếu shard/path traversal, reward train/test.
- Chạy vòng lặp PPO thật với model/trainer giả rất nhỏ: run 2 bước vẫn đánh giá bước cuối; giữ SFT khi PPO kém hơn và chọn PPO khi tốt hơn; kiểm tra log từng bước.
- Các tiến trình clone, sửa file và kiểm thử chạy qua runner Job Object đã cài. Runner báo xác nhận job membership và toàn bộ tiến trình trong job đã kết thúc. Kiểm thử dùng dưới 17 MiB peak job commit; đây không phải số đo chi phí training.

Lệnh kiểm thử: `python -m unittest discover -s tests -v`. Khi agent thực hiện, dùng runner đã cài theo AGENTS.md.

**Chưa chạy training, inference bằng model thật hoặc đo chất lượng trên dataset.** Không tải model/dataset, không cài package, không dùng GPU. Không kết luận ROUGE tăng hay model tốt hơn chỉ dựa trên unit test. Chưa kiểm chứng đầy đủ các cơ chế containment cho workload nặng; cần làm trước khi train theo chính sách máy.

## Dữ liệu và công việc tiếp theo

Repo chỉ có link Drive trong `notebooks/data/data.txt` và `MODEL_CHECKPOINTS.txt`, chưa có JSONL train/val/test hoặc trọng số thật. README dữ liệu nhắc đến crawler và script dựng nhãn nhưng các file đó không có trong cây repo hiện tại. Vì vậy clone repo chưa đủ để tái lập thí nghiệm.

Thứ tự đề xuất:

1. Chỉ định dataset/checkpoint đang có trên máy; nếu cần tải từ Drive, xác nhận nguồn, phạm vi, kích thước và đích tải trước.
2. Rà soát nhãn extractive, trùng/gần trùng và tỷ lệ cắt đầu vào theo từng split. Không điều chỉnh test để cải thiện điểm. Kiểm tra preprocessing/tách từ BARTpho-word khớp checkpoint Stage 1.
3. Kiểm tra phiên bản thư viện, cơ chế giới hạn workload và tài nguyên thực tế. Tách rõ runtime Colab hiện tại với môi trường Windows; các notebook vẫn giữ đường dẫn Colab cần cấu hình lại.
4. Chạy lại đánh giá SFT và reranking bằng metric mới trên cùng tập validation; quyết định có cần train Stage 2 lại. Sau đó mới thử PPO và so sánh cùng tiêu chí, cùng tập test.
5. Nếu tiếp tục nâng cấp kiến trúc: tách helper chung thành package, thêm CLI/config và resume checkpoint đầy đủ. Đợt sửa này giữ notebook tự chứa để không làm hỏng cách mở trực tiếp trên Colab.

## Đợt rà soát 28/10/2026 (source Python)

Chỉ sửa `summ_rl/`, `scripts/` và `tests/`; không đổi notebook, dữ liệu hay cách gọi API mà notebook RL v2 đang dùng.

1. **Resume RL (`GroupRLTrainer.train`):** yêu cầu dừng qua `should_stop` giờ lưu `resume_state.pt` trước khi thoát, nên không mất các bước đã chạy kể từ lần lưu gần nhất. Khi resume, nếu `best_model/` trên đĩa tốt hơn điểm trong trạng thái đã lưu (best được ghi sau lần lưu cuối) thì lấy điểm trên đĩa làm mốc, tránh ghi đè bằng model kém hơn. `resume=True` khi chưa có trạng thái nào báo lỗi rõ thay vì gợi ý ngược "hãy dùng resume=True". `rl_config.json` được ghi ngay đầu run.
2. **`eval_kwargs`:** truyền `max_input_len`, `max_new_tokens`... trong `eval_kwargs` trước đây gây `TypeError` trùng tham số; giờ giá trị truyền vào ghi đè mặc định. `evaluate_model` chỉ truyền `early_stopping` khi `num_beams > 1` (hết cảnh báo của Transformers) và báo lỗi rõ khi danh sách rỗng.
3. **Kiểm tra cấu hình:** `GroupRLConfig` và `RewardConfig` từ chối giá trị vô nghĩa ngay khi tạo (ví dụ `group_size < 2` làm lợi thế nhóm luôn bằng 0, `min_ratio >= max_ratio`, trọng số âm). Bỏ lượt encoder thừa cho nhóm không có tín hiệu khi `sft_coef = 0`.
4. **`build_training_data.fit_window`:** khi câu đầu tiên đã dài hơn cửa sổ, hàm cũ trả về nguyên bài (vẫn vượt cửa sổ) nhưng gắn cờ đã cắt. Giờ cắt tại từ thứ `max_words`, kết quả luôn là tiền tố vừa cửa sổ. Bỏ nhánh lọc `label_type` không bao giờ chạy; không còn crash khi không mẫu nào có sapo hoặc mọi mẫu bị loại. Kiểm tra nhóm trùng vượt split chạy trước khi ghi file.
5. **`check_token_lengths`:** gợi ý `--max-source-words` trước đây chỉ dựa trên split cuối cùng (test) do biến vòng lặp bị dùng lại. Giờ gộp cả ba split và dùng phân vị 95 thay vì trung bình. Phần tính toán tách thành hàm thuần có test.
6. **Kiểm định dữ liệu không dùng `assert`:** `verify_health_multisource.py`, `prepare_health_multisource.py`, `prepare_health_data.py` dùng kiểm tra tường minh, nên vẫn chạy dưới `python -O`. Trình kiểm định liệt kê mọi lỗi và trả mã thoát 1 khi có lỗi; có argparse.
7. **Trang duyệt (`build_health_review.py`):** có argparse, bỏ qua dòng trống, báo lỗi dễ hiểu khi `review.html` đã tồn tại. Trang không còn crash với review queue cũ thiếu `evidence_spans`, `prepared_source_sha256` hoặc `src_word_count`.
8. **Nhỏ:** `tokens(None)` trả về rỗng thay vì `['none']`; `verify_cafef_health` ghi lỗi ngày không hợp lệ thay vì crash; `probe_health_sources` không tạo thư mục khi chỉ import; tạo thư mục cha của output trước khi kiểm tra dung lượng; giữ nguyên nguyên nhân gốc khi dừng crawler vì lỗi HTTP.
9. **Lint:** thêm `ruff.toml`; `ruff check .` sạch trên `summ_rl/`, `scripts/`, `tests/`.

Kiểm chứng: 54 test offline đạt (thêm 11 test cho các mục trên, gồm dừng giữa chừng rồi resume cho trọng số trùng với run không ngắt, và kiểm định bắt được dữ liệu bị sửa). Vẫn **chưa chạy trên ViT5 thật hay dữ liệu thật**; các số liệu chất lượng trong tài liệu không thay đổi.
