# Hệ thống tóm tắt văn bản tiếng Việt tự động

Đây là đề tài báo cáo cuối kỳ của môn **Xử lý ngôn ngữ tự nhiên**. Mục tiêu của đề tài là xây dựng và đánh giá hệ thống tóm tắt văn bản tiếng Việt, tập trung vào miền y tế, nơi yêu cầu đầu ra không chỉ ngắn gọn và mạch lạc mà còn phải hạn chế sai lệch số liệu.

Hệ thống được triển khai theo hướng thực nghiệm nhiều giai đoạn: thu thập và chuẩn hóa dữ liệu, fine-tune mô hình tóm tắt có giám sát, tăng cường dữ liệu an toàn, đánh giá bằng các thước đo ngôn ngữ và thước đo chuyên biệt về số liệu, sau đó thử nghiệm học tăng cường PPO để giảm hiện tượng sinh số liệu không có trong văn bản nguồn.

## Thành viên và vai trò

| MSSV | Thành viên | Vai trò chính |
|---|---|---|
| 52300115 | Đoàn Lâm Phúc Khang | Crawl data, thực nghiệm ViT5 checkpoint, thiết kế workflow tổng thể |
| 52300138 | Nguyễn Trí Nhân | ViT5-base, PPO Stage 3, tăng cường dữ liệu |
| 52300143 | Nguyễn Lại Vũ Phong | BARTpho, tăng cường dữ liệu |

## Nội dung chính của dự án

- Thu thập và xử lý dữ liệu văn bản y tế tiếng Việt.
- Xây dựng tập dữ liệu dạng `source-summary` phục vụ bài toán tóm tắt.
- Fine-tune các mô hình encoder-decoder cho tiếng Việt:
  - ViT5-base
  - ViT5 checkpoint
  - BARTpho
- Áp dụng tăng cường dữ liệu có kiểm soát, tránh làm sai lệch thực thể số.
- Đánh giá bằng các chỉ số:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore F1
  - Number Recall
  - Number Hallucination
  - Length Compliance
  - Length Ratio
- Thử nghiệm PPO ở Stage 3 để đưa các ràng buộc về số liệu, độ dài và hallucination vào quá trình tối ưu hóa mô hình.

## Cấu trúc thư mục chính

```text
.
├── notebooks/
│   ├── ViT5_base/
│   │   ├── stage1_vit5_base.ipynb
│   │   ├── stage2_vit5_base.ipynb
│   │   └── stage3_vit5_RLUsingPPO.ipynb
│   ├── ViT5_Checkpoint/
│   │   └── stage2_vit5_checkpoint.ipynb
│   ├── BARTPho/
│   │   ├── stage1_bartpho_word.ipynb
│   │   └── stage2_bartpho_word.ipynb
│   └── gradio_comparision/
│       ├── gradio_compare_stage2_vit5_vs_bartpho.ipynb
│       └── gradio_compare_stage3_ppo_vs_stage2_sft.ipynb
├── summ_rl/                  # lõi RL v2 (group-relative PPO), reward, ROUGE; có test offline
├── scripts/                  # crawl, tiền xử lý, kiểm chứng và xuất nhãn (chỉ dùng thư viện chuẩn)
├── tests/                    # kiểm thử offline (42 test)
├── data/                     # dữ liệu crawl/đã xử lý (JSONL không đưa lên Git, xem bên dưới)
├── DATA_QUALITY_PROTOCOL.md  # quy trình chất lượng dữ liệu trước khi train
├── REVIEW_AND_UPGRADES.md    # nhật ký rà soát và sửa notebook
├── 52300115_52300138_52300143.docx
├── NLP_Final.pptx
└── README.md
```

## Pipeline dữ liệu (offline, Python 3.10+)

Chi tiết quy trình và các kiểm tra nằm trong [DATA_QUALITY_PROTOCOL.md](DATA_QUALITY_PROTOCOL.md). Toàn bộ script chỉ dùng thư viện chuẩn, không tải model và không train.

```bash
# 1. Crawl (tuần tự, tuân thủ robots, có ngân sách tải)
python scripts/crawl_cafef_health.py --output data/cafef_health
python scripts/crawl_health_multisource.py --output data/health_multisource_raw_v1

# 2. Làm sạch, gom nhóm trùng, chia train/val/test giữ nguyên manifest cũ, tạo nhãn nháp extractive
python scripts/prepare_health_multisource.py --output data/health_multisource_prepared_v1
python scripts/verify_health_multisource.py data/health_multisource_prepared_v1

# 3. Duyệt nhãn: mở review.html, xuất health-reviewed-annotations.jsonl
python scripts/build_health_review.py data/health_multisource_prepared_v1

# 4. Chỉ xuất các nhãn đã được người duyệt (kiểm hash, người duyệt, tỷ lệ 20-25%, số liệu)
python scripts/export_reviewed_splits.py --prepared data/health_multisource_prepared_v1     --annotations health-reviewed-annotations.jsonl --output data/health_multisource_reviewed_v1

# 5. Dữ liệu sẵn sàng cho notebook: làm sạch, cắt theo cửa sổ đầu vào, vẽ lại nhãn nháp (xem TRAINING_PLAN.md)
python scripts/build_training_data.py --prepared data/health_multisource_prepared_v1 --output data/health_training_v1
python scripts/check_token_lengths.py --tokenizer <checkpoint_vit5> --data data/health_training_v1
```

RL Stage 3 v2 (`summ_rl/`, notebook `stage3_vit5_RL_v2_grpo.ipynb`) và ước tính thời gian train từng notebook nằm trong [TRAINING_PLAN.md](TRAINING_PLAN.md).

Trạng thái hiện tại: bản `health_multisource_prepared_v1` giữ 995/1000 bài (train 795, val 100, test 100; 993 nhóm độc lập), **0 nhãn được người duyệt, chưa sẵn sàng train**. Nhãn nháp là `extractive_pseudo`, không dùng điểm trên chúng để kết luận chất lượng.

Văn bản bài báo đã crawl (`data/**/*.jsonl`) không được commit vì robots.txt không cấp quyền phân phối lại; chạy lại các lệnh trên để tái tạo. Chạy kiểm thử:

```bash
python -m unittest discover -s tests -v
```

## Checkpoint models

Các notebook Gradio hiện được thiết kế để tự dò checkpoint trên Google Drive khi chạy bằng Colab. Những đường dẫn thường được notebook kiểm tra gồm:

### ViT5 Stage 2

```text
/content/drive/MyDrive/NLP_Final/Output stage 2 vit5 checkpoint/best_model
/content/drive/MyDrive/NLP_Final/Output stage 2 vit5 checkpoint/model_for_rl
/content/drive/MyDrive/NLP_Final/ViT5_stage_2/runs/vit5_base_stage2/best_model
/content/drive/MyDrive/NLP_Final/ViT5_stage_2/runs/vit5_base_stage2/checkpoints/checkpoint-735
/content/drive/MyDrive/C.Science - TDTU/NLP/Finalterm NLP/ViT5_stage_2/runs/vit5_base_stage2/checkpoints/checkpoint-735
```

### BARTpho Stage 2

```text
/content/drive/MyDrive/NLP_Final/BARTpho_stage_2/runs/bartpho_word_stage2/best_model
/content/drive/MyDrive/NLP_Final/Output stage 2 bartpho word/runs/bartpho_word_stage2/best_model
/content/drive/MyDrive/NLP_Final/Output stage 2 bartpho checkpoint/runs/bartpho_word_stage2/best_model
/content/drive/MyDrive/C.Science - TDTU/NLP/Finalterm NLP/BARTpho_stage_2/runs/bartpho_word_stage2/best_model
```

### ViT5 Stage 3 PPO

```text
/content/drive/MyDrive/NLP_Final/ppo_run/runs/ppo_run/best_model
/content/drive/MyDrive/NLP_Final/ppo_run/ppo_run/best_model
/content/drive/MyDrive/NLP_Final/ppo_run/best_model
/content/drive/MyDrive/NLP_Final/ViT5_stage_3/runs/ppo_run/best_model
```

Thông tin Drive chứa checkpoint được tổng hợp trong file `MODEL_CHECKPOINTS.txt`:

| Nhóm model | Link Drive |
|---|---|
| BARTpho | https://drive.google.com/drive/folders/1bj8wnZAHj53VYgJeIAgAl3SENBDY0u8j?usp=sharing |
| ViT5-base | https://drive.google.com/drive/folders/1aTUlU9HsD-uy3oJrJXWmz_78dXNBpfhG?usp=sharing |
| ViT5 checkpoint | https://drive.google.com/drive/folders/1JtzJXYVKgRXy0i6AdhM69A4D4TCusT9M?usp=drive_link |

Toàn bộ hệ thống được xây dựng và chạy trên Google Colab. Khi chạy notebook Gradio, cần mount Google Drive trước, sau đó kiểm tra lại các biến đường dẫn model. Nếu notebook tự dò sai vị trí checkpoint, chỉnh trực tiếp các biến sau để trỏ tới thư mục checkpoint tương ứng trên Drive:

```python
VIT5_MODEL_DIR = "/content/drive/MyDrive/<duong_dan_checkpoint_vit5_stage2>"
BARTPHO_MODEL_DIR = "/content/drive/MyDrive/<duong_dan_checkpoint_bartpho_stage2>"
STAGE2_MODEL_DIR = "/content/drive/MyDrive/<duong_dan_checkpoint_vit5_stage2>"
STAGE3_MODEL_DIR = "/content/drive/MyDrive/<duong_dan_checkpoint_vit5_stage3_ppo>"
```

Mỗi checkpoint hợp lệ cần có tối thiểu:

```text
config.json
model.safetensors hoặc pytorch_model.bin
tokenizer files tương ứng
```

Với ViT5, tokenizer thường cần `spiece.model`. Với BARTpho, tokenizer thường cần `vocab.txt` và `bpe.codes`.

## Hướng dẫn chạy notebook comparison

Các notebook comparison nằm tại:

```text
notebooks/gradio_comparision/
```

Toàn bộ phần so sánh được chạy trên **Google Colab**, khuyến nghị dùng runtime GPU để tốc độ sinh tóm tắt ổn định hơn. Không cần khởi tạo môi trường ngoài Colab.

### 1. So sánh ViT5 Stage 2 và BARTpho Stage 2

Notebook:

```text
notebooks/gradio_comparision/gradio_compare_stage2_vit5_vs_bartpho.ipynb
```

Mục đích:

- So sánh đầu ra tóm tắt giữa ViT5 Stage 2 và BARTpho Stage 2.
- Cho phép chọn bài mẫu từ dataset hoặc dán văn bản thủ công.
- Hiển thị output của hai mô hình, bảng metric nhanh và đối chiếu số liệu trong source/output.

Các bước chạy:

1. Mở notebook trên Colab.
2. Chạy cell cài đặt dependency nếu runtime thiếu package.
3. Mount Google Drive.
4. Kiểm tra các biến:

```python
VIT5_MODEL_DIR
BARTPHO_MODEL_DIR
DATA_PATH
BARTPHO_PRED_PATH
```

5. Nếu notebook tự dò sai đường dẫn, sửa trực tiếp `VIT5_MODEL_DIR` và `BARTPHO_MODEL_DIR`.
6. Chạy các cell load model và launch Gradio.
7. Mở link Gradio được Colab sinh ra để thử nghiệm.

Ghi chú:

- Notebook có cơ chế dùng cached prediction của BARTpho nếu tìm thấy file prediction đã export.
- Nếu checkpoint BARTpho thiếu weight, cần kiểm tra lại thư mục model trước khi chạy live generation.

### 2. So sánh ViT5 Stage 2 SFT và Stage 3 PPO

Notebook:

```text
notebooks/gradio_comparision/gradio_compare_stage3_ppo_vs_stage2_sft.ipynb
```

Mục đích:

- So sánh output giữa ViT5 Stage 2 SFT và ViT5 Stage 3 PPO.
- Kiểm tra tác động của PPO đối với độ dài, số liệu giữ lại và số liệu sinh mới.
- Hỗ trợ chọn mẫu từ dataset hoặc dán văn bản thủ công.

Các bước chạy:

1. Mở notebook trên Colab.
2. Nếu thiếu package, bật `INSTALL_DEPS = True` ở cell dependency rồi chạy lại.
3. Mount Google Drive.
4. Kiểm tra các biến:

```python
STAGE2_MODEL_DIR
STAGE3_MODEL_DIR
DATA_PATH
```

5. Nếu notebook tự dò sai, sửa trực tiếp các biến path ở cell dò đường dẫn.
6. Chạy các cell load model.
7. Chạy cell Gradio cuối cùng.

Ghi chú quan trọng:

- Stage 3 PPO được giữ gần Stage 2 bằng KL penalty, nên không phải mọi input đều cho output khác Stage 2.
- Khi dùng beam search, hai model có thể trả về cùng một output ở một số mẫu.
- Việc đánh giá PPO nên dựa trên thống kê toàn tập test, đặc biệt là Number Hallucination, thay vì chỉ quan sát một mẫu đơn lẻ.

## Cài đặt môi trường khuyến nghị

Các notebook được thiết kế để chạy trên Google Colab. Môi trường khuyến nghị:

```text
Python 3.10+
PyTorch có GPU CUDA
transformers==4.46.3
tokenizers==0.20.3
datasets==3.1.0
evaluate==0.4.3
accelerate==1.1.1
sentencepiece==0.2.0
safetensors==0.4.5
bert-score==0.3.13
rouge-score==0.1.2
gradio==4.44.1
```

Tùy notebook, một số package có thể đã được cài sẵn hoặc được cài tự động ở cell đầu.

## Ghi chú thực nghiệm

- Stage 1 ViT5-base dùng VietNews để học tác vụ tóm tắt tiếng Việt.
- Stage 2 ViT5-base dùng dữ liệu y tế và chọn best checkpoint theo ROUGE-L validation.
- Stage 2 có thêm Number-Aware Re-ranking ở giai đoạn suy diễn để chọn candidate tốt hơn.
- Stage 3 PPO dùng Actor/Reference model, reward thủ công và KL penalty để giảm rủi ro hallucination số liệu.
- Kết quả hiện tại cho thấy SFT + Re-ranking cân bằng tốt nhất về tổng thể, còn PPO có ưu thế chuyên biệt ở tiêu chí giảm Number Hallucination.
