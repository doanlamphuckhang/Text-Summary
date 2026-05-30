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
├── UI/
│   └── demo_app.py
├── 52300115_52300138_52300143.docx
├── NLP_Final.pptx
└── README.md
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

Nếu lưu checkpoint trong thư mục local `MODEL_CHECKPOINTS`, có thể dùng cấu trúc gợi ý sau và sửa trực tiếp biến path trong notebook Gradio:

```text
MODEL_CHECKPOINTS/
├── vit5_stage2/
│   └── best_model/
├── bartpho_stage2/
│   └── best_model/
└── vit5_stage3_ppo/
    └── best_model/
```

Khi đó các biến có thể được chỉnh thành:

```python
VIT5_MODEL_DIR = PROJECT_ROOT / "MODEL_CHECKPOINTS" / "vit5_stage2" / "best_model"
BARTPHO_MODEL_DIR = PROJECT_ROOT / "MODEL_CHECKPOINTS" / "bartpho_stage2" / "best_model"
STAGE2_MODEL_DIR = PROJECT_ROOT / "MODEL_CHECKPOINTS" / "vit5_stage2" / "best_model"
STAGE3_MODEL_DIR = PROJECT_ROOT / "MODEL_CHECKPOINTS" / "vit5_stage3_ppo" / "best_model"
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

Nên chạy trên **Google Colab GPU** để tốc độ sinh tóm tắt ổn định hơn. Nếu chạy CPU, thời gian generate có thể tăng đáng kể vì mỗi lần so sánh phải chạy hai mô hình seq2seq.

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

Các notebook chủ yếu chạy trên Colab. Môi trường khuyến nghị:

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

