# Bộ dữ liệu tóm tắt văn bản tiếng Việt - VnExpress Sức khỏe

## Tổng quan

Dataset này dùng domain y tế/sức khỏe từ VnExpress, phù hợp yêu cầu "tin y tế/dịch tễ" và có mật độ số liệu thấp hơn domain tài chính/chứng khoán.

| Split | Số mẫu | Ghi chú |
| --- | ---: | --- |
| train | 780 | 80% |
| val | 94 | 10% |
| test | 105 | 10%, đạt ngưỡng >= 100 |
| tổng | 979 | train/val/test |

Nguồn crawl:

- Raw: `data/crawl_health/vnexpress_health_raw.jsonl` - 1.429 bài.
- Clean: `data/crawl_health/vnexpress_health_clean.jsonl` - 1.422 bài.
- Sau sinh nhãn và lọc chất lượng: 979 cặp source/summary.

Phân bố category trong clean set:

| Category | Số bài |
| --- | ---: |
| `suc-khoe` | 411 |
| `tin-tuc` | 326 |
| `song-khoe` | 304 |
| `vaccine` | 283 |
| `cac-benh` | 98 |

## Chỉ số chất lượng nhãn

Kết quả từ `scripts/validate_dataset.py`:

| Chỉ số | train | val | test |
| --- | ---: | ---: | ---: |
| Number Accuracy avg | 100.0% | 100.0% | 100.0% |
| Number Accuracy perfect | 100.0% | 100.0% | 100.0% |
| Length Compliance [20%, 25%] | 73.08% | 74.47% | 71.43% |
| Avg source words | 679.9 | 683.1 | 697.5 |
| Avg summary words | 159.9 | 158.9 | 164.4 |
| Avg length ratio | 0.2352 | 0.2340 | 0.2378 |
| Avg unique source numbers | 8.83 | 9.10 | 8.95 |

Quality tag toàn bộ dataset:

| Tag | Định nghĩa | Số mẫu |
| --- | --- | ---: |
| `high` | number_accuracy = 1.0 và length_ratio trong [0.20, 0.25] | 715 |
| `medium` | number_accuracy = 1.0 và length_ratio trong [0.18, 0.30] ngoài vùng high | 264 |
| `loose` | Không giữ trong bản health hiện tại | 0 |

## Phương pháp sinh nhãn

Nhãn được sinh bằng constrained extractive summarization trong `scripts/build_summary_dataset.py`:

1. Tách câu bằng regex tiếng Việt, tránh cắt giữa số liệu.
2. Chọn câu theo set-cover để bảo toàn 100% số liệu duy nhất trong bài gốc.
3. Bổ sung câu nếu summary ngắn hơn mục tiêu.
4. Loại các mẫu không đạt `number_accuracy = 1.0` hoặc `length_ratio` ngoài [0.18, 0.30].
5. Chia 80/10/10 stratified theo `(category, quality_tag)`.

## Schema `dataset.jsonl`

Mỗi dòng là một JSON object:

```json
{
  "id": "...",
  "url": "https://vnexpress.net/...",
  "category": "suc-khoe",
  "title": "...",
  "pub_date": "...",
  "source": "<văn bản gốc>",
  "summary": "<tóm tắt extractive>",
  "src_word_count": 680,
  "tgt_word_count": 160,
  "length_ratio": 0.2353,
  "src_numbers": ["..."],
  "tgt_numbers": ["..."],
  "missing_numbers": [],
  "number_accuracy": 1.0,
  "length_compliance": true,
  "quality_tag": "high"
}
```

## Tái tạo

```powershell
$env:PYTHONIOENCODING='utf-8'
python data/crawl_health/vnexpress_health_crawler.py
python data/crawl_health/validate_health_crawl.py
python scripts/build_summary_dataset.py
python scripts/validate_dataset.py
```

Lưu ý: crawler đã được chỉnh để parse song song giới hạn 4 worker và resume theo `id` nếu raw file đã tồn tại.

## File index

```text
data/
├── README.md
├── stats.json
├── crawl/
│   └── ... dữ liệu CafeF cũ, giữ lại làm backup/tham chiếu
├── crawl_health/
│   ├── vnexpress_health_crawler.py
│   ├── validate_health_crawl.py
│   ├── vnexpress_health_raw.jsonl
│   └── vnexpress_health_clean.jsonl
├── train/dataset.jsonl     ← dùng cho huấn luyện
├── val/dataset.jsonl       ← dùng cho validation
├── test/dataset.jsonl      ← dùng cho test
└── format/                 ← bản Markdown để đọc / trình bày
    ├── README.md
    ├── train.full_text.md
    ├── val.full_text.md
    ├── test.full_text.md
    └── sample_15.full_text.md
```
