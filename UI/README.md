# ViT5 Demo UI

Web UI demo cho mô hình tóm tắt tin y tế tiếng Việt.

## Cấu trúc cần có

```text
NLP_CK/
├── best_model/
├── data/
└── UI/
    ├── demo_app.py
    ├── requirements_demo.txt
    ├── run_demo.bat
    └── run_demo.ps1
```

`demo_app.py` tự tìm model ở `../best_model` và dữ liệu mẫu ở `../data/test/dataset.jsonl`.

## Chạy nhanh

Double-click:

```text
run_demo.bat
```

Hoặc chạy PowerShell:

```powershell
cd UI
.\run_demo.ps1
```

Sau đó mở:

```text
http://127.0.0.1:7860
```
