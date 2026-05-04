"""Validate crawl đã làm sạch data — tương tự validate_crawl.py cho CafeF."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "vnexpress_health_raw.jsonl"
OUTPUT = ROOT / "vnexpress_health_clean.jsonl"

NUMBER_PATTERNS = [
    r'\d+(?:[.,]\d+)*\s*%',
    r'\d+(?:[.,]\d+)*\s*(?:tuổi|năm|tháng|tuần|ngày|giờ|phút|giây)',
    r'\d+(?:[.,]\d+)*\s*(?:ca|người|bệnh nhân|trẻ em|trường hợp)',
    r'\d+(?:[.,]\d+)*\s*(?:mg|g|kg|ml|cm|m|độ)',
    r'\d+/\d+',
    r'\b\d{3,}\b',
]

SPAM = [
    "click vào đây", "đăng ký ngay", "khuyến mãi", "mua ngay",
    "liên hệ ngay", "tư vấn miễn phí",
]


def count_numbers(text: str) -> int:
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in NUMBER_PATTERNS)


def content_hash(text: str) -> str:
    clean = re.sub(r'[^\w]', '', text.lower())
    return hashlib.md5(clean[:500].encode()).hexdigest()


def validate(art: dict) -> tuple[bool, str]:
    text = art.get("full_text", "")
    wc = art.get("word_count", 0)
    if wc < 250:
        return False, f"too_short ({wc}w)"
    if wc > 2500:
        return False, f"too_long ({wc}w)"
    if not art.get("title") or len(art.get("title", "")) < 10:
        return False, "no_title"
    if count_numbers(text) < 3:
        return False, "too_few_numbers"
    if any(p in text.lower() for p in SPAM):
        return False, "spam"
    return True, "ok"


def main():
    if not INPUT.exists():
        print(f"❌ Không thấy {INPUT}")
        return

    articles = []
    with INPUT.open(encoding="utf-8") as f:
        for line in f:
            try:
                articles.append(json.loads(line))
            except Exception:
                pass

    valid, invalid, seen = [], {}, set()
    for art in articles:
        ok, reason = validate(art)
        if not ok:
            invalid[reason] = invalid.get(reason, 0) + 1
            continue
        h = content_hash(art["full_text"])
        if h in seen:
            invalid["duplicate"] = invalid.get("duplicate", 0) + 1
            continue
        seen.add(h)
        art["num_count"] = count_numbers(art["full_text"])
        valid.append(art)

    with OUTPUT.open("w", encoding="utf-8") as f:
        for a in valid:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    df = pd.DataFrame(valid)
    print(f"✅ {len(valid)}/{len(articles)} bài → {OUTPUT.name}")
    print(f"   Loại: {sum(invalid.values())} | reasons: {invalid}")
    if len(valid):
        print(f"   Số từ TB: {df['word_count'].mean():.0f} | Số liệu TB: {df['num_count'].mean():.1f}")
        print(f"\n   Phân bố category:")
        print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
