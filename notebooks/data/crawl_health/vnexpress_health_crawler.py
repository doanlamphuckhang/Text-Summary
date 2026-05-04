"""
VnExpress Sức Khỏe Crawler
==========================
Thu thập bài báo y tế / dịch tễ / sức khỏe từ VnExpress.

Domain "tin y tế / dịch tễ" — được đề bài liệt kê là một trong các miền
chuyên biệt giàu số liệu phù hợp cho tóm tắt văn bản.

Cách dùng:
    pip install requests beautifulsoup4 tqdm pandas
    python data/crawl_health/vnexpress_health_crawler.py

Output: cafef_raw.jsonl → vnexpress_health_raw.jsonl
"""
import hashlib
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
TARGET_ARTICLES = 1600
DELAY_MIN = 0.25
DELAY_MAX = 0.75
MIN_WORDS = 250
MAX_WORDS = 2500
MAX_PAGES = 80
MAX_WORKERS = 4

OUTPUT_FILE = ROOT / "vnexpress_health_raw.jsonl"
LOG_FILE = ROOT / "vnexpress_health_crawl.log"

# VnExpress sức khỏe có nhiều subcategory dùng chung path /suc-khoe/...
CATEGORIES = {
    "suc-khoe":               "https://vnexpress.net/suc-khoe-p{p}",
    "tin-tuc":                "https://vnexpress.net/suc-khoe/tin-tuc-p{p}",
    "tu-van":                 "https://vnexpress.net/suc-khoe/tu-van-p{p}",
    "cac-benh":               "https://vnexpress.net/suc-khoe/cac-benh-p{p}",
    "dinh-duong":             "https://vnexpress.net/suc-khoe/dinh-duong-p{p}",
    "khoe-dep":               "https://vnexpress.net/suc-khoe/khoe-dep-p{p}",
    "vaccine":                "https://vnexpress.net/suc-khoe/vaccine-p{p}",
    "song-khoe":              "https://vnexpress.net/suc-khoe/song-khoe-p{p}",
}

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.6,en;q=0.5",
    "Referer":         "https://vnexpress.net/",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

ARTICLE_URL_RE = re.compile(r'^https?://vnexpress\.net/[a-z0-9\-]+-\d{7,9}\.html$')

# Chỉ giữ bài có ≥ 3 số liệu (đề bài: miền chuyên biệt giàu số liệu)
NUMBER_PATTERNS = [
    r'\d+(?:[.,]\d+)*\s*%',
    r'\d+(?:[.,]\d+)*\s*(?:tuổi|năm|tháng|tuần|ngày|giờ|phút|giây)',
    r'\d+(?:[.,]\d+)*\s*(?:ca|người|bệnh nhân|trẻ em|trường hợp)',
    r'\d+(?:[.,]\d+)*\s*(?:mg|g|kg|ml|cm|m|km|độ|°C)',
    r'\d+(?:[.,]\d+)*\s*(?:USD|VND|đồng|triệu|tỷ|nghìn)',
    r'\d+/\d+',
    r'\b\d{3,}\b',
]


def safe_get(url: str, retries: int = 3) -> Optional[requests.Response]:
    """GET với retry và backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                time.sleep(10 + attempt * 5)
            elif resp.status_code in (403, 404):
                return None
        except requests.RequestException as e:
            log.warning(f"Request error (lần {attempt + 1}): {e}")
            time.sleep(2 ** attempt)
    return None


def url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def has_numbers(text: str, min_count: int = 3) -> bool:
    count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in NUMBER_PATTERNS)
    return count >= min_count


def collect_urls(slug: str, url_template: str) -> dict[str, str]:
    """Lấy URL bài báo từ trang chuyên mục, tối đa MAX_PAGES trang."""
    found: dict[str, str] = {}
    consecutive_empty = 0

    for page in range(1, MAX_PAGES + 1):
        page_url = url_template.format(p=page)
        resp = safe_get(page_url)
        if not resp:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        new_urls = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href.startswith("http"):
                continue
            if ARTICLE_URL_RE.match(href) and href not in found:
                new_urls.append(href)

        if not new_urls:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                log.info(f"  [{slug}] Hết bài tại page {page}")
                break
        else:
            consecutive_empty = 0
            for u in new_urls:
                found[u] = slug
            log.info(f"  [{slug}] Page {page}: +{len(new_urls)} (tổng {len(found)})")

        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return found


def parse_article(url: str, category: str) -> Optional[dict]:
    resp = safe_get(url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Tiêu đề
    title = ""
    h1 = soup.find("h1", class_=re.compile(r"title-detail|title", re.I))
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        title = og.get("content", "").strip() if og else ""
    if not title or len(title) < 10:
        return None

    # Sapo
    sapo = ""
    desc_tag = soup.find("p", class_=re.compile(r"description", re.I))
    if desc_tag:
        sapo = desc_tag.get_text(" ", strip=True)
    if not sapo:
        og_d = soup.find("meta", property="og:description") or soup.find("meta", {"name": "description"})
        sapo = og_d.get("content", "").strip() if og_d else ""

    # Body — VnExpress dùng article.fck_detail hoặc div .sidebar-1 article
    body_tag = soup.find("article", class_=re.compile(r"fck_detail|content_detail", re.I))
    if not body_tag:
        body_tag = soup.find("div", class_=re.compile(r"fck_detail|sidebar-1.*detail", re.I))
    if not body_tag:
        return None

    # Bỏ các thành phần không phải nội dung chính
    for tag in body_tag.find_all(["script", "style", "iframe", "figure", "table", "aside"]):
        tag.decompose()
    for tag in body_tag.find_all(class_=re.compile(r"related|ads|share|author|tag|box-tag|video|image", re.I)):
        tag.decompose()

    paragraphs = []
    for p in body_tag.find_all("p"):
        # Bỏ class "Normal author" thường là chữ ký
        cls = " ".join(p.get("class") or [])
        if "author" in cls.lower():
            continue
        txt = p.get_text(" ", strip=True)
        if len(txt) > 20:
            paragraphs.append(txt)
    body = "\n".join(paragraphs)

    full_text = (sapo + "\n" + body).strip() if sapo and sapo not in body else body.strip()

    # Ngày đăng
    pub_date = ""
    for attr in [{"itemprop": "datePublished"}, {"property": "article:published_time"}, {"name": "pubdate"}]:
        m = soup.find("meta", attr)
        if m and m.get("content"):
            pub_date = m["content"]
            break
    if not pub_date:
        span_date = soup.find("span", class_=re.compile(r"date", re.I))
        if span_date:
            pub_date = span_date.get_text(strip=True)

    wc = len(full_text.split())
    if not (MIN_WORDS <= wc <= MAX_WORDS):
        return None
    if not has_numbers(full_text):
        return None

    return {
        "id":         url_id(url),
        "url":        url,
        "category":   category,
        "title":      title,
        "sapo":       sapo,
        "body":       body,
        "full_text":  full_text,
        "word_count": wc,
        "pub_date":   pub_date,
        "crawled_at": datetime.now().isoformat(),
    }


def parse_article_with_delay(url: str, category: str) -> Optional[dict]:
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
    return parse_article(url, category)


def main():
    seen_ids: set[str] = set()
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    seen_ids.add(obj["id"])
                except Exception:
                    pass
        log.info(f"Resume: đã có {len(seen_ids)} bài")

    if len(seen_ids) >= TARGET_ARTICLES:
        log.info("Đã đủ bài!")
        return

    log.info("=" * 60)
    log.info("BƯỚC 1: Thu thập URL")
    log.info("=" * 60)

    all_urls: dict[str, str] = {}
    for slug, tmpl in CATEGORIES.items():
        log.info(f"\n📂 {slug}")
        urls = collect_urls(slug, tmpl)
        if urls:
            all_urls.update({u: c for u, c in urls.items() if u not in all_urls})
            log.info(f"  ✓ +{len(urls)} URL (tổng pool: {len(all_urls)})")

    log.info(f"\nTổng URL pool: {len(all_urls)}")

    log.info("=" * 60)
    log.info("BƯỚC 2: Parse nội dung")
    log.info("=" * 60)

    pending = [(u, c) for u, c in all_urls.items() if url_id(u) not in seen_ids]
    log.info(f"Cần parse: {len(pending)} URL")

    success = len(seen_ids)
    skipped = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(parse_article_with_delay, url, cat): (url, cat)
                for url, cat in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Crawling"):
                if success >= TARGET_ARTICLES:
                    break
                try:
                    art = future.result()
                except Exception as e:
                    url, _ = futures[future]
                    log.warning(f"Parse error {url}: {e}")
                    art = None

                if art and art["id"] not in seen_ids:
                    f.write(json.dumps(art, ensure_ascii=False) + "\n")
                    f.flush()
                    seen_ids.add(art["id"])
                    success += 1
                    if success % 100 == 0:
                        log.info(f"  crawled {success} articles")
                else:
                    skipped += 1

    log.info("=" * 60)
    log.info(f"Done: {success} articles | Skipped: {skipped}")
    return

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        for url, cat in tqdm(pending, desc="Crawling"):
            if success >= TARGET_ARTICLES:
                break
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            art = parse_article(url, cat)
            if art:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")
                f.flush()
                seen_ids.add(art["id"])
                success += 1
                if success % 100 == 0:
                    log.info(f"  ✓ {success} bài")
            else:
                skipped += 1

    log.info("=" * 60)
    log.info(f"✅ Tổng: {success} bài | Bỏ qua: {skipped}")


if __name__ == "__main__":
    main()
