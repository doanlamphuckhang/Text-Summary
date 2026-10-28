"""Sequential, resumable health corpus collection. No model resources or labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import shutil
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError
from urllib.robotparser import RobotFileParser
from crawl_cafef_health import Page as BasePage, Node, NOISE, NOISE_CLASSES, normalized, health_decision, append_json, load_rows

class Page(BasePage):
    """Handle omitted paragraph/list end tags in minified publisher HTML."""
    def handle_starttag(self, tag, attrs):
        if tag in ('p', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'ul', 'ol', 'table'):
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i].tag == 'p':
                    del self.stack[i:]
                    break
        if tag == 'li' and self.stack[-1].tag == 'li':
            self.stack.pop()
        super().handle_starttag(tag, attrs)

HOSTS = {'suckhoedoisong.vn': 'Sức khỏe & Đời sống', 'tuoitre.vn': 'Tuổi Trẻ'}
UA = 'HealthResearchCrawler/0.2'
SOURCES = [('suckhoedoisong.vn', '1691499', 'y-hoc-360'),
           ('tuoitre.vn', '12', 'suc-khoe'),
           ('suckhoedoisong.vn', '1691512', 'dinh-duong')]
PROMO = ('hotline', 'đặt mua', 'mua ngay', 'khuyến mãi', 'khuyến mại', 'thông tin liên hệ',
         'nội dung được tài trợ', 'bài viết được tài trợ', 'thông tin doanh nghiệp',
         'sản phẩm không phải là thuốc', 'sản phẩm này không phải là thuốc')

def safe_url(url, host):
    p = urlsplit(urljoin('https://' + host, url))
    if p.scheme not in ('http', 'https') or p.hostname != host or p.username or p.password or p.port not in (None, 443, 80):
        raise ValueError('Unexpected URL/redirect')
    return 'https://' + host + p.path

def article_url(url, host):
    try:
        url = safe_url(url, host)
    except ValueError:
        return None
    return url if re.fullmatch(r'/[^/]+-\d{12,25}\.htm', urlsplit(url).path) else None

def discover(html, host, fragment=False):
    root = Page(html).root
    scopes = [root] if fragment else [n for n in root.walk() if
        ('timeline_list' in n.attrs.get('class', '').split()) or
        n.attrs.get('data-marked-zoneid') == 'tto_cate_stream']
    found = {}
    for scope in scopes:
        for n in scope.walk():
            if n.tag == 'a':
                url = article_url(n.attrs.get('href', ''), host)
                if url:
                    found[url] = normalized(n.attrs.get('title', '') or n.text())
    return list(found.items())

def parse_article(html, url, listing):
    host = urlsplit(url).hostname
    nodes = list(Page(html).root.walk())
    meta = {n.attrs.get('property') or n.attrs.get('name'): n.attrs.get('content', '') for n in nodes if n.tag == 'meta'}
    body = next((n for n in nodes if n.attrs.get('data-role') == 'content'), None)
    if body is None:
        raise ValueError('missing_article_body')
    paras = []
    def collect(n):
        classes = set(n.attrs.get('class', '').split())
        if n.tag in NOISE or classes & (NOISE_CLASSES | {'readmore', 'related', 'detail-author', 'author-info'}):
            return
        if n.tag in ('p', 'h2', 'h3', 'h4', 'li', 'blockquote'):
            text = normalized(n.text())
            if text and not text.casefold().startswith(('xem thêm:', 'đọc thêm:', 'mời bạn đọc', 'xem video:')):
                if not paras or paras[-1] != text:
                    paras.append(text)
            return
        for c in n.children:
            if isinstance(c, Node):
                collect(c)
    collect(body)
    canonical = next((n.attrs.get('href') for n in nodes if n.tag == 'link' and n.attrs.get('rel') == 'canonical'), url)
    canonical = article_url(canonical, host)
    if not canonical:
        raise ValueError('invalid_canonical')
    date = meta.get('article:published_time') or next((n.attrs.get('datetime') for n in nodes if n.attrs.get('itemprop') == 'datePublished'), '')
    parsed_date = datetime.fromisoformat(date)
    if parsed_date.tzinfo is None or parsed_date > datetime.now(timezone.utc):
        raise ValueError('missing_timezone_or_future_date')
    authors = [normalized(n.text()) for n in nodes if n.attrs.get('data-role') == 'author']
    authors = list(dict.fromkeys(a for a in authors if a))
    source = '\n\n'.join(paras)
    refs = sorted({urljoin(url, n.attrs['href']) for n in body.walk() if n.tag == 'a' and n.attrs.get('href', '').startswith(('http://', 'https://')) and urlsplit(n.attrs['href']).hostname != host})
    return dict(id=host.split('.')[0] + '_' + re.search(r'-(\d+)\.htm$', canonical).group(1),
        url=canonical, publisher=HOSTS[host], title=normalized(meta.get('og:title', '')),
        pub_date=date, category=meta.get('article:section', 'health_listing'), discovery_url=listing,
        source=source, sapo=normalized(meta.get('og:description', '')), authors=authors,
        external_reference_urls=refs, src_word_count=len(source.split()),
        source_sha256=hashlib.sha256(normalized(source).casefold().encode()).hexdigest(),
        html_sha256=hashlib.sha256(html.encode()).hexdigest(),
        crawled_at=datetime.now(timezone.utc).isoformat(),
        label_status='unlabeled; publisher description is not gold summary')

def quality(row):
    keep, reason, matches = health_decision(row)
    content = (row['title'] + '\n' + row['source']).casefold()
    if any(term in content for term in PROMO):
        return False, 'commercial_content', matches
    if len(row['source'].split('\n\n')) < 4:
        return False, 'too_few_paragraphs', matches
    return keep, reason, matches

class Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, safe_url(newurl, urlsplit(req.full_url).hostname))

class Fetcher:
    def __init__(self):
        self.opener = build_opener(Redirects())
        self.rules, self.last, self.bytes, self.requests = {}, {}, 0, 0
    def stop(self):
        signal = os.environ.get('CODEX_SAFETY_STOP_FILE')
        if signal and Path(signal).exists():
            raise InterruptedError('runner_stop')
    def fetch(self, url):
        host = urlsplit(url).hostname
        if host not in HOSTS:
            raise ValueError('Host not allowed')
        url = safe_url(url, host)
        rules = self.rules.get(host)
        if rules and not rules.can_fetch(UA, url):
            raise PermissionError('robots_denied')
        delay = max(1.5, (rules.crawl_delay(UA) or 0) if rules else 0)
        while time.monotonic() - self.last.get(host, 0) < delay:
            self.stop()
            time.sleep(.1)
        self.stop()
        if self.bytes >= 512 * 1024 * 1024 or self.requests >= 1800:
            raise InterruptedError('download_budget')
        self.last[host] = time.monotonic()
        self.requests += 1
        with self.opener.open(Request(url, headers={'User-Agent': UA, 'Accept-Encoding': 'identity'}), timeout=20) as response:
            data = response.read(4 * 1024 * 1024 + 1)
        self.bytes += len(data)
        if len(data) > 4 * 1024 * 1024:
            raise InterruptedError('response_budget')
        if data.startswith(b'\x1f\x8b'):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                data = stream.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise InterruptedError('decompressed_response_budget')
        return data.decode('utf-8-sig')

def crawl(output, target=901):
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 1024 ** 3:
        raise RuntimeError('Need 1 GiB free')
    rows = load_rows(output / 'articles.jsonl')
    audit = load_rows(output / 'crawl_audit.jsonl')
    visited = {r['url'] for r in rows} | {r['requested_url'] for r in audit}
    hashes = {r['source_sha256'] for r in rows}
    fetcher = Fetcher()
    reason = 'page_limit'
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    try:
        for host in HOSTS:
            text = fetcher.fetch('https://' + host + '/robots.txt')
            rules = RobotFileParser()
            rules.parse(text.splitlines())
            fetcher.rules[host] = rules
            (output / (host + '_robots_' + run + '.txt')).write_text(text, encoding='utf-8')
        for page in range(1, 61):
            for host, zone, slug in SOURCES:
                if len(rows) >= target or len(audit) >= 1500:
                    break
                listing = 'https://' + host + (f'/{slug}.htm' if page == 1 else
                    f'/timelinelist/{zone}/{page}.htm' if host == 'suckhoedoisong.vn' else f'/timeline/{zone}trang-{page}.htm')
                # TuoiTre's observed JS adds a slash after the zone ID.
                if host == 'tuoitre.vn' and page > 1:
                    listing = f'https://{host}/timeline/{zone}/trang-{page}.htm'
                candidates = discover(fetcher.fetch(listing), host, page > 1)
                print(json.dumps(dict(event='listing', page=page, host=host, candidates=len(candidates))), flush=True)
                for url, title in candidates:
                    if len(rows) >= target or len(audit) >= 1500:
                        break
                    if url in visited:
                        continue
                    visited.add(url)
                    result = dict(requested_url=url, listing_url=listing, title=title)
                    try:
                        row = parse_article(fetcher.fetch(url), url, listing)
                        keep, why, terms = quality(row)
                        if row['source_sha256'] in hashes:
                            keep, why = False, 'duplicate_content'
                        result.update(accepted=keep, reason=why, words=row['src_word_count'])
                        if keep:
                            row['filter_evidence'] = terms
                            append_json(output / 'articles.jsonl', row)
                            rows.append(row)
                            hashes.add(row['source_sha256'])
                    except HTTPError as error:
                        result.update(accepted=False, reason='http_' + str(error.code))
                        if error.code in (401, 403, 429) or error.code >= 500:
                            append_json(output / 'crawl_audit.jsonl', result)
                            audit.append(result)
                            raise InterruptedError(result['reason'])
                    except (ValueError, UnicodeError, URLError, TimeoutError, HTTPException) as error:
                        result.update(accepted=False, reason='parse_or_fetch_error', error=str(error))
                    append_json(output / 'crawl_audit.jsonl', result)
                    audit.append(result)
                    print(json.dumps(dict(event='article', attempted=len(audit), accepted=len(rows), reason=result['reason'])), flush=True)
            if len(rows) >= target or len(audit) >= 1500:
                reason = 'target_reached' if len(rows) >= target else 'attempt_budget'
                break
    except (InterruptedError, PermissionError, HTTPError, URLError, HTTPException) as error:
        reason = str(error)
    except Exception as error:
        reason = 'unexpected_error: ' + type(error).__name__
        raise
    finally:
        stats = dict(accepted=len(rows), attempted=len(audit), stop_reason=reason,
            publishers=dict(Counter(r['publisher'] for r in rows)), requests=fetcher.requests,
            downloaded_bytes=fetcher.bytes, rejections=dict(Counter(r['reason'] for r in audit if not r['accepted'])))
        (output / ('stats_' + run + '.json')).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(stats, ensure_ascii=True), flush=True)
    return 0 if reason in ('target_reached', 'page_limit', 'attempt_budget') else 2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/health_multisource_raw_v1'))
    parser.add_argument('--target', type=int, default=901, choices=range(1, 902))
    args = parser.parse_args()
    raise SystemExit(crawl(args.output, args.target))
