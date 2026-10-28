"""Bounded, sequential CafeF health crawler using only Python's standard library.

Never imports training code, downloads media, fabricates summaries or changes old splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.robotparser import RobotFileParser

BASE = 'https://cafef.vn'
LISTING = BASE + '/song/suc-khoe.chn'
ZONE = '188125'  # Read from the live health listing on 2026-10-14.
USER_AGENT = 'HealthResearchCrawler/0.1'
MAX_RESPONSE = 4 * 1024 * 1024
HEALTH_TERMS = (
    'sức khỏe', 'sức khoẻ', 'bác sĩ', 'bệnh viện', 'dinh dưỡng', 'ung thư', 'tiểu đường',
    'đường huyết', 'huyết áp', 'tim mạch', 'đột quỵ', 'gan nhiễm mỡ', 'suy thận',
    'bệnh nhân', 'giấc ngủ', 'tập thể dục', 'sống thọ', 'tuổi thọ', 'trường thọ',
    'vitamin', 'chữa bệnh', 'điều trị', 'vắc xin', 'vaccine', 'béo phì', 'giảm cân',
    'cholesterol', 'tiêu hóa', 'miễn dịch', 'thực phẩm', 'dưỡng chất', 'mất ngủ',
    'đau dạ dày', 'nhiễm trùng', 'xương khớp', 'ngộ độc', 'thuốc độc', 'lão hóa',
)
FINANCE_TERMS = ('cổ phiếu', 'chứng khoán', 'bất động sản', 'lợi nhuận', 'doanh thu', 'giá vàng')
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
NOISE = {'script', 'style', 'noscript', 'iframe', 'figure', 'figcaption', 'button', 'nav'}
NOISE_CLASSES = {'tindnd', 'tinlienquan', 'PhotoCMS_Caption', 'link-content-footer',
                 'VCSortableInPreviewMode', 'ads', 'advertisement', 'related-news'}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


class Node:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()

    def text(self):
        if self.tag in NOISE or NOISE_CLASSES.intersection(self.attrs.get('class', '').split()):
            return ''
        return ''.join(child.text() if isinstance(child, Node) else child for child in self.children)


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node()
        self.stack = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)
        elif tag == 'br':
            self.stack[-1].children.append(' ')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def local_url(url):
    parts = urlsplit(urljoin(BASE, url))
    if parts.scheme not in ('http', 'https') or parts.hostname not in ('cafef.vn', 'www.cafef.vn'):
        raise ValueError('Only public CafeF URLs are allowed')
    if parts.username or parts.password or parts.port not in (None, 80, 443):
        raise ValueError('Invalid CafeF URL')
    return urlunsplit(('https', 'cafef.vn', parts.path, parts.query, ''))


def article_url(url):
    try:
        url = local_url(url)
    except ValueError:
        return None
    parts = urlsplit(url)
    if re.fullmatch(r'/[^/]+-\d{12,25}\.chn', parts.path):
        return urlunsplit((parts.scheme, parts.netloc, parts.path, '', ''))
    return None


def discover(html, fragment=False):
    root = Page(html).root
    scopes = [root] if fragment else [n for n in root.walk()
        if re.search(r'zone(?:id)?' + ZONE + r'(?!\d)', n.attrs.get('data-cd-key', ''))]
    found = {}
    for scope in scopes:
        for node in scope.walk():
            if node.tag != 'a':
                continue
            url = article_url(node.attrs.get('href', ''))
            title = normalized(node.attrs.get('title', '') or node.text())
            if url and title:
                found[url] = title
    return list(found.items())


def parse_article(html, url):
    root = Page(html).root
    nodes = list(root.walk())
    meta = {n.attrs.get('property') or n.attrs.get('name'): n.attrs.get('content', '')
            for n in nodes if n.tag == 'meta'}
    title = normalized(meta.get('og:title', ''))
    if not title:
        title = next((normalized(n.text()) for n in nodes if n.tag == 'h1'), '')
    body = next((n for n in nodes if 'detail-content' in n.attrs.get('class', '').split()
                 or n.attrs.get('data-role') == 'content'), None)
    if body is None:
        raise ValueError('Article body container missing')
    paragraphs = []

    def collect(node):
        if node.tag in NOISE or NOISE_CLASSES.intersection(node.attrs.get('class', '').split()):
            return
        if node.tag in ('p', 'h2', 'h3', 'h4', 'li', 'blockquote'):
            text = normalized(node.text())
            if text and (not paragraphs or paragraphs[-1] != text):
                paragraphs.append(text)
            return
        for child in node.children:
            if isinstance(child, Node):
                collect(child)

    collect(body)
    published = next((n.attrs.get('datetime', '') for n in nodes
                      if n.attrs.get('itemprop') == 'datePublished' and n.attrs.get('datetime')), '')
    canonical = next((n.attrs.get('href') for n in nodes
                      if n.tag == 'link' and n.attrs.get('rel') == 'canonical'), url)
    canonical = article_url(canonical)
    if not canonical:
        raise ValueError('Unexpected canonical URL')
    source = '\n\n'.join(paragraphs)
    return dict(id='cafef_' + re.search(r'-(\d+)\.chn$', canonical).group(1), url=canonical,
                title=title, pub_date=published, category='suc-khoe', publisher='CafeF',
                discovery_url=LISTING, source=source, sapo=normalized(meta.get('og:description', '')),
                src_word_count=len(source.split()), crawled_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(normalized(source).casefold().encode()).hexdigest())


def health_decision(record, min_words=250, max_words=2500):
    title, body = record['title'].casefold(), record['source'].casefold()
    matches = [term for term in HEALTH_TERMS if term in body]
    if not title:
        return False, 'missing_title', matches
    if any(term in title for term in FINANCE_TERMS):
        return False, 'finance_title', matches
    if not min_words <= record['src_word_count'] <= max_words:
        return False, 'word_count_outside_limits', matches
    # Listing membership plus body evidence; a menu/tag alone cannot pass this filter.
    if len(matches) < 3:
        return False, 'insufficient_health_content', matches
    return True, 'health_listing_and_content', matches


def shingles(text):
    words = normalized(text).casefold().split()
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


class CafeFRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, local_url(newurl))


class Fetcher:
    def __init__(self, delay=1.5, max_bytes=64 * 1024 * 1024):
        self.delay, self.max_bytes, self.bytes, self.requests, self.last = delay, max_bytes, 0, 0, 0.
        self.opener = build_opener(CafeFRedirects())
        self.robots = None

    def check_stop(self):
        stop = os.environ.get('CODEX_SAFETY_STOP_FILE')
        if stop and Path(stop).exists():
            raise InterruptedError('Runner requested cooperative stop')

    def fetch(self, url):
        self.check_stop()
        url = local_url(url)
        if self.robots and not self.robots.can_fetch(USER_AGENT, url):
            raise PermissionError('robots.txt disallows this URL')
        if self.bytes >= self.max_bytes:
            raise InterruptedError('Total download limit reached')
        deadline = self.last + self.delay
        while time.monotonic() < deadline:
            self.check_stop()
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        self.last = time.monotonic()
        self.requests += 1
        request = Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'text/html,text/plain'})
        with self.opener.open(request, timeout=20) as response:
            local_url(response.url)
            payload = response.read(min(MAX_RESPONSE, self.max_bytes - self.bytes) + 1)
            self.bytes += len(payload)
            if len(payload) > MAX_RESPONSE or self.bytes > self.max_bytes:
                raise InterruptedError('Response/download byte limit reached')
            return payload.decode('utf-8-sig', errors='strict')


def append_json(path, record):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def load_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def crawl(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 512 * 1024 * 1024:
        raise RuntimeError('Need at least 512 MiB free on destination volume')
    articles_path, audit_path = output / 'articles.jsonl', output / 'crawl_audit.jsonl'
    rows, audit = load_rows(articles_path), load_rows(audit_path)
    visited = {r['requested_url'] for r in audit} | {r['url'] for r in rows}
    fingerprints = {r['source_sha256'] for r in rows}
    ids = {r['id'] for r in rows}
    existing_shingles = [(r['id'], shingles(r['source'])) for r in rows]
    fetched = len(visited)
    fetcher = Fetcher(delay=args.delay)
    reason, new_count, listing_pages = 'article_limit_reached', 0, 0
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        robots_text = fetcher.fetch(BASE + '/robots.txt')
        rules = RobotFileParser()
        rules.parse(robots_text.splitlines())
        fetcher.robots = rules
        fetcher.delay = max(fetcher.delay, rules.crawl_delay(USER_AGENT) or 0)
        (output / ('robots_' + run_id + '.txt')).write_text(robots_text, encoding='utf-8')
        for page in range(1, args.max_pages + 1):
            if fetched >= args.max_articles:
                break
            page_url = LISTING if page == 1 else f'{BASE}/timelinelist/{ZONE}/{page}.chn'
            html = fetcher.fetch(page_url)
            listing_pages += 1
            candidates = discover(html, fragment=page > 1)
            print(json.dumps({'event': 'listing', 'page': page, 'candidates': len(candidates)}, ensure_ascii=True), flush=True)
            if not candidates:
                reason = 'no_more_listing_candidates'
                break
            for url, title in candidates:
                if fetched >= args.max_articles:
                    break
                if url in visited:
                    continue
                fetcher.check_stop()
                fetched += 1
                visited.add(url)
                result = dict(requested_url=url, listing_url=page_url, title=title,
                              fetched_at=datetime.now(timezone.utc).isoformat())
                try:
                    record = parse_article(fetcher.fetch(url), url)
                    keep, why, matches = health_decision(record)
                    if record['id'] in ids or record['source_sha256'] in fingerprints:
                        keep, why = False, 'duplicate_id_or_content'
                    tokens = shingles(record['source'])
                    if keep:
                        for other_id, previous in existing_shingles:
                            union = tokens | previous
                            if union and len(tokens & previous) / len(union) >= .85:
                                keep, why = False, 'near_duplicate'
                                result['duplicate_of'] = other_id
                                break
                    result.update(accepted=keep, reason=why, health_terms=matches,
                                  src_word_count=record['src_word_count'], canonical_url=record['url'])
                    if keep:
                        record['filter_evidence'] = matches
                        record['label_status'] = 'unlabeled; sapo is publisher description, not validated gold summary'
                        append_json(articles_path, record)
                        rows.append(record)
                        ids.add(record['id'])
                        fingerprints.add(record['source_sha256'])
                        existing_shingles.append((record['id'], tokens))
                        new_count += 1
                except HTTPError as error:
                    result.update(accepted=False, reason='http_error', status=error.code)
                    if error.code in (401, 403, 429) or error.code >= 500:
                        append_json(audit_path, result)
                        audit.append(result)
                        raise InterruptedError(f'HTTP {error.code}; stopped without automatic retries')
                except (ValueError, UnicodeError, URLError, TimeoutError) as error:
                    result.update(accepted=False, reason='fetch_or_parse_error', error=str(error))
                append_json(audit_path, result)
                audit.append(result)
                print(json.dumps({'event': 'article', 'attempted': fetched, 'kept': len(rows),
                                  'reason': result['reason']}, ensure_ascii=True), flush=True)
        else:
            reason = 'listing_page_limit_reached'
    except (InterruptedError, KeyboardInterrupt) as error:
        reason = str(error)
    finally:
        dates = sorted(r['pub_date'] for r in rows if r['pub_date'])
        stats = dict(run_id=run_id, listing=LISTING, zone_id=ZONE, max_articles=args.max_articles,
                     attempted_total=fetched, accepted_total=len(rows), accepted_this_run=new_count,
                     listing_pages_this_run=listing_pages, requests_this_run=fetcher.requests,
                     downloaded_bytes_this_run=fetcher.bytes, stop_reason=reason,
                     min_pub_date=dates[0] if dates else None, max_pub_date=dates[-1] if dates else None,
                     min_words=min((r['src_word_count'] for r in rows), default=0),
                     max_words=max((r['src_word_count'] for r in rows), default=0),
                     rejection_counts=dict(Counter(r['reason'] for r in audit if not r['accepted'])),
                     summary_labels_created=False, train_val_test_created=False)
        (output / ('stats_' + run_id + '.json')).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'finished', **stats}, ensure_ascii=True), flush=True)
    return 0 if reason in ('article_limit_reached', 'no_more_listing_candidates', 'listing_page_limit_reached') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/cafef_health'))
    parser.add_argument('--max-articles', type=int, default=100, choices=range(1, 101), metavar='1..100')
    parser.add_argument('--max-pages', type=int, default=10, choices=range(1, 11), metavar='1..10')
    parser.add_argument('--delay', type=float, default=1.5)
    args = parser.parse_args()
    if args.delay < 1:
        parser.error('--delay must be at least 1 second')
    return crawl(args)


if __name__ == '__main__':
    raise SystemExit(main())
