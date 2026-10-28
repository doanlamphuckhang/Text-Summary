"""Print structure hints (scripts, zone IDs, article links, meta, body container) of saved probe pages."""
import gzip
import re
from pathlib import Path

from crawl_cafef_health import Page


def inspect(directory=Path('outputs/source-probes')):
    for path in sorted(directory.glob('*.html')):
        data = path.read_bytes()
        if data.startswith(b'\x1f\x8b'):
            data = gzip.decompress(data)
        text = data.decode('utf-8-sig')
        nodes = list(Page(text).root.walk())
        print('\nFILE', path.name)
        print('SCRIPTS', [n.attrs.get('src') for n in nodes if n.tag == 'script' and n.attrs.get('src')])
        print('ZONES', re.findall(r'.{0,70}(?:hdZoneId|zoneId|zoneid|timeline|loadmore).{0,100}', text, re.I)[-15:])
        print('ARTICLES', list(dict.fromkeys(n.attrs.get('href', '') for n in nodes
                                             if n.tag == 'a' and re.search(r'-\d{10,25}\.htm', n.attrs.get('href', ''))))[:3])
        print('META', [n.attrs for n in nodes if n.tag == 'meta'][:12])
        print('BODY', [n.attrs for n in nodes
                       if n.attrs.get('data-role') == 'content' or 'detail-content' in n.attrs.get('class', '')])


if __name__ == '__main__':
    inspect()
