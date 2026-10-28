"""Small bounded source inspection; no media or model downloads."""
from pathlib import Path
from urllib.request import Request, urlopen
import time
import hashlib
import os

OUT = Path('outputs/source-probes')
URLS = [
    'https://suckhoedoisong.vn/robots.txt',
    'https://suckhoedoisong.vn/y-hoc-360.htm',
    'https://suckhoedoisong.vn/dinh-duong.htm',
    'https://tuoitre.vn/robots.txt',
    'https://tuoitre.vn/suc-khoe.htm',
]
if __name__ == '__main__':
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    for url in sys.argv[1:] or URLS:
        if os.environ.get('CODEX_SAFETY_STOP_FILE') and Path(os.environ['CODEX_SAFETY_STOP_FILE']).exists():
            raise SystemExit('stop requested')
        with urlopen(Request(url, headers={'User-Agent': 'HealthResearchCrawler/0.2'}), timeout=20) as response:
            data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError('Response too large')
        path = OUT / (hashlib.sha256(url.encode()).hexdigest()[:12] + '.html')
        path.write_bytes(data)
        print(url, str(path), len(data), flush=True)
        time.sleep(1.5)
