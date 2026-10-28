"""Independent artifact-level invariants; no downloads or model execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

if __name__ == '__main__':
    output=Path(sys.argv[1])
    report=json.loads((output/'report.json').read_text(encoding='utf-8'))
    clean=read(output/'clean_articles.jsonl')
    manifest=read(output/'split_manifest.jsonl')
    frozen={r['id']:r for r in read(Path(report['frozen_manifest']))}
    rows=[r for split in ('train','val','test') for r in read(output/'draft_splits'/split/'dataset.jsonl')]
    queue=read(output/'review_queue.jsonl')
    rejected=read(output/'rejected.jsonl')
    assert len(rows)==len(clean)==len(manifest)==len(queue)==report['retained']
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['content_key'] for r in rows})==len(rows)
    assert len(rows)+len(rejected)==report['input_records']
    assert {r['id'] for r in rows}=={r['id'] for r in manifest}
    assert Counter(r['split'] for r in rows)==report['split_counts']
    by_id={r['id']:r for r in rows}
    groups={}
    for r in rows:
        assert hashlib.sha256(r['source'].encode()).hexdigest()==r['prepared_source_sha256']
        assert r['src_word_count']==len(r['source'].split())
        assert r['tgt_word_count']==len(r['summary'].split())
        assert r['label_status']=='pending_human_review'
        assert r['quality_tag']=='unreviewed'
        assert r['summary']==' '.join(e['text'] for e in r['evidence_spans'])
        assert all(r['source'][e['start']:e['end']]==e['text'] for e in r['evidence_spans'])
        if r['id'] in frozen:
            assert r['split']==frozen[r['id']]['split']
            assert r['prepared_source_sha256']==frozen[r['id']]['source_sha256']
        groups.setdefault(r['group_id'],set()).add(r['split'])
    assert all(len(s)==1 for s in groups.values())
    for link in read(output/'group_links.jsonl'):
        if link['a'] in by_id and link['b'] in by_id:
            assert by_id[link['a']]['split']==by_id[link['b']]['split']
    assert all(r['approved'] is False and r['reviewer']=='' for r in queue)
    for path,expected in report['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()==expected
    result=dict(passed=True,records=len(rows),groups=len(groups),frozen_records_preserved=sum(r['id'] in frozen for r in rows),human_reviewed=0,training_ready=False)
    with (output/'validation.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))
