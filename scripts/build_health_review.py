"""Build an offline review page; no external scripts or automatic approvals."""
import argparse
import json
from pathlib import Path

PAGE = '''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'">
<title>Duyệt dữ liệu sức khỏe</title><style>
body{font:16px/1.6 system-ui;margin:0;color:#18312b;background:#f4f7f5}header{padding:20px 3%;background:#163d34;color:white}main{padding:20px 3%}button,input,select,textarea{font:inherit;padding:8px;border:1px solid #aebdb5;border-radius:6px}button{cursor:pointer;background:#fff}nav{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}article,section{padding:20px;background:white;border:1px solid #d7e0da;border-radius:10px}#source{white-space:pre-wrap}mark{background:#d9edb8}textarea{box-sizing:border-box;width:100%;min-height:260px}small{color:#527064}#flags{color:#9b491b}a{color:#1b7259}@media(max-width:800px){.grid{grid-template-columns:1fr}}.hint{background:#fff6da;padding:10px}
</style><header><h1>Duyệt dữ liệu sức khỏe</h1><p>Bản tóm tắt nháp, chưa phải nhãn chuẩn. Màu xanh đánh dấu câu được trích trong bài gốc.</p></header>
<main><nav><input id="search" placeholder="Tìm tiêu đề"><select id="split"><option value="">Mọi tập</option><option>train</option><option>val</option><option>test</option></select><select id="flag"><option value="">Mọi cờ kiểm tra</option></select><button id="prev">← Trước</button><button id="next">Tiếp →</button><button id="export">Xuất nhãn đã duyệt</button></nav>
<p id="position"></p><div class="grid"><article><h2 id="title"></h2><p id="metadata"></p><a id="link" target="_blank" rel="noreferrer">Mở bài gốc</a><p id="source"></p></article><section><h2>Bản tóm tắt để chỉnh sửa</h2><p id="flags"></p><textarea id="summary"></textarea><p id="ratio"></p><p class="hint">Kiểm tra đủ ý chính, chủ thể, số liệu, phủ định, điều kiện và đối tượng áp dụng. Không biến phát biểu của nguồn thành kết luận y khoa đã xác thực.</p><details><summary>Câu có điều kiện/cảnh báo chưa được chọn</summary><div id="caveats"></div></details><p><input id="reviewer" placeholder="Tên người duyệt"></p><textarea id="notes" placeholder="Ghi chú, sửa lỗi hoặc lý do loại" style="min-height:100px"></textarea><p><label><input type="checkbox" id="approved"> Tôi đã đọc đối chiếu và duyệt nhãn này</label></p><small>Thay đổi chỉ giữ trong tab hiện tại. Hãy xuất JSONL trước khi đóng. Xuất file không tự sửa dataset hay bắt đầu train.</small></section></div></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('data').textContent), edits=new Map();
const el=id=>document.getElementById(id);let visible=rows,index=0,current=null;
const words=s=>s.trim()?s.trim().split(/\\s+/).length:0;
// Older review queues (prepare_health_data.py) lack some fields; fall back instead of crashing.
const flagsOf=r=>r.review_flags||[], srcWords=r=>r.src_word_count||words(r.source);
for(const flag of [...new Set(rows.flatMap(flagsOf))].sort()){const o=document.createElement('option');o.value=o.textContent=flag;el('flag').append(o)}
function save(){if(!current)return;edits.set(current.id,{id:current.id,source_sha256:current.prepared_source_sha256||current.source_sha256,summary:el('summary').value,approved:el('approved').checked,reviewer:el('reviewer').value.trim(),notes:el('notes').value,split:current.split})}
function ratio(){if(current)el('ratio').textContent=`${words(el('summary').value)} từ / ${srcWords(current)} từ nguồn · ${(100*words(el('summary').value)/srcWords(current)).toFixed(1)}% (mục tiêu 20–25%)`}
function render(){current=visible[index];el('position').textContent=visible.length?`${index+1}/${visible.length} bài trong bộ lọc · ${rows.length} bài toàn bộ`:'Không có bài phù hợp';if(!current){document.querySelector('.grid').hidden=true;return}document.querySelector('.grid').hidden=false;const r=current,e=edits.get(r.id);el('title').textContent=r.title;el('metadata').textContent=`${r.publisher||''} · ${r.pub_date||''} · ${r.split} · ${(r.authors||[]).join(', ')||'Chưa có tác giả'}`;el('link').href=r.url;el('flags').textContent=flagsOf(r).join(' · ');el('source').replaceChildren();let p=0;for(const s of r.evidence_spans||[]){el('source').append(document.createTextNode(r.source.slice(p,s.start)));const mark=document.createElement('mark');mark.textContent=r.source.slice(s.start,s.end);el('source').append(mark);p=s.end}el('source').append(document.createTextNode(r.source.slice(p)));el('summary').value=e?e.summary:r.summary;el('notes').value=e?e.notes:'';el('reviewer').value=e?e.reviewer:'';el('approved').checked=e?e.approved:false;el('caveats').replaceChildren();for(const s of r.omitted_caveat_candidates||[]){const p=document.createElement('p');p.textContent=s;el('caveats').append(p)}ratio()}
function filter(){save();visible=rows.filter(r=>(!el('split').value||r.split===el('split').value)&&(!el('flag').value||flagsOf(r).includes(el('flag').value))&&r.title.toLocaleLowerCase('vi').includes(el('search').value.toLocaleLowerCase('vi')));index=0;render()}
for(const id of ['search','split','flag'])el(id).oninput=filter;
el('prev').onclick=()=>{save();index=Math.max(0,index-1);render()};el('next').onclick=()=>{save();index=Math.min(visible.length-1,index+1);render()};el('summary').oninput=()=>{el('approved').checked=false;ratio()};
el('export').onclick=()=>{save();const accepted=[...edits.values()].filter(e=>e.approved);if(accepted.some(e=>!e.reviewer||!e.summary.trim())){alert('Nhãn được duyệt phải có người duyệt và nội dung tóm tắt.');return}if(!accepted.length){alert('Chưa có nhãn được duyệt.');return}const blob=new Blob([accepted.map(e=>JSON.stringify(e)).join('\\n')+'\\n'],{type:'application/x-ndjson;charset=utf-8'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='health-reviewed-annotations.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};render();
</script></html>'''

def build(directory):
    """Write review.html next to review_queue.jsonl; never overwrites an existing page."""
    queue = directory / 'review_queue.jsonl'
    rows = [json.loads(line) for line in queue.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'{queue} has no records to review')
    # Escape characters that could close the <script> element inside the JSON payload.
    payload = json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    target = directory / 'review.html'
    with target.open('x', encoding='utf-8') as stream:
        stream.write(PAGE.replace('PAYLOAD', payload))
    return dict(review_records=len(rows), output=str(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='prepared directory containing review_queue.jsonl')
    try:
        print(json.dumps(build(parser.parse_args().directory), ensure_ascii=False))
    except FileExistsError as error:
        raise SystemExit(f'{error.filename} already exists; move it away first (it may hold unsaved review work)') from None
