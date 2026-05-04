import json
import os
import random
import re
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Paths & constants ─────────────────────────────────────────────────────────
UI_DIR = Path(__file__).resolve().parent
PROJECT_DIR = UI_DIR.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_DIR / "best_model"))
DATA_PATH = Path(os.environ.get("DATA_PATH", PROJECT_DIR / "data" / "test" / "dataset.jsonl"))

MAX_INPUT_LEN  = int(os.environ.get("MAX_INPUT_LEN",  "1024"))
MAX_TARGET_LEN = int(os.environ.get("MAX_TARGET_LEN", "384"))
NO_REPEAT_NGRAM = int(os.environ.get("NO_REPEAT_NGRAM", "3"))
TARGET_RATIO = 0.225
NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")

# ── Metric helpers ────────────────────────────────────────────────────────────
def extract_numbers(text):
    return set(NUM_RE.findall(text or ""))

def number_recall(source, pred):
    src = extract_numbers(source)
    if not src: return 1.0
    return len(src & extract_numbers(pred)) / len(src)

def number_hallucination(source, pred):
    pred_n = extract_numbers(pred)
    if not pred_n: return 0.0
    return len(pred_n - extract_numbers(source)) / len(pred_n)

def hallucination_abs(source, pred):
    src = extract_numbers(source)
    if not src: return float(len(extract_numbers(pred)) > 0)
    return len(extract_numbers(pred) - src) / len(src)

def length_ratio(source, pred):
    return len((pred or "").split()) / max(len((source or "").split()), 1)

def length_score(source, pred, target=TARGET_RATIO):
    r = length_ratio(source, pred)
    if r <= target: return max(0.0, 1.0 - (target - r) / target)
    return max(0.0, 1.0 - 1.5 * (r - target) / target)

def number_avoidance(source, pred):
    src, pred_n = extract_numbers(source), extract_numbers(pred)
    if not src: return 0.0
    return float(len(pred_n) == 0 or len(pred_n & src) / len(src) < 0.20)

def score_candidate(source, cand):
    return (0.55 * number_recall(source, cand)
            + 0.35 * length_score(source, cand)
            - 0.55 * number_hallucination(source, cand)
            - 0.10 * hallucination_abs(source, cand)
            - 0.10 * number_avoidance(source, cand))

# ── Data ─────────────────────────────────────────────────────────────────────
def load_examples(limit=8):
    if not DATA_PATH.exists(): return [], {}
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows: return [], {}
    rng = random.Random(42)
    rows = sorted(rows, key=lambda r: abs(r.get("src_word_count", 700) - 650))
    sample = rows[:max(limit * 3, limit)]
    rng.shuffle(sample)
    sample = sample[:limit]
    choices, mapping = [], {}
    for i, row in enumerate(sample, 1):
        title = row.get("title") or row.get("id", f"Mẫu {i}")
        label = f"{i}. {title[:82]}"
        choices.append(label)
        mapping[label] = row
    return choices, mapping

EXAMPLE_CHOICES, EXAMPLE_MAP = load_examples()

# ── Model ─────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Không tìm thấy MODEL_DIR: {MODEL_DIR}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32
    tok    = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model  = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype)
    model.to(device).eval()
    model.config.use_cache = True
    return tok, model, device

@torch.no_grad()
def generate_candidates(source, mode, beams, candidates, temperature):
    tok, model, device = load_model()
    batch = tok([source], max_length=MAX_INPUT_LEN, truncation=True,
                padding=True, return_tensors="pt").to(device)

    if mode == "Nhanh":
        ids = model.generate(**batch, max_length=MAX_TARGET_LEN,
                             num_beams=max(1, beams),
                             no_repeat_ngram_size=NO_REPEAT_NGRAM,
                             length_penalty=1.0, early_stopping=beams > 1,
                             use_cache=True)
        decoded = tok.batch_decode(ids, skip_special_tokens=True)
        return decoded, [score_candidate(source, decoded[0])]

    k = max(2, int(candidates))
    num_beams = max(k * 2, int(beams), 4)
    ids = model.generate(**batch, max_length=MAX_TARGET_LEN,
                         num_beams=num_beams, num_beam_groups=k,
                         num_return_sequences=k, diversity_penalty=0.45,
                         no_repeat_ngram_size=NO_REPEAT_NGRAM,
                         length_penalty=1.0, early_stopping=True, use_cache=True)
    decoded = tok.batch_decode(ids, skip_special_tokens=True)

    if mode == "Chất lượng + Rerank" and temperature > 0:
        sampled = model.generate(**batch, max_length=MAX_TARGET_LEN,
                                 do_sample=True, top_p=0.90,
                                 temperature=float(temperature),
                                 no_repeat_ngram_size=NO_REPEAT_NGRAM,
                                 use_cache=True)
        decoded.extend(tok.batch_decode(sampled, skip_special_tokens=True))

    unique, seen = [], set()
    for t in decoded:
        k2 = " ".join(t.split())
        if k2 and k2 not in seen:
            unique.append(t); seen.add(k2)
    return unique, [score_candidate(source, c) for c in unique]

# ── UI helpers ────────────────────────────────────────────────────────────────
def make_metric_html(source, summary, elapsed, mode):
    src_w = len(source.split()); sum_w = len(summary.split())
    src_n = extract_numbers(source); pred_n = extract_numbers(summary)
    kept = src_n & pred_n; hall = pred_n - src_n
    ratio = length_ratio(source, summary)
    nr = number_recall(source, summary)
    nh = number_hallucination(source, summary)
    c_nr = "#16a34a" if nr >= .8 else ("#d97706" if nr >= .5 else "#dc2626")
    c_nh = "#16a34a" if nh < .1  else ("#d97706" if nh < .3  else "#dc2626")

    def pct(x): return f"{x*100:.1f}%"

    cards = [
        ("Số liệu giữ lại",   pct(nr),          f"{len(kept)}/{len(src_n)} số trong bài", c_nr),
        ("Ảo giác số liệu",   pct(nh),           f"{len(hall)} số ngoài nguồn",            c_nh),
        ("Tỉ lệ độ dài",      f"{ratio:.3f}",    f"{sum_w}/{src_w} từ",                    "#0f172a"),
        ("Thời gian",         f"{elapsed:.1f}s", mode,                                     "#0f172a"),
    ]
    items = "".join(
        f"""<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                        padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.05)">
              <div style="color:#64748b;font-size:11px;font-weight:600;
                          text-transform:uppercase;letter-spacing:.05em">{label}</div>
              <div style="color:{color};font-size:26px;font-weight:700;
                          line-height:1.15;margin:4px 0 2px">{value}</div>
              <div style="color:#64748b;font-size:12px">{sub}</div>
            </div>"""
        for label, value, sub, color in cards
    )
    return (f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
            f'gap:10px;margin-top:2px">{items}</div>')

def make_numbers_rows(source, summary):
    src_n = extract_numbers(source); pred_n = extract_numbers(summary)
    rows = []
    for n in sorted(src_n & pred_n)[:40]:
        rows.append([n, "✅ Giữ lại", "Có trong bài và tóm tắt"])
    for n in sorted(pred_n - src_n)[:30]:
        rows.append([n, "⚠️ Cần kiểm tra", "Có trong tóm tắt nhưng không có trong bài gốc"])
    for n in sorted(src_n - pred_n)[:30]:
        rows.append([n, "➖ Bị bỏ qua", "Có trong bài gốc nhưng không có trong tóm tắt"])
    if not rows:
        rows.append(["-", "—", "Không phát hiện số liệu trong bài / tóm tắt"])
    return rows

EMPTY_HTML = "<div style='border:1.5px dashed #cbd5e1;border-radius:10px;background:#fafafa;" \
             "padding:22px;color:#64748b;font-size:14px;text-align:center'>" \
             "Nhấn \"Tóm tắt ngay\" để xem kết quả.</div>"

def summarize(source, mode, beams, candidates, temperature):
    source = (source or "").strip()
    if not source:
        return "", EMPTY_HTML.replace("Nhấn", "Hãy dán nội dung bài báo hoặc chọn mẫu, rồi nhấn"), [], []
    if len(source.split()) < 40:
        return "", EMPTY_HTML.replace("Nhấn \"Tóm tắt ngay\" để xem kết quả.",
                                      "Nội dung hơi ngắn — hãy nhập bài đầy đủ hơn."), [], []
    start = time.time()
    cands, scores = generate_candidates(source, mode, beams, candidates, temperature)
    best = max(range(len(cands)), key=lambda i: scores[i])
    summary = cands[best]
    elapsed = time.time() - start
    cand_rows = [
        [i, round(float(s), 4), round(length_ratio(source, c), 4),
         round(number_hallucination(source, c) * 100, 2),
         c[:260] + ("…" if len(c) > 260 else "")]
        for i, (c, s) in enumerate(zip(cands, scores), 1)
    ]
    return summary, make_metric_html(source, summary, elapsed, mode), make_numbers_rows(source, summary), cand_rows

def load_example(choice):
    if not choice or choice not in EXAMPLE_MAP: return ""
    return EXAMPLE_MAP[choice]["source"]

def clear_all():
    return "", "", EMPTY_HTML.replace("Nhấn \"Tóm tắt ngay\" để xem kết quả.",
                                      "Sẵn sàng tóm tắt bài viết mới."), [], []

def model_badge():
    status = "GPU ✓" if torch.cuda.is_available() else "CPU"
    return (
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:10px 16px;display:flex;align-items:center;gap:14px;font-size:13px;'
        f'margin-bottom:10px;flex-wrap:wrap">'
        f'<b style="color:#0f172a">ViT5 Stage 2 — Best Model</b>'
        f'<span style="color:#94a3b8;font-family:monospace;font-size:11px;flex:1;'
        f'overflow-wrap:anywhere">{MODEL_DIR}</span>'
        f'<span style="color:#0f766e;font-weight:600;font-size:12px">Runtime: {status}</span>'
        f'</div>'
    )

# ── Styling ───────────────────────────────────────────────────────────────────
CSS = """
/* === GLOBAL LIGHT OVERRIDE === */
*, *::before, *::after { box-sizing: border-box; }

html, body { background: #f0f9ff !important; color: #0f172a !important; }

.gradio-container {
    max-width: 1380px !important;
    margin: 0 auto !important;
    background: linear-gradient(150deg, #f0f9ff 0%, #ecfeff 55%, #f0fdf4 100%) !important;
    color: #0f172a !important;
    font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif !important;
}

/* Block backgrounds */
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel {
    background: #ffffff !important;
    border-color: #e2e8f0 !important;
}

/* === ALL TEXT DARK — including readonly/disabled output boxes === */
.gradio-container textarea,
.gradio-container textarea:focus,
.gradio-container textarea:disabled,
.gradio-container textarea[disabled],
.gradio-container textarea[readonly],
.gradio-container textarea:read-only,
.gradio-container input[type="text"],
.gradio-container input[type="text"]:disabled,
.gradio-container input[type="number"] {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    opacity: 1 !important;
    background-color: #ffffff !important;
    border-color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: #94a3b8 !important; }

/* Labels */
.gradio-container label > span,
.gradio-container .label-wrap > span,
.gradio-container [data-testid="block-label"] span {
    color: #334155 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    background: transparent !important;
}
.gradio-container [data-testid="block-label"],
.gradio-container .label-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding-bottom: 6px !important;
}

/* === BUTTONS === */
.gradio-container button[variant="primary"],
.gradio-container .primary {
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600 !important;
}
.gradio-container button:not([variant="primary"]):not(.primary) {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
}

/* === DROPDOWN === */
.gradio-container select,
.gradio-container [class*="dropdown"],
.gradio-container [class*="select"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #e2e8f0 !important;
}
.gradio-container [role="option"],
.gradio-container [role="listbox"] {
    background: #ffffff !important;
    color: #0f172a !important;
}
.gradio-container [role="option"]:hover {
    background: #f0fdfa !important;
}

/* === RADIO === */
.gradio-container input[type="radio"] + span,
.gradio-container .gr-radio span { color: #0f172a !important; }

/* === SLIDER === */
.gradio-container input[type="range"] { accent-color: #0f766e; }

/* === TABLES === */
.gradio-container table, .gradio-container td, .gradio-container tr,
.gradio-container tbody, .gradio-container .table-wrap,
.gradio-container [class*="dataframe"] {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #e2e8f0 !important;
}
.gradio-container th,
.gradio-container thead tr,
.gradio-container [role="columnheader"] {
    background: #f0fdfa !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    border-color: #99f6e4 !important;
}

/* === ACCORDION === */
.gradio-container details > summary {
    background: #f8fafc !important;
    color: #334155 !important;
    font-weight: 600 !important;
    border-color: #e2e8f0 !important;
}
.gradio-container details {
    border-color: #e2e8f0 !important;
    background: #ffffff !important;
}

/* Misc */
footer, .footer, .wrap { background: transparent !important; }
"""

# JS: inject a high-priority stylesheet, then watch DOM mutations so the fix
# survives every Gradio re-render (Gradio replaces textarea content via Svelte reactivity).
FIX_JS = """
() => {
    const CSS = `
        .gradio-container textarea,
        .gradio-container textarea:focus,
        .gradio-container textarea:disabled,
        .gradio-container textarea[readonly],
        .gradio-container textarea:read-only,
        .gradio-container input[type="text"],
        .gradio-container input[type="text"]:disabled {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            opacity: 1 !important;
            background: #ffffff !important;
        }
    `;

    // Inject once into <head>
    if (!document.getElementById('gradio-text-fix')) {
        const tag = document.createElement('style');
        tag.id = 'gradio-text-fix';
        tag.textContent = CSS;
        document.head.appendChild(tag);
    }

    // Also directly patch every textarea — catches cases where inline styles win
    const patch = () => {
        document.querySelectorAll('.gradio-container textarea').forEach(el => {
            el.style.setProperty('color',                  '#0f172a', 'important');
            el.style.setProperty('-webkit-text-fill-color','#0f172a', 'important');
            el.style.setProperty('opacity',               '1',        'important');
        });
    };

    patch();

    // Re-patch whenever Gradio mutates the DOM (e.g. after receiving output)
    new MutationObserver(patch).observe(document.body, {
        childList: true, subtree: true, characterData: true
    });
}
"""

THEME = gr.themes.Default(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#f0f9ff",
    body_text_color="#0f172a",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_label_background_fill="transparent",
    block_label_border_color="transparent",
    block_label_text_color="#334155",
    input_background_fill="#ffffff",
    input_border_color="#e2e8f0",
    input_placeholder_color="#94a3b8",
    button_secondary_background_fill="#ffffff",
    button_secondary_text_color="#0f172a",
)

# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=THEME, css=CSS, js=FIX_JS,
               title="ViT5 Health News Summarizer") as demo:

    gr.HTML("""
    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;
                padding:28px 32px;margin-bottom:6px;
                box-shadow:0 1px 4px rgba(15,23,42,.06)">
      <h1 style="font-size:26px;font-weight:700;color:#0f172a;margin:0 0 8px;
                 letter-spacing:-.01em">
        ViT5 — Tóm tắt tin tức y tế tiếng Việt
      </h1>
      <p style="color:#64748b;font-size:14px;line-height:1.65;margin:0 0 16px;max-width:700px">
        Demo hệ thống tóm tắt tự động dùng model <b style="color:#0f172a">ViT5 fine-tuned (Stage 2)</b>.
        Tập trung vào hai yếu tố quan trọng: <b style="color:#0f172a">nội dung tóm tắt</b>
        và <b style="color:#0f172a">độ trung thực của số liệu</b>.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span style="background:#f0fdfa;border:1px solid #99f6e4;color:#0f766e;
                     padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600">
          Stage 2 · best_model</span>
        <span style="background:#f0fdfa;border:1px solid #99f6e4;color:#0f766e;
                     padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600">
          Number-aware Reranking</span>
        <span style="background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;
                     padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600">
          Local inference</span>
        <span style="background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;
                     padding:4px 12px;border-radius:999px;font-size:12px;font-weight:600">
          VnExpress Sức khoẻ</span>
      </div>
    </div>
    """)

    gr.HTML(model_badge())

    with gr.Row(equal_height=False):

        # ── Left: Input ───────────────────────────────────────────────────────
        with gr.Column(scale=5):
            example = gr.Dropdown(
                choices=EXAMPLE_CHOICES,
                label="Chọn bài mẫu",
                value=EXAMPLE_CHOICES[0] if EXAMPLE_CHOICES else None,
                interactive=True,
            )
            source = gr.Textbox(
                label="Nội dung bài báo",
                placeholder="Dán nội dung bài báo tiếng Việt vào đây, hoặc chọn bài mẫu ở trên…",
                lines=20,
                max_lines=32,
            )
            with gr.Row():
                load_btn  = gr.Button("Nạp bài mẫu",    size="sm")
                run_btn   = gr.Button("▶  Tóm tắt ngay", variant="primary")
                clear_btn = gr.Button("Xoá",             size="sm")

            with gr.Accordion("⚙️  Cài đặt nâng cao", open=False):
                mode = gr.Radio(
                    ["Chất lượng + Rerank", "Nhanh"],
                    value="Chất lượng + Rerank",
                    label="Chế độ sinh",
                )
                with gr.Row():
                    beams       = gr.Slider(1, 8,   value=4,   step=1,   label="Beam search")
                    candidates  = gr.Slider(2, 6,   value=3,   step=1,   label="Số ứng viên")
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Nhiệt độ lấy mẫu")

        # ── Right: Output ─────────────────────────────────────────────────────
        with gr.Column(scale=7):
            summary = gr.Textbox(
                label="Tóm tắt của model",
                lines=8,
                max_lines=14,
                show_copy_button=True,
                elem_id="summary-output",
            )
            metrics = gr.HTML(EMPTY_HTML)

            with gr.Accordion("🔢  Kiểm tra số liệu chi tiết", open=True):
                numbers = gr.Dataframe(
                    headers=["Số liệu", "Trạng thái", "Ghi chú"],
                    datatype=["str", "str", "str"],
                    label="",
                    wrap=True,
                    row_count=(5, "dynamic"),
                )

            with gr.Accordion("📋  Tất cả ứng viên (rerank)", open=False):
                candidate_table = gr.Dataframe(
                    headers=["#", "Score", "Tỉ lệ dài", "Ảo giác %", "Nội dung"],
                    datatype=["number", "number", "number", "number", "str"],
                    label="",
                    wrap=True,
                )

    load_btn.click(load_example, inputs=example, outputs=source)
    example.change(load_example, inputs=example, outputs=source)
    run_btn.click(
        summarize,
        inputs=[source, mode, beams, candidates, temperature],
        outputs=[summary, metrics, numbers, candidate_table],
    )
    clear_btn.click(clear_all, outputs=[source, summary, metrics, numbers, candidate_table])


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
