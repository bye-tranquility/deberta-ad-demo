"""Демо: поиск рекламы в ответе LLM.

  uv run streamlit run app.py

Нужны обученные модели: runs/deberta/model (скачать с Modal, см. README) и
runs/baseline_*/model.pkl (scripts/baselines.py). Недоступные модели в списке не показываются.
"""
import html
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from adblock import baseline_models as bm  # noqa: E402
from adblock.data import PROCESSED, RUNS, parse, read_jsonl  # noqa: E402
from adblock.metrics import boundary_errors, char_metrics, iou, predict_spans, to_spans  # noqa: E402
from adblock.tokenize import MODEL, label_tokens  # noqa: E402

MODELS = {
    "DeBERTa": "deberta",
    "TF-IDF: абзацы": "baseline_tfidf_para",
    "TF-IDF: скользящее окно": "baseline_tfidf_window",
    "TF-IDF: предложения": "baseline_tfidf_sent",
    "Позиция в тексте": "baseline_position",
}
HUMAN = ROOT / "data" / "naiad" / "NaiAD_human.jsonl"

# цвета подсветки
BOTH, EXTRA, MISSED, PRED = "#bbf7d0", "#fecaca", "#bfdbfe", "#fde68a"


# ---------- загрузка (кэшируется между перерисовками) ----------

def available_models():
    out = {}
    for label, name in MODELS.items():
        path = RUNS / name / ("model" if name == "deberta" else "model.pkl")
        if path.exists():
            out[label] = name
    return out


@st.cache_resource
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@st.cache_resource
def load_model(name):
    if name == "deberta":
        from adblock.predict import load_deberta
        return load_deberta(RUNS / name / "model")
    return bm.load(RUNS / name / "model.pkl")


@st.cache_data
def defaults(name):
    """Порог и min_len, подобранные на val (из metrics.json)."""
    path = RUNS / name / "metrics.json"
    if path.exists():
        m = json.loads(path.read_text())
        return float(m["threshold"]), int(m["min_len"])
    return 0.5, 1


@st.cache_data
def examples(source):
    if source == "test":
        return [{"text": r["text"], "ad_spans": r["ad_spans"]} for r in read_jsonl(PROCESSED / "test.jsonl")]
    rows = []
    for r in read_jsonl(HUMAN):
        parsed = parse(r["response"])
        if parsed is not None:
            rows.append({"text": parsed[0], "ad_spans": parsed[1]})
    return rows


@st.cache_data(max_entries=200, show_spinner="Модель читает текст…")
def run_model(name, text):
    """Вероятности по токенам. Кэш: при движении ползунков модель не перезапускается."""
    tok = tokenizer()
    _, offsets, _ = label_tokens(tok, text, [])
    if name == "deberta":
        from adblock.predict import predict
        model, deberta_tok = load_model(name)
        return predict(model, deberta_tok, text)[0]
    return load_model(name).predict(text, offsets)


# ---------- отрисовка ----------

def char_mask(n, spans):
    m = [False] * n
    for a, b in spans:
        m[a:b] = [True] * (b - a)
    return m


def render_highlight(text, pred_chars, gt_chars):
    """Подсветка: если есть разметка — совпадение / лишнее / пропущено, иначе просто предсказание."""
    pred, gt = char_mask(len(text), pred_chars), char_mask(len(text), gt_chars or [])

    def style(i):
        if gt_chars is None:
            return f"background:{PRED}" if pred[i] else ""
        if pred[i] and gt[i]:
            return f"background:{BOTH}"
        if pred[i]:
            return f"background:{EXTRA}"
        if gt[i]:
            return f"background:{MISSED}"
        return ""

    return runs_to_html(text, [style(i) for i in range(len(text))])


def render_heatmap(text, offsets, probs, gt_chars):
    """Фон каждого токена — вероятность рекламы; настоящая реклама подчёркнута."""
    styles = [""] * len(text)
    for (s, e), p in zip(offsets, probs):
        for i in range(s, e):
            styles[i] = f"background:rgba(220,38,38,{p:.2f})"
    for i, g in enumerate(char_mask(len(text), gt_chars or [])):
        if g:
            styles[i] += ";border-bottom:2px solid #2563eb"
    return runs_to_html(text, styles)


def runs_to_html(text, styles):
    parts, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and styles[j] == styles[i]:
            j += 1
        chunk = html.escape(text[i:j])
        parts.append(f'<span style="{styles[i]};color:#111">{chunk}</span>' if styles[i] else chunk)
        i = j
    return ('<div style="white-space:pre-wrap;line-height:1.6;font-size:15px;'
            'background:#fff;color:#111;border:1px solid #e5e7eb;border-radius:8px;padding:16px;'
            'max-height:600px;overflow-y:auto">'
            + "".join(parts) + "</div>")


def remove_ads(text, pred_chars):
    out, pos = [], 0
    for a, b in pred_chars:
        out.append(text[pos:a])
        pos = b
    out.append(text[pos:])
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


def legend(items):
    return " ".join(f'<span style="background:{c};color:#111;padding:2px 8px;border-radius:4px;margin-right:6px">{t}</span>'
                    for c, t in items)


# ---------- страница ----------

st.set_page_config(page_title="AdBlock: реклама в ответах LLM", layout="wide")
st.title("Поиск рекламы в ответе LLM")

models = available_models()
if not models:
    st.error("Не найдено ни одной обученной модели в runs/. См. README.")
    st.stop()

with st.sidebar:
    st.header("Настройки")
    label = st.selectbox("Модель", list(models))
    name = models[label]
    thr_default, min_len_default = defaults(name)

    # при смене модели ползунки встают на её значения, подобранные на val
    reset = st.button("Сбросить на значения с val")
    if reset or st.session_state.get("sliders_for") != name:
        st.session_state.update(sliders_for=name, thr=thr_default, min_len=min_len_default)

    thr = st.slider("Порог вероятности", 0.05, 0.95, step=0.05, key="thr",
                    help="Токен считается рекламой, если вероятность выше порога.")
    min_len = st.slider("min_len, токенов", 1, 60, key="min_len",
                        help="Найденные куски короче этого отбрасываются как шум.")
    st.caption(f"Значения с val для этой модели: порог {thr_default:.2f}, min_len {min_len_default}")
    st.divider()
    show_gt = st.checkbox("Показывать настоящую разметку", value=True)
    heatmap = st.checkbox("Тепловая карта вероятностей", value=False)

source = st.radio("Текст", ["Пример из test", "Пример из NaiAD_human", "Свой текст"], horizontal=True)
if source == "Свой текст":
    text = st.text_area("Вставьте ответ LLM", height=200, placeholder="Вставьте сюда текст…")
    gt_spans = None
else:
    key = "test" if source == "Пример из test" else "human"
    rows = examples(key)
    if st.button("🎲 Случайный пример") or st.session_state.get("example_src") != key:
        st.session_state.update(example_src=key, example_idx=random.randrange(len(rows)))
    row = rows[st.session_state.example_idx]
    st.caption(f"Пример №{st.session_state.example_idx} из {len(rows)}")
    text, gt_spans = row["text"], [tuple(s) for s in row["ad_spans"]]

if not text.strip():
    st.info("Вставьте текст, чтобы найти в нём рекламу.")
    st.stop()

probs = run_model(name, text)
_, offsets, labels = label_tokens(tokenizer(), text, gt_spans or [])
pred = predict_spans(probs, thr, min_len)
pred_chars = [(offsets[a][0], offsets[b - 1][1]) for a, b in pred]
gt_shown = gt_spans if (show_gt and gt_spans is not None) else None

# --- сводка ---
cols = st.columns(5)
cols[0].metric("Найдено блоков", len(pred))
n_chars = sum(not c.isspace() for c in text)
cut = sum(not text[i].isspace() for a, b in pred_chars for i in range(a, b))
cols[1].metric("Помечено как реклама", f"{cut / max(1, n_chars):.0%} текста")
if gt_spans is not None and gt_spans:
    gt_tok = to_spans(labels)
    best_iou = max((iou(p, g) for p in pred for g in gt_tok), default=0.0)
    be = boundary_errors([probs], [labels], thr, min_len)
    cm = char_metrics([text], [gt_spans], [offsets], [probs], thr, min_len)
    cols[2].metric("IoU с разметкой", f"{best_iou:.2f}")
    if be["start"]:
        cols[3].metric("Ошибка границ, токенов", f"{be['start']['mean']:+.0f} / {be['end']['mean']:+.0f}",
                       help="Начало / конец: минус — раньше разметки, плюс — позже.")
    else:
        cols[3].metric("Ошибка границ, токенов", "блок не найден")
    cols[4].metric("Утечка / потери", f"{cm['ad_leakage']:.0%} / {cm['text_loss']:.0%}",
                   help="Утечка — доля рекламы, оставшейся в тексте. Потери — доля нормального текста, вырезанного по ошибке.")
elif gt_spans is not None:
    cols[2].metric("Разметка", "рекламы нет")

# --- вкладки ---
tab_hl, tab_clean, tab_plot = st.tabs(["Подсветка", "Текст без рекламы", "Вероятности по токенам"])
with tab_hl:
    if heatmap:
        st.html(legend([("#fbe3e3", "низкая вероятность"), ("#e14b4b", "высокая")])
                + (" <span style='border-bottom:2px solid #2563eb'>настоящая реклама</span>" if gt_shown else ""))
        st.html(render_heatmap(text, offsets, probs, gt_shown))
    else:
        if gt_shown is not None:
            st.html(legend([(BOTH, "найдено верно"), (EXTRA, "лишнее"), (MISSED, "пропущено")]))
        else:
            st.html(legend([(PRED, "найденная реклама")]))
        st.html(render_highlight(text, pred_chars, gt_shown))
with tab_clean:
    st.html(runs_to_html(remove_ads(text, pred_chars), [""] * len(remove_ads(text, pred_chars))))
with tab_plot:
    df = pd.DataFrame({"вероятность рекламы": probs, "порог": [thr] * len(probs)})
    colors = ["#dc2626", "#9ca3af"]                     # вероятность — красная, порог — серый
    if gt_shown is not None:
        df["настоящая реклама"] = labels
        colors.append("#2563eb")                        # разметка — синяя
    st.line_chart(df, x_label="номер токена", color=colors, height=320)
