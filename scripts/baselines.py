"""Baselines. Каждый выдаёт вероятность рекламы на токенной сетке DeBERTa для val и test
и кладёт её в runs/<имя>/probs_{val,test}.json, а саму модель — в runs/<имя>/model.pkl (для демо).
Дальше общий декодер:

  uv run python scripts/baselines.py                    # все
  uv run python scripts/baselines.py --only window
  uv run python scripts/evaluate.py --name baseline_tfidf_window   # порог и min_len по val, метрики на test

1. baseline_position      — доля рекламы по 50 корзинам относительной позиции (на train)
2. baseline_tfidf_para    — TF-IDF + LogReg по абзацам
3. baseline_tfidf_sent    — TF-IDF + LogReg по предложениям: [пред. | текущее | след. | флаги] + сглаживание
4. baseline_tfidf_window  — TF-IDF + LogReg по скользящим окнам слов: [левый контекст | центр | правый контекст]

Гиперпараметры (C, окно сглаживания) у всех TF-IDF baselines выбираются одинаково:
лучший span_f1@0.5 на случайных 2000 текстах val после той же сетки порогов и min_len, что у общего декодера.
Признаки и предсказание описаны в src/adblock/baseline_models.py.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock import baseline_models as bm  # noqa: E402
from adblock.data import PROCESSED, RUNS, read_jsonl  # noqa: E402
from adblock.metrics import best_span_f1  # noqa: E402
from adblock.tokenize import MODEL, label_tokens  # noqa: E402
from adblock.units import to_tokens, unit_labels  # noqa: E402

N_BINS = 50
CS = (0.5, 2, 8)
SELECT_DOCS = 2000          # размер подвыборки val для выбора гиперпараметров
NAMES = {"para": "baseline_tfidf_para", "sent": "baseline_tfidf_sent", "window": "baseline_tfidf_window"}
SMOOTH = {"para": (1,), "sent": (1, 3), "window": (1,)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tokenize(tok, rows):
    """offsets и метки токенов (та же сетка, что у DeBERTa)."""
    out = [label_tokens(tok, r["text"], r["ad_spans"]) for r in rows]
    return [o for _, o, _ in out], [y for _, _, y in out]


def save(name, model, val_probs, val_labels, test_probs, test_labels, info):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    for split, p, y in (("val", val_probs, val_labels), ("test", test_probs, test_labels)):
        (d / f"probs_{split}.json").write_text(json.dumps({"probs": [[round(x, 4) for x in q] for q in p], "labels": y}))
    (d / "baseline_info.json").write_text(json.dumps(info, indent=2))
    bm.save(model, d / "model.pkl")
    log(f"{name}: сохранено, " + str({k: v for k, v in info.items() if k != "table"}))


def token_probs(p, units, offsets, docs=None):
    """Вероятности единиц -> вероятности токенов документов (docs — индексы нужных документов)."""
    starts = np.cumsum([0] + [len(u) for u in units])
    docs = range(len(units)) if docs is None else docs
    return [to_tokens(offsets[i], units[i], p[starts[i]:starts[i + 1]]) for i in docs]


# ---------- 1. position prior ----------

def run_position(tok, data, val_t, test_t):
    log("position: токенизирую train")
    tr_off, tr_lab = tokenize(tok, data["train"])
    ad, total = np.zeros(N_BINS), np.zeros(N_BINS)
    for r, offs, y in zip(data["train"], tr_off, tr_lab):
        for (s, _), lab in zip(offs, y):
            b = min(N_BINS - 1, int(s / len(r["text"]) * N_BINS))
            total[b] += 1
            ad[b] += lab
    model = bm.PositionPrior((ad / np.maximum(total, 1)).tolist())

    def probs(rows, offsets):
        return [model.predict(r["text"], o) for r, o in zip(rows, offsets)]

    save("baseline_position", model, probs(data["val"], val_t[0]), val_t[1],
         probs(data["test"], test_t[0]), test_t[1], {"bins": N_BINS, "table": model.table})


# ---------- 2–4. TF-IDF baselines ----------

def run_tfidf(kind, data, val_t, test_t):
    """Одинаковый для всех TF-IDF baselines выбор C (и окна сглаживания) по span_f1@0.5 на val."""
    name = NAMES[kind]
    model = bm.TfidfBaseline(kind)
    texts = {s: [r["text"] for r in data[s]] for s in data}
    units = {s: [model.units(t) for t in texts[s]] for s in data}
    y = [l for r, u in zip(data["train"], units["train"]) for l in unit_labels(r["text"], r["ad_spans"], u)]
    log(f"{name}: {len(y)} единиц в train, доля рекламных {np.mean(y):.3f}")
    model.fit_vectorizer(texts["train"], units["train"])
    X = {s: model.matrix(texts[s], units[s]) for s in data}
    log(f"{name}: признаков {X['train'].shape[1]}")

    sel = sorted(random.Random(42).sample(range(len(units["val"])), SELECT_DOCS))
    sel_labels = [val_t[1][i] for i in sel]
    best = None
    for C in CS:
        model.clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                       max_iter=1000, random_state=42).fit(X["train"], y)
        for w in SMOOTH[kind]:
            model.smooth_window = w
            probs = token_probs(model.unit_probs(X["val"], units["val"]), units["val"], val_t[0], sel)
            score, thr, min_len = best_span_f1(probs, sel_labels)
            log(f"{name}: C={C} окно={w} val span_f1@0.5={score:.4f} (порог {thr}, min_len {min_len})")
            if best is None or score > best[0]:
                best = (score, C, w, model.clf)
    score, C, model.smooth_window, model.clf = best
    info = {"C": C, "smooth_window": model.smooth_window, "val_span_f1@0.5_subset": score}
    if kind == "window":
        info.update(center_words=model.center, context_words=model.context)
    save(name, model,
         token_probs(model.unit_probs(X["val"], units["val"]), units["val"], val_t[0]), val_t[1],
         token_probs(model.unit_probs(X["test"], units["test"]), units["test"], test_t[0]), test_t[1], info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["position", "para", "sent", "window"])
    args = ap.parse_args()

    data = {s: read_jsonl(PROCESSED / f"{s}.jsonl") for s in ("train", "val", "test")}
    tok = AutoTokenizer.from_pretrained(MODEL)
    log("токенизирую val/test")
    val_t, test_t = tokenize(tok, data["val"]), tokenize(tok, data["test"])

    if args.only in (None, "position"):
        run_position(tok, data, val_t, test_t)
    for kind in ("para", "sent", "window"):
        if args.only in (None, kind):
            run_tfidf(kind, data, val_t, test_t)


if __name__ == "__main__":
    main()
