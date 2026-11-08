"""Baselines с обучением на NaiAD_human по фолдам — та же смесь данных, что у DeBERTa:
human train-фолды (повторены 4 раза) + 4000 случайных синтетических текстов train.
C (и окно сглаживания) выбираются по span_f1@0.5 на val-фолде human; модели — runs/<baseline>_human_fold<k>/model.pkl.

  uv run python scripts/baselines_cv.py            # все baselines, все фолды, затем оценка фолдов
"""
import random
import subprocess
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
from adblock.human import N_FOLDS, load_cv  # noqa: E402
from adblock.metrics import best_span_f1  # noqa: E402
from adblock.tokenize import MODEL, label_tokens  # noqa: E402
from adblock.units import to_tokens, unit_labels  # noqa: E402

SYNTH_N, HUMAN_REPEAT, CS, N_BINS = 4000, 4, (0.5, 2, 8), 50
SMOOTH = {"para": (1,), "sent": (1, 3), "window": (1,)}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def train_rows(fold):
    synth = random.Random(42).sample(read_jsonl(PROCESSED / "train.jsonl"), SYNTH_N)
    return synth + load_cv(fold, "train") * HUMAN_REPEAT


def token_probs(p, units, offsets):
    starts = np.cumsum([0] + [len(u) for u in units])
    return [to_tokens(offsets[i], units[i], p[starts[i]:starts[i + 1]]) for i in range(len(units))]


def fit_position(tok, rows):
    ad, total = np.zeros(N_BINS), np.zeros(N_BINS)
    for r in rows:
        _, offs, y = label_tokens(tok, r["text"], r["ad_spans"])
        for (s, _), lab in zip(offs, y):
            b = min(N_BINS - 1, int(s / len(r["text"]) * N_BINS))
            total[b] += 1
            ad[b] += lab
    return bm.PositionPrior((ad / np.maximum(total, 1)).tolist())


def fit_tfidf(kind, rows, val, val_offsets, val_labels):
    model = bm.TfidfBaseline(kind)
    texts = [r["text"] for r in rows]
    units = [model.units(t) for t in texts]
    y = [l for r, u in zip(rows, units) for l in unit_labels(r["text"], r["ad_spans"], u)]
    model.fit_vectorizer(texts, units)
    X = model.matrix(texts, units)
    v_units = [model.units(r["text"]) for r in val]
    X_val = model.matrix([r["text"] for r in val], v_units)
    best = None
    for C in CS:
        model.clf = LogisticRegression(C=C, class_weight="balanced", solver="liblinear",
                                       max_iter=1000, random_state=42).fit(X, y)
        for w in SMOOTH[kind]:
            model.smooth_window = w
            score = best_span_f1(token_probs(model.unit_probs(X_val, v_units), v_units, val_offsets), val_labels)[0]
            if best is None or score > best[0]:
                best = (score, model.clf, w)
    _, model.clf, model.smooth_window = best
    return model


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    for fold in range(N_FOLDS):
        rows = train_rows(fold)
        val = load_cv(fold, "val")
        val_tok = [label_tokens(tok, r["text"], r["ad_spans"]) for r in val]
        val_offsets, val_labels = [o for _, o, _ in val_tok], [y for _, _, y in val_tok]
        for kind in ("position", "para", "sent", "window"):
            name = "baseline_position_human" if kind == "position" else f"baseline_tfidf_{kind}_human"
            run = f"{name}_fold{fold}"
            d = RUNS / run
            if not (d / "model.pkl").exists():
                model = fit_position(tok, rows) if kind == "position" else fit_tfidf(kind, rows, val, val_offsets, val_labels)
                d.mkdir(parents=True, exist_ok=True)
                bm.save(model, d / "model.pkl")
                log(f"{run}: обучено" + ("" if kind == "position" else f", C={model.clf.C}, окно={model.smooth_window}"))
            subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_cv.py"), "--run", run, "--fold", str(fold)],
                           check=True)


if __name__ == "__main__":
    main()
