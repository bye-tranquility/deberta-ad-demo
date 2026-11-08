"""Сравнение вариантов декодера на кросс-валидации NaiAD_human (по уже посчитанным вероятностям, без GPU).

Варианты: старый (порог + min_len), + склейка близких кусков (merge_gap), + выравнивание границ по предложениям
(snap), оба. Все настройки варианта подбираются на val-фолде по span_f1@0.5, метрики — на test-фолде;
итог — по всем 588 текстам. Контроль: 2000 синтетических текстов test и они же без рекламы.

  uv run python scripts/decoder_cv.py
Результат: runs/decoder_cv.json и таблица в консоли.
"""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock.data import PROCESSED, RUNS, normalize_row, read_jsonl, strip_ads  # noqa: E402
from adblock.decoder import merge_close, raw_spans, sentence_boundaries, snap_spans, to_binary  # noqa: E402
from adblock.human import N_FOLDS, load_cv  # noqa: E402
from adblock.metrics import (MIN_LENS, THRESHOLDS, boundary_errors, char_metrics, evaluate, f1, iou,  # noqa: E402
                             to_spans)
from adblock.tokenize import MODEL, label_tokens  # noqa: E402

N_SYNTH = 2000
MERGE_GAPS = [0, 3, 10, 25, 50]
SNAPS = [0, 5, 15, 40]
VARIANTS = {
    "старый декодер": ([0], [0]),
    "+ склейка": (MERGE_GAPS, [0]),
    "+ выравнивание по предложениям": ([0], SNAPS),
    "+ склейка и выравнивание": (MERGE_GAPS, SNAPS),
}
# (подпись, на какой модели подбирать настройки декодера по val, какую модель оценивать на test, нормализация)
SETUPS = [
    ("DeBERTa — без дообучения", lambda k: "deberta", lambda k: "deberta", False),
    ("DeBERTa — дообучена на human", lambda k: f"deberta_human_fold{k}", lambda k: f"deberta_human_fold{k}", False),
    ("DeBERTa — v2: нормализация, 2 повтора, эпоха по span_f1",
     lambda k: f"deberta_human3_fold{k}", lambda k: f"deberta_human3_fold{k}", True),
    # переобученная модель видела val, поэтому настройки декодера берутся с модели первого этапа
    ("DeBERTa — v2 + переобучение на train+val",
     lambda k: f"deberta_human3_fold{k}", lambda k: f"deberta_human3_refit_fold{k}", True),
    ("TF-IDF, предложения — обучен на смеси с human",
     lambda k: f"baseline_tfidf_sent_human_fold{k}", lambda k: f"baseline_tfidf_sent_human_fold{k}", False),
    ("TF-IDF, окно — обучен на смеси с human",
     lambda k: f"baseline_tfidf_window_human_fold{k}", lambda k: f"baseline_tfidf_window_human_fold{k}", False),
]


def prepare(tok, rows):
    out = []
    for r in rows:
        _, offsets, labels = label_tokens(tok, r["text"], r["ad_spans"])
        out.append({"row": r, "offsets": offsets, "labels": labels, "gt": to_spans(labels),
                    "bounds": sentence_boundaries(r["text"], offsets)})
    return out


def decode_all(docs, probs, thr, min_len, gap, snap):
    res = []
    for d, p in zip(docs, probs):
        s = merge_close(raw_spans(p, thr), gap)
        s = [x for x in s if x[1] - x[0] >= min_len]
        res.append(snap_spans(s, d["bounds"], snap) if snap else s)
    return res


def span_f1(docs, spans):
    tp = sum(any(iou(a, g) >= 0.5 for a in s) for d, s in zip(docs, spans) for g in d["gt"])
    return f1(tp, sum(map(len, spans)), sum(len(d["gt"]) for d in docs))


def tune(docs, probs, gaps, snaps):
    """Лучшие (порог, min_len, merge_gap, snap) на val по span_f1@0.5."""
    best = (-1, None)
    for thr in THRESHOLDS:
        raw = [raw_spans(p, thr) for p in probs]
        for gap in gaps:
            merged = [merge_close(s, gap) for s in raw]
            for ml in MIN_LENS:
                kept = [[x for x in s if x[1] - x[0] >= ml] for s in merged]
                for snap in snaps:
                    spans = [snap_spans(s, d["bounds"], snap) for s, d in zip(kept, docs)] if snap else kept
                    score = span_f1(docs, spans)
                    if score > best[0]:
                        best = (score, (thr, ml, gap, snap))
    return best[1]


def datasets(tok, normalize):
    """Тексты для оценки: human-фолды и контроль на синтетике (с той же предобработкой, что у модели)."""
    prep = (lambda rows: prepare(tok, [normalize_row(r) for r in rows])) if normalize else (lambda rows: prepare(tok, rows))
    synth_rows = read_jsonl(PROCESSED / "test.jsonl")[:N_SYNTH]
    return {"synth": prep(synth_rows),
            "stripped": prep([{"text": strip_ads(r["text"], r["ad_spans"]), "ad_spans": []} for r in synth_rows]),
            "folds": {k: {"val": prep(load_cv(k, "val")), "test": prep(load_cv(k, "test"))} for k in range(N_FOLDS)}}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    data = {}

    results = {}
    for setup, tune_of, run_of, normalize in SETUPS:
        needed = [RUNS / tune_of(k) / f"cv_probs_fold{k}_val.json" for k in range(N_FOLDS)]
        needed += [RUNS / run_of(k) / f"cv_probs_fold{k}_test.json" for k in range(N_FOLDS)]
        if not all(p.exists() for p in needed):
            print(f"пропускаю {setup}: нет вероятностей")
            continue
        if normalize not in data:
            data[normalize] = datasets(tok, normalize)
        synth, stripped, folds = data[normalize]["synth"], data[normalize]["stripped"], data[normalize]["folds"]
        for variant, (gaps, snaps) in VARIANTS.items():
            if variant not in ("старый декодер", "+ склейка и выравнивание"):
                continue
            test_docs, test_bin, synth_f1, false_blocks, chosen = [], [], [], [], []
            for k in range(N_FOLDS):
                load = lambda part, of=run_of: json.loads((RUNS / of(k) / f"cv_probs_fold{k}_{part}.json").read_text())  # noqa: E731
                params = tune(folds[k]["val"], load("val", tune_of), gaps, snaps)
                chosen.append(params)
                spans = decode_all(folds[k]["test"], load("test"), *params)
                test_docs += folds[k]["test"]
                test_bin += [to_binary(s, len(d["labels"])) for s, d in zip(spans, folds[k]["test"])]
                synth_f1.append(span_f1(synth, decode_all(synth, load("synth"), *params)))
                false_blocks.append(sum(bool(s) for s in decode_all(stripped, load("stripped"), *params)) / len(stripped))
            labels = [d["labels"] for d in test_docs]
            ev = evaluate(test_bin, labels, 0.5, 1)
            be = boundary_errors(test_bin, labels, 0.5, 1)
            cm = char_metrics([d["row"]["text"] for d in test_docs], [d["row"]["ad_spans"] for d in test_docs],
                              [d["offsets"] for d in test_docs], test_bin, 0.5, 1)
            results[f"{setup} | {variant}"] = {
                **ev, **cm, "start_exact": be["start"].get("exact", 0), "end_exact": be["end"].get("exact", 0),
                "start_mean_abs": be["start"].get("mean_abs", 0), "end_mean_abs": be["end"].get("mean_abs", 0),
                "synth_span_f1@0.5": sum(synth_f1) / N_FOLDS, "false_block_without_ads": sum(false_blocks) / N_FOLDS,
                "params_by_fold (порог, min_len, merge_gap, snap)": chosen}
            r = results[f"{setup} | {variant}"]
            print(f"{setup:48s} {variant:32s} @0.5={r['span_f1@0.5']:.3f} @0.7={r['span_f1@0.7']:.3f} "
                  f"утечка={r['ad_leakage']:.1%} потери={r['text_loss']:.1%} синт={r['synth_span_f1@0.5']:.3f} "
                  f"ложн={r['false_block_without_ads']:.1%} | {chosen}", flush=True)
    (RUNS / "decoder_cv.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
