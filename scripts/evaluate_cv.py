"""Кросс-валидация на NaiAD_human: оценка одного фолда и сводка по всем фолдам.

Оценка фолда k для модели в runs/<run>/ (DeBERTa в model/ или baseline в model.pkl):
  uv run python scripts/evaluate_cv.py --run deberta_human_fold0 --fold 0
  uv run python scripts/evaluate_cv.py --run deberta --fold 0          # старая модель без дообучения
- порог и min_len подбираются на val-фолде human ((k + 1) % 5) по span_f1@0.5;
- метрики считаются на test-фолде k, плюс контроль на синтетике: первые 2000 текстов test и test_stripped.
Результат: runs/<run>/cv_fold<k>.json, вероятности — runs/<run>/cv_probs_fold<k>_<часть>.json.

Сводка (метрики по всем 588 текстам human, каждый предсказан моделью своего фолда со своим порогом):
  uv run python scripts/evaluate_cv.py --summary
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock import baseline_models as bm  # noqa: E402
from adblock.data import PROCESSED, RUNS, normalize_row, read_jsonl, strip_ads  # noqa: E402
from adblock.human import N_FOLDS, load_cv  # noqa: E402
from adblock.metrics import (auc_metrics, best_span_f1, boundary_errors, char_metrics, detection,  # noqa: E402
                             evaluate)
from adblock.tokenize import MODEL, label_tokens  # noqa: E402

N_SYNTH = 2000
# модели в сводке: (подпись, имя без дообучения, префикс дообученных по фолдам)
SETUPS = [
    ("DeBERTa", "deberta", "deberta_human"),
    ("TF-IDF, абзацы", "baseline_tfidf_para", "baseline_tfidf_para_human"),
    ("TF-IDF, скользящее окно", "baseline_tfidf_window", "baseline_tfidf_window_human"),
    ("TF-IDF, предложения", "baseline_tfidf_sent", "baseline_tfidf_sent_human"),
    ("Позиция в тексте", "baseline_position", "baseline_position_human"),
]


def tokenized(tok, rows):
    out = [label_tokens(tok, r["text"], r["ad_spans"]) for r in rows]
    return [o for _, o, _ in out], [y for _, _, y in out]


def predictor(run_dir):
    if (run_dir / "model").exists():
        from adblock.predict import load_deberta, predict
        model, dtok = load_deberta(run_dir / "model")
        return lambda text, offsets: predict(model, dtok, text)[0]
    model = bm.load(run_dir / "model.pkl")
    return model.predict


def probs_for(run_dir, fold, part, rows, offsets, get_predict):
    cache = run_dir / f"cv_probs_fold{fold}_{part}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    probs = [[round(float(x), 4) for x in get_predict()(r["text"], o)] for r, o in zip(rows, offsets)]
    cache.write_text(json.dumps(probs))
    return probs


def span_report(rows, offsets, labels, probs, thr, min_len):
    return {**evaluate(probs, labels, thr, min_len), **auc_metrics(probs, labels),
            "boundaries_tokens": boundary_errors(probs, labels, thr, min_len),
            **char_metrics([r["text"] for r in rows], [r["ad_spans"] for r in rows], offsets, probs, thr, min_len)}


def evaluate_fold(run, fold, normalize=False):
    run_dir = RUNS / run
    tok = AutoTokenizer.from_pretrained(MODEL)
    loaded = {}

    def get_predict():
        if "f" not in loaded:
            loaded["f"] = predictor(run_dir)
        return loaded["f"]

    synth = read_jsonl(PROCESSED / "test.jsonl")[:N_SYNTH]
    parts = {"val": load_cv(fold, "val"), "test": load_cv(fold, "test"), "synth": synth,
             "stripped": [{"text": strip_ads(r["text"], r["ad_spans"]), "ad_spans": []} for r in synth]}
    if normalize:   # та же предобработка, что при обучении модели
        parts = {k: [normalize_row(r) for r in rows] for k, rows in parts.items()}
    data = {}
    for part, rows in parts.items():
        offsets, labels = tokenized(tok, rows)
        data[part] = (rows, offsets, labels, probs_for(run_dir, fold, part, rows, offsets, get_predict))

    _, _, val_labels, val_probs = data["val"]
    val_f1, thr, min_len = best_span_f1(val_probs, val_labels)
    out = {"run": run, "fold": fold, "threshold": thr, "min_len": min_len, "val_span_f1@0.5": val_f1,
           "human_test": span_report(*data["test"], thr, min_len),
           "synth_test": evaluate(data["synth"][3], data["synth"][2], thr, min_len),
           "synth_vs_stripped": detection(data["synth"][3], data["stripped"][3], thr, min_len)}
    (run_dir / f"cv_fold{fold}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    h = out["human_test"]
    print(f"{run} фолд {fold}: порог {thr}, min_len {min_len} | human test span_f1@0.5={h['span_f1@0.5']:.3f} "
          f"| синтетика span_f1@0.5={out['synth_test']['span_f1@0.5']:.3f} "
          f"| ложный блок без рекламы {out['synth_vs_stripped']['found_neg']:.1%}")


def summary():
    tok = AutoTokenizer.from_pretrained(MODEL)
    table = {}
    for label, frozen, prefix in SETUPS:
        for kind, run_of in (("без дообучения", lambda k: frozen), ("дообучена на human", lambda k: f"{prefix}_fold{k}")):
            folds = [RUNS / run_of(k) / f"cv_fold{k}.json" for k in range(N_FOLDS)]
            if not all(f.exists() for f in folds):
                continue
            rows, offsets, labels, probs, thrs = [], [], [], [], []
            synth, stripped_fp = [], []
            for k, f in enumerate(folds):
                m = json.loads(f.read_text())
                part = load_cv(k, "test")
                o, y = tokenized(tok, part)
                p = json.loads((RUNS / run_of(k) / f"cv_probs_fold{k}_test.json").read_text())
                rows += part
                offsets += o
                labels += y
                probs += p
                thrs.append((m["threshold"], m["min_len"]))
                synth.append(m["synth_test"]["span_f1@0.5"])
                stripped_fp.append(m["synth_vs_stripped"]["found_neg"])
            # у каждого фолда свой порог: считаем метрики по фолдам и объединяем счётчики через пул документов
            pooled = pooled_report(rows, offsets, labels, probs, thrs)
            pooled["synth_span_f1@0.5_mean"] = sum(synth) / len(synth)
            pooled["stripped_false_block_mean"] = sum(stripped_fp) / len(stripped_fp)
            pooled["thresholds"] = thrs
            table[f"{label} — {kind}"] = pooled
    (RUNS / "cv_summary.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    print("| Модель | span_f1@0.3 | @0.5 | @0.7 | token_f1 | PR-AUC | утечка рекламы | потери текста | синтетика span_f1@0.5 | ложный блок без рекламы |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, m in table.items():
        print(f"| {name} | {m['span_f1@0.3']:.3f} | {m['span_f1@0.5']:.3f} | {m['span_f1@0.7']:.3f} | {m['token_f1']:.3f} "
              f"| {m['pr_auc']:.3f} | {m['ad_leakage']:.1%} | {m['text_loss']:.1%} "
              f"| {m['synth_span_f1@0.5_mean']:.3f} | {m['stripped_false_block_mean']:.1%} |")


def pooled_report(rows, offsets, labels, probs, thrs):
    """Метрики по всем текстам сразу, каждый документ — со своим (порогом, min_len) фолда.

    Трюк: переводим вероятности в 0/1 по порогу своего фолда и обнуляем спаны короче min_len фолда,
    после чего считаем общие метрики с порогом 0.5 и min_len 1. PR-AUC считается по сырым вероятностям.
    """
    from adblock.metrics import predict_spans

    sizes = [len(load_cv(k, "test")) for k in range(N_FOLDS)]
    binary, i = [], 0
    for (thr, min_len), n in zip(thrs, sizes):
        for p in probs[i:i + n]:
            b = [0.0] * len(p)
            for a, c in predict_spans(p, thr, min_len):
                b[a:c] = [1.0] * (c - a)
            binary.append(b)
        i += n
    out = span_report(rows, offsets, labels, binary, 0.5, 1)
    out.update(auc_metrics(probs, labels))
    out["n_docs"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--normalize", action="store_true", help="числа и ссылки -> NUMBER_TOKEN / URL_TOKEN во всех текстах")
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        evaluate_fold(args.run, args.fold, args.normalize)


if __name__ == "__main__":
    main()
