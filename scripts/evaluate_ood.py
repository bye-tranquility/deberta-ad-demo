"""Оценка всех моделей вне обучающего распределения, без дообучения и с порогом/min_len, выбранными на val.

1. NaiAD_human — настоящие транскрипты YouTube (607 текстов с целыми тегами, из них 19 без рекламы).
2. test_stripped — тексты test, из которых вырезан рекламный блок (искусственные негативы).
3. test_flat — тексты test, где каждый перенос строки заменён пробелом (разметка та же):
   проверка, насколько модель опирается на переносы строк.
   Это нижняя оценка ложных срабатываний: на месте склейки остаются артефакты,
   а настоящих текстов без рекламы в NaiAD почти нет.

  uv run python scripts/evaluate_ood.py                 # все модели из runs/
  uv run python scripts/evaluate_ood.py --only deberta
Вероятности кэшируются в runs/<имя>/probs_{human,test_stripped}.json, итог — runs/<имя>/metrics_ood.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock import baseline_models as bm  # noqa: E402
from adblock.data import PROCESSED, RUNS, parse, read_jsonl, strip_ads  # noqa: E402
from adblock.metrics import auc_metrics, boundary_errors, char_metrics, detection, evaluate  # noqa: E402
from adblock.tokenize import MODEL, label_tokens  # noqa: E402

MODELS = ["deberta", "baseline_tfidf_para", "baseline_tfidf_window", "baseline_tfidf_sent", "baseline_position"]
HUMAN = ROOT / "data" / "naiad" / "NaiAD_human.jsonl"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_human():
    rows, dropped = [], 0
    for r in read_jsonl(HUMAN):
        parsed = parse(r["response"])
        if parsed is None:
            dropped += 1
            continue
        rows.append({"text": parsed[0], "ad_spans": parsed[1]})
    log(f"human: {len(rows)} текстов, отброшено с битыми тегами: {dropped}")
    return rows


def load_stripped():
    return [{"text": strip_ads(r["text"], r["ad_spans"]), "ad_spans": []}
            for r in read_jsonl(PROCESSED / "test.jsonl")]


def load_flat():
    """\n -> пробел: замена 1 символ на 1 символ, поэтому символьная разметка рекламы не меняется."""
    return [{"text": r["text"].replace("\n", " "), "ad_spans": r["ad_spans"]}
            for r in read_jsonl(PROCESSED / "test.jsonl")]


def get_probs(name, split, rows, tok):
    """Вероятности по токенам (кэш на диске) и метки/оффсеты на сетке DeBERTa."""
    offsets, labels = zip(*((o, y) for _, o, y in (label_tokens(tok, r["text"], r["ad_spans"]) for r in rows)))
    cache = RUNS / name / f"probs_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())["probs"], list(offsets), list(labels)
    log(f"{name}: считаю вероятности на {split} ({len(rows)} текстов)")
    if name == "deberta":
        from adblock.predict import load_deberta, predict
        model, dtok = load_deberta(RUNS / name / "model")
        probs = [predict(model, dtok, r["text"])[0] for r in rows]
    else:
        model = bm.load(RUNS / name / "model.pkl")
        probs = [model.predict(r["text"], o) for r, o in zip(rows, offsets)]
    cache.write_text(json.dumps({"probs": [[round(float(x), 4) for x in p] for p in probs]}))
    return probs, list(offsets), list(labels)


def span_metrics(rows, probs, offsets, labels, thr, min_len):
    """Те же метрики, что на test: только по документам с рекламой."""
    ix = [i for i, r in enumerate(rows) if r["ad_spans"]]
    P, Y = [probs[i] for i in ix], [labels[i] for i in ix]
    return {"n_docs": len(ix), **evaluate(P, Y, thr, min_len), **auc_metrics(P, Y),
            "boundaries_tokens": boundary_errors(P, Y, thr, min_len),
            **char_metrics([rows[i]["text"] for i in ix], [rows[i]["ad_spans"] for i in ix],
                           [offsets[i] for i in ix], P, thr, min_len)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=MODELS)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    human, stripped, flat = load_human(), load_stripped(), load_flat()
    test = read_jsonl(PROCESSED / "test.jsonl")

    for name in MODELS:
        if args.only not in (None, name) or not (RUNS / name / "metrics.json").exists():
            continue
        m = json.loads((RUNS / name / "metrics.json").read_text())
        thr, min_len = m["threshold"], m["min_len"]          # выбраны на val, здесь не меняются
        hp, ho, hl = get_probs(name, "human", human, tok)
        sp, _, _ = get_probs(name, "test_stripped", stripped, tok)
        fp, fo, fl = get_probs(name, "test_flat", flat, tok)
        tp = json.loads((RUNS / name / "probs_test.json").read_text())["probs"]
        assert len(tp) == len(test) == len(sp)

        has_ad = [i for i, r in enumerate(human) if r["ad_spans"]]
        no_ad = [i for i, r in enumerate(human) if not r["ad_spans"]]
        out = {
            "threshold": thr, "min_len": min_len,
            "human": span_metrics(human, hp, ho, hl, thr, min_len),
            # человеческие тексты: с рекламой против без рекламы (без рекламы всего 19 — оценка грубая)
            "human_detection": detection([hp[i] for i in has_ad], [hp[i] for i in no_ad], thr, min_len),
            # синтетика: test (с рекламой) против того же test с вырезанной рекламой
            "test_vs_stripped": detection(tp, sp, thr, min_len),
            "test_flat": span_metrics(flat, fp, fo, fl, thr, min_len),
            "note": "test_stripped — нижняя оценка ложных срабатываний: склейки оставляют артефакты.",
        }
        (RUNS / name / "metrics_ood.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        h, d, s = out["human"], out["human_detection"], out["test_vs_stripped"]
        log(f"{name}: human span_f1@0.5={h['span_f1@0.5']:.3f} | human без рекламы: спан найден в "
            f"{d['found_neg']:.0%} | stripped: спан найден в {s['found_neg']:.1%} | "
            f"test без переносов span_f1@0.5={out['test_flat']['span_f1@0.5']:.3f}")


if __name__ == "__main__":
    main()
