"""Шаг 5. Оценка: порог и min_len подбираем на val, потом один раз считаем test.

  uv run python scripts/evaluate.py --name smoke
Вероятности сохраняются в runs/<name>/probs_<split>.json, повторный запуск их переиспользует.
"""
import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock.data import PROCESSED, RUNS, read_jsonl  # noqa: E402
from adblock.metrics import MIN_LENS, THRESHOLDS, auc_metrics, boundary_errors, char_metrics, evaluate  # noqa: E402
from adblock.predict import load_deberta, predict  # noqa: E402
from adblock.tokenize import MODEL, label_tokens  # noqa: E402


def get_probs(run_dir, split, rows, load_model):
    cache = run_dir / f"probs_{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    model, tokenizer = load_model()
    probs, labels = zip(*(predict(model, tokenizer, r["text"], r["ad_spans"]) for r in rows))
    data = {"probs": list(probs), "labels": list(labels)}
    cache.write_text(json.dumps(data))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="deberta")
    args = ap.parse_args()
    run_dir = RUNS / args.name

    loaded = {}
    def load_model():
        if not loaded:
            loaded["model"], loaded["tok"] = load_deberta(run_dir / "model")
        return loaded["model"], loaded["tok"]

    n = 200 if args.name.startswith("smoke") else None   # smoke-модели оцениваем на 200 текстах
    val = get_probs(run_dir, "val", read_jsonl(PROCESSED / "val.jsonl")[:n], load_model)
    test_rows = read_jsonl(PROCESSED / "test.jsonl")[:n]
    test = get_probs(run_dir, "test", test_rows, load_model)

    # подбор порога и минимальной длины спана — только на val, по span_f1@0.5
    grid = {(t, m): evaluate(val["probs"], val["labels"], t, m) for t in THRESHOLDS for m in MIN_LENS}
    thr, min_len = max(grid, key=lambda k: grid[k]["span_f1@0.5"])

    print("Влияние min_len на val при выбранном пороге:")
    for m in MIN_LENS:
        r = grid[(thr, m)]
        print(f"  min_len={m:3d}  " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))

    # оффсеты токенов нужны для метрик в символах; токенизатор тот же, что у модели
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    offsets = [label_tokens(tokenizer, r["text"], r["ad_spans"])[1] for r in test_rows]
    tp, tl = test["probs"], test["labels"]

    result = {"threshold": thr, "min_len": min_len,
              "val": grid[(thr, min_len)],
              "test": evaluate(tp, tl, thr, min_len),
              "test_no_filter": evaluate(tp, tl, thr, 1),
              "test_auc": auc_metrics(tp, tl),                          # не зависит от порога
              "test_boundaries_tokens": boundary_errors(tp, tl, thr, min_len),
              "test_chars": char_metrics([r["text"] for r in test_rows], [r["ad_spans"] for r in test_rows],
                                         offsets, tp, thr, min_len)}
    print(json.dumps(result, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
