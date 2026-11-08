"""Шаг 2. Сплит по брендам: ни один бренд не попадает в два сплита.

Запуск: uv run python -m adblock.split
"""
from sklearn.model_selection import GroupShuffleSplit

from .data import PROCESSED, RAW, normalize_name, parse, read_jsonl, write_jsonl


def load_rows():
    rows = []
    for r in read_jsonl(RAW):
        parsed = parse(r["response"])
        if parsed is None:
            print("битые теги, пропускаю:", r["id"])
            continue
        text, spans = parsed
        rows.append({"id": r["id"], "text": text, "ad_spans": spans,
                     "brand": normalize_name(r["ad_name"]), "query": r["query"]})
    return rows


def split_by_group(rows, test_size, seed=42):
    groups = [r["brand"] for r in rows]
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(rows, groups=groups))
    return [rows[i] for i in a], [rows[i] for i in b]


def main():
    rows = load_rows()
    train, rest = split_by_group(rows, 0.2)   # 80% брендов в train
    val, test = split_by_group(rest, 0.5)     # остальное пополам

    train_queries = {r["query"] for r in train}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        brands = len({r["brand"] for r in part})
        overlap = sum(r["query"] in train_queries for r in part) / len(part)
        print(f"{name}: {len(part)} строк, {brands} брендов, query из train: {overlap:.0%}")
        write_jsonl(PROCESSED / f"{name}.jsonl", part)


if __name__ == "__main__":
    main()
