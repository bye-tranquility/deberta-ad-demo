"""NaiAD_human для кросс-валидации: 5 фолдов, похожие рекламные вставки (обычно один спонсор) — в одном фолде.

Запуск: uv run python -m adblock.human   ->  data/processed/human_cv.jsonl

- Берутся только тексты с разметкой рекламы: 19 текстов «без рекламы» исключены, потому что почти все
  они содержат неразмеченную рекламу (см. REPORT.md) и научили бы модель её пропускать.
- Группы: рекламные блоки сравниваются по TF-IDF (косинус > 0.3), тексты с похожими блоками и все блоки
  одного текста объединяются в одну группу; фолды — GroupKFold по этим группам.
- Схема фолда k: test = k, val = (k + 1) % 5 (порог, min_len, лучшая эпоха), train = остальные три.
"""
import collections

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from .data import PROCESSED, ROOT, parse, read_jsonl, write_jsonl

HUMAN_RAW = ROOT / "data" / "naiad" / "NaiAD_human.jsonl"
HUMAN_CV = PROCESSED / "human_cv.jsonl"
N_FOLDS = 5
SIMILARITY = 0.3


def split_roles(fold):
    """Номера фолдов: (train, val, test) для фолда k."""
    test, val = fold, (fold + 1) % N_FOLDS
    return [f for f in range(N_FOLDS) if f not in (test, val)], val, test


def load_cv(fold=None, role=None):
    """Строки human_cv.jsonl; с fold и role ('train'/'val'/'trainval'/'test') — только нужная часть."""
    rows = read_jsonl(HUMAN_CV)
    if fold is None:
        return rows
    train, val, test = split_roles(fold)
    wanted = {"train": set(train), "val": {val}, "trainval": set(train) | {val}, "test": {test}}[role]
    return [r for r in rows if r["fold"] in wanted]


def groups_by_similarity(rows):
    blocks, owner = [], []
    for d, r in enumerate(rows):
        for s, e in r["ad_spans"]:
            blocks.append(r["text"][s:e].replace("NUMBER_TOKEN", " ").replace("URL_TOKEN", " "))
            owner.append(d)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(blocks)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.where(S > SIMILARITY)):
        parent[find(owner[i])] = find(owner[j])
    return [find(d) for d in range(len(rows))]


def main():
    rows, dropped, no_ads = [], 0, 0
    for r in read_jsonl(HUMAN_RAW):
        parsed = parse(r["response"])
        if parsed is None:
            dropped += 1
        elif not parsed[1]:
            no_ads += 1
        else:
            rows.append({"id": r["id"], "text": parsed[0], "ad_spans": parsed[1]})
    groups = groups_by_similarity(rows)
    fold_of = np.zeros(len(rows), dtype=int)
    for k, (_, test_ix) in enumerate(GroupKFold(n_splits=N_FOLDS).split(rows, groups=groups)):
        fold_of[test_ix] = k
    for r, g, k in zip(rows, groups, fold_of):
        r["group"], r["fold"] = int(g), int(k)
    write_jsonl(HUMAN_CV, rows)

    sizes = collections.Counter(groups)
    print(f"human: {len(rows)} текстов с рекламой; отброшено: битые теги {dropped}, без разметки {no_ads}")
    print(f"групп {len(sizes)}, крупнейшие {sorted(sizes.values(), reverse=True)[:5]}")
    for k in range(N_FOLDS):
        part = [r for r in rows if r["fold"] == k]
        print(f"  фолд {k}: {len(part)} текстов, {sum(len(r['ad_spans']) for r in part)} блоков, "
              f"{len({r['group'] for r in part})} групп")
    print(f"записано {HUMAN_CV}")


if __name__ == "__main__":
    main()
