"""Шаг 1. Чтение NaiAD и разбор тегов <ad>…</ad>."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "naiad" / "NaiAD_main.jsonl"
# на Modal данные и результаты лежат на volume, пути задаются переменными окружения
PROCESSED = Path(os.environ.get("ADBLOCK_PROCESSED", ROOT / "data" / "processed"))
RUNS = Path(os.environ.get("ADBLOCK_RUNS", ROOT / "runs"))

TAG = re.compile(r"<(/?)ad>")


def parse(response):
    """'a <ad>b</ad> c' -> ('a b c', [(2, 3)]). При битых тегах возвращает None.

    Вложенные, но сбалансированные теги (<ad>a <ad>b</ad> c</ad>) дают один внешний блок:
    такое встречается в NaiAD_human.
    """
    text, spans = "", []
    pos, start, depth = 0, None, 0
    for m in TAG.finditer(response):
        text += response[pos:m.start()]
        pos = m.end()
        if m.group(1) == "":          # <ad>
            if depth == 0:
                start = len(text)
            depth += 1
        else:                          # </ad>
            if depth == 0:
                return None
            depth -= 1
            if depth == 0:
                spans.append((start, len(text)))
    if depth != 0:
        return None
    text += response[pos:]

    # убираем пробелы/переводы строк по краям рекламы
    clean = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            clean.append((s, e))
    return text, clean


def normalize_name(name):
    """'Luxury Finder.com' -> 'luxuryfindercom'."""
    return re.sub(r"[\W_]+", "", name.lower())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def strip_ads(text, spans):
    """Вырезает рекламные блоки вместе с пробелами вокруг (для проверки на текстах без рекламы).

    На месте склейки ставится тот же тип разрыва, что был вокруг блока:
    пустая строка, перенос строки или пробел (если реклама была внутри абзаца).
    """
    for s, e in sorted(spans, reverse=True):
        left, right = text[:s].rstrip(), text[e:].lstrip()
        gap = text[len(left):s] + text[e:len(text) - len(right)]
        sep = "\n\n" if "\n\n" in gap else "\n" if "\n" in gap else " "
        text = left + (sep if left and right else "") + right
    return text
