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



URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|io|co|gg|tv|ly|ai|app|me|us|uk|gov|edu)\b",
                    re.IGNORECASE)
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _replace(text, spans, pattern, token):
    """Замена совпадений pattern на token с пересчётом спанов.

    piece[i] — кусок нового текста, в который превратился старый символ i: обычный символ -> (n, n + 1),
    символ внутри замены -> весь токен. Спан (s, e) -> (начало куска s, конец куска e - 1).
    """
    out, piece, pos = [], [], 0
    for m in pattern.finditer(text):
        for ch in text[pos:m.start()]:
            n = sum(map(len, out)) if not piece else piece[-1][1]
            piece.append((n, n + 1))
            out.append(ch)
        n = piece[-1][1] if piece else 0
        piece += [(n, n + len(token))] * (m.end() - m.start())
        out.append(token)
        pos = m.end()
    for ch in text[pos:]:
        n = piece[-1][1] if piece else 0
        piece.append((n, n + 1))
        out.append(ch)
    return "".join(out), [(piece[a][0], piece[b - 1][1]) for a, b in spans]


def normalize(text, spans):
    """Предобработка как в NaiAD_human: домен ссылки -> URL_TOKEN (путь остаётся), группа цифр -> NUMBER_TOKEN.

    Возвращает (новый текст, спаны рекламы в новых координатах). На уже обработанном тексте ничего не меняет.
    """
    text, spans = _replace(text, [tuple(x) for x in spans], URL_RE, "URL_TOKEN")
    return _replace(text, spans, NUMBER_RE, "NUMBER_TOKEN")


def normalize_row(r):
    text, spans = normalize(r["text"], r["ad_spans"])
    return {**r, "text": text, "ad_spans": [list(x) for x in spans]}
