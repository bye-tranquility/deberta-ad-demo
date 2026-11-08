"""Текстовые единицы для baselines: абзацы и предложения, их метки и перенос на токены."""
import re
from bisect import bisect_right

LINE = re.compile(r"[^\n]+")
SENT_END = re.compile(r"[.!?…]+(?=\s)")


def _trim(text, a, b):
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def paragraphs(text):
    """Абзацы = строки между \\n; пустые пропускаются. Возвращает [(start, end)]."""
    out = []
    for m in LINE.finditer(text):
        a, b = _trim(text, m.start(), m.end())
        if b > a:
            out.append((a, b))
    return out


def sentences(text):
    """Предложения внутри абзацев: разрыв после .!?… перед пробелом. Абзацы не склеиваются.

    Возвращает [(start, end, первое_в_абзаце, последнее_в_абзаце)].
    """
    out = []
    for ps, pe in paragraphs(text):
        cuts = [ps] + [m.end() for m in SENT_END.finditer(text, ps, pe)] + [pe]
        segs = [s for s in (_trim(text, a, b) for a, b in zip(cuts, cuts[1:])) if s[1] > s[0]]
        for k, (a, b) in enumerate(segs):
            out.append((a, b, k == 0, k == len(segs) - 1))
    return out


def unit_labels(text, ad_spans, units):
    """Метка единицы = 1, если больше половины её непробельных символов внутри рекламы."""
    is_ad = bytearray(len(text))
    for s, e in ad_spans:
        is_ad[s:e] = b"\x01" * (e - s)
    labels = []
    for u in units:
        a, b = u[0], u[1]
        total = ad = 0
        for i in range(a, b):
            if not text[i].isspace():
                total += 1
                ad += is_ad[i]
        labels.append(int(2 * ad > total))
    return labels


def to_tokens(offsets, units, unit_probs):
    """Каждому токену — вероятность единицы, в которую попадает его начало (0, если ни в какую)."""
    starts = [u[0] for u in units]
    out = []
    for s, _ in offsets:
        j = bisect_right(starts, s) - 1
        out.append(float(unit_probs[j]) if j >= 0 and s < units[j][1] else 0.0)
    return out


WORD = re.compile(r"\S+")


def windows(text, center=10, context=20):
    """Скользящие окна по словам внутри строки (окно обрывается на переносе строки).

    Центр — `center` подряд идущих слов, центры не перекрываются и покрывают все слова строки.
    Левый/правый контекст — до `context` слов слева/справа в той же строке (у края строки — пусто).
    Возвращает [(a, b, la, lb, ra, rb)]: символьные границы центра, левого и правого контекста.
    """
    out = []
    for m in LINE.finditer(text):
        words = [(w.start(), w.end()) for w in WORD.finditer(text, m.start(), m.end())]
        for i in range(0, len(words), center):
            c = words[i:i + center]
            left = words[max(0, i - context):i]
            right = words[i + center:i + center + context]
            la, lb = (left[0][0], left[-1][1]) if left else (c[0][0], c[0][0])
            ra, rb = (right[0][0], right[-1][1]) if right else (c[-1][1], c[-1][1])
            out.append((c[0][0], c[-1][1], la, lb, ra, rb))
    return out
