"""Декодер: вероятности токенов -> рекламные блоки (спаны токенов [a, b)).

Шаги:
1. порог: токены с вероятностью выше threshold;
2. склейка: соседние куски, между которыми не больше merge_gap токенов, объединяются в один блок
   (длинная вставка часто находится кусками с короткими разрывами);
3. фильтр: блоки короче min_len токенов отбрасываются как шум (после склейки, чтобы куски одной
   вставки сначала соединились);
4. выравнивание: начало и конец блока сдвигаются к ближайшей границе предложения, если она не дальше
   snap токенов (snap = 0 — выключено); пересёкшиеся после сдвига блоки сливаются.

Старый декодер — частный случай: merge_gap = 0, snap = 0.
"""
from bisect import bisect_left

from .units import sentences


def merge_close(spans, gap):
    out = []
    for a, b in spans:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def raw_spans(probs, threshold):
    spans, start = [], None
    for i, p in enumerate(list(probs) + [0.0]):
        if p > threshold and start is None:
            start = i
        elif p <= threshold and start is not None:
            spans.append((start, i))
            start = None
    return spans


def sentence_boundaries(text, offsets):
    """Индексы токенов, с которых начинаются предложения, плюс конец документа (len(offsets))."""
    starts = [s for s, _ in offsets]
    bounds = {bisect_left(starts, a) for a, *_ in sentences(text)}
    bounds.add(len(offsets))
    bounds.add(0)
    return sorted(b for b in bounds if b <= len(offsets))


def _nearest(bounds, x, limit):
    """Ближайшая к x граница предложения, если она не дальше limit токенов; иначе сам x."""
    i = bisect_left(bounds, x)
    near = min((bounds[j] for j in (i - 1, i) if 0 <= j < len(bounds)), key=lambda v: abs(v - x))
    return near if abs(near - x) <= limit else x


def snap_spans(spans, bounds, limit):
    if not limit:
        return spans
    out = []
    for a, b in spans:
        a2, b2 = _nearest(bounds, a, limit), _nearest(bounds, b, limit)
        if b2 > a2:
            out.append((a2, b2))
    return merge_close(sorted(out), 0)


def decode(probs, threshold=0.5, min_len=1, merge_gap=0, snap=0, bounds=None):
    spans = merge_close(raw_spans(probs, threshold), merge_gap)
    spans = [s for s in spans if s[1] - s[0] >= min_len]
    if snap:
        spans = snap_spans(spans, bounds, snap)
    return spans


def to_binary(spans, n):
    out = [0.0] * n
    for a, b in spans:
        out[a:b] = [1.0] * (b - a)
    return out
