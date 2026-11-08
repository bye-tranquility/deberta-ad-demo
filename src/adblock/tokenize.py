"""Шаг 3. Текст -> токены с метками 0 (не реклама) / 1 (реклама) -> окна по 512 токенов."""

MODEL = "microsoft/deberta-v3-base"
MAX_LEN = 512   # длина окна вместе с [CLS] и [SEP]
STRIDE = 128    # перекрытие соседних окон


def label_tokens(tokenizer, text, spans):
    """Возвращает ids, offsets, labels для всего текста (без спецтокенов).

    Токен = реклама, если его символы пересекаются с рекламным спаном.
    SentencePiece иногда включает в оффсет ведущий пробел — срезаем его,
    иначе токен перед рекламой мог бы задеть её начало.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets, labels = [], []
    for s, e in enc["offset_mapping"]:
        while s < e and text[s].isspace():
            s += 1
        offsets.append((s, e))
        labels.append(int(any(s < b and e > a for a, b in spans)))
    return enc["input_ids"], offsets, labels


def window_starts(n_tokens):
    """Начала окон. Окно вмещает MAX_LEN-2 токенов, соседние окна перекрываются на STRIDE."""
    size = MAX_LEN - 2
    starts = [0]
    while starts[-1] + size < n_tokens:
        starts.append(starts[-1] + size - STRIDE)
    return starts


def merge_windows(n_tokens, windows, window_probs):
    """Собирает вероятности окон обратно в документ: одна вероятность на токен.

    Токен из перекрытия двух окон получает среднее из двух предсказаний,
    поэтому в метриках каждый токен считается ровно один раз.
    """
    total, count = [0.0] * n_tokens, [0] * n_tokens
    for w, probs in zip(windows, window_probs):
        for k, p in enumerate(probs):
            total[w["start"] + k] += p
            count[w["start"] + k] += 1
    return [t / c for t, c in zip(total, count)]


def make_windows(tokenizer, ids, labels):
    """Режет документ на окна, добавляет [CLS]/[SEP]. Метка -100 = не учитывать в loss."""
    size = MAX_LEN - 2
    out = []
    for a in window_starts(len(ids)):
        out.append({
            "start": a,
            "input_ids": [tokenizer.cls_token_id] + ids[a:a + size] + [tokenizer.sep_token_id],
            "labels": [-100] + labels[a:a + size] + [-100],
        })
    return out
