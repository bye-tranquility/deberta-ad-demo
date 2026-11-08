"""Шаг 5. Метрики: по токенам и по спанам (блокам рекламы)."""

IOUS = (0.3, 0.5, 0.7)


def to_spans(binary):
    """[0,1,1,0,1] -> [(1,3), (4,5)]: подряд идущие единицы."""
    spans, start = [], None
    for i, x in enumerate(list(binary) + [0]):
        if x and start is None:
            start = i
        if not x and start is not None:
            spans.append((start, i))
            start = None
    return spans


def predict_spans(probs, threshold=0.5, min_len=1):
    """Спаны из вероятностей. Спаны короче min_len токенов отбрасываются как шум."""
    return [s for s in to_spans([p > threshold for p in probs]) if s[1] - s[0] >= min_len]


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter)


def f1(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def evaluate(probs, labels, threshold=0.5, min_len=1):
    """probs, labels — списки по документам (вероятность рекламы и 0/1 на каждый токен).

    Перед подсчётом из предсказания удаляются спаны короче min_len токенов,
    и token-метрики, и span-метрики считаются по одному и тому же отфильтрованному предсказанию.
    Спан засчитан, если есть предсказанный спан с IoU >= порога (для каждого порога из IOUS).
    """
    tp = n_pred = n_true = 0
    span_tp = {t: 0 for t in IOUS}
    span_pred = span_true = 0
    for p, y in zip(probs, labels):
        ps, ys = predict_spans(p, threshold, min_len), to_spans(y)
        pred = [0] * len(y)
        for a, b in ps:
            pred[a:b] = [1] * (b - a)
        tp += sum(a and b for a, b in zip(pred, y))
        n_pred += sum(pred)
        n_true += sum(y)

        span_pred += len(ps)
        span_true += len(ys)
        for t in IOUS:
            span_tp[t] += sum(any(iou(a, b) >= t for a in ps) for b in ys)

    out = {"token_f1": f1(tp, n_pred, n_true)}
    for t in IOUS:
        out[f"span_f1@{t}"] = f1(span_tp[t], span_pred, span_true)
    out["pred_spans_per_doc"] = span_pred / max(1, len(labels))
    return out


def auc_metrics(probs, labels):
    """PR-AUC и ROC-AUC для класса «реклама». Не зависят от порога и min_len.

    pooled   — по всем токенам всех документов сразу (длинные документы весят больше);
    doc_mean — посчитано в каждом документе и усреднено (документы равноправны).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    flat_p = [x for p in probs for x in p]
    flat_y = [x for y in labels for x in y]
    out = {"pr_auc": average_precision_score(flat_y, flat_p), "roc_auc": roc_auc_score(flat_y, flat_p)}
    pr, roc = [], []
    for p, y in zip(probs, labels):
        if 0 < sum(y) < len(y):          # нужны оба класса
            pr.append(average_precision_score(y, p))
            roc.append(roc_auc_score(y, p))
    out["pr_auc_doc_mean"] = sum(pr) / max(1, len(pr))
    out["roc_auc_doc_mean"] = sum(roc) / max(1, len(roc))
    return out


def boundary_errors(probs, labels, threshold=0.5, min_len=1):
    """Ошибка границ в токенах со знаком: предсказание минус правда.

    Для каждого настоящего блока берём предсказанный спан с наибольшим IoU (если IoU >= 0.5).
    start < 0 — начали раньше (захватили лишний текст), start > 0 — пропустили начало рекламы.
    end > 0 — захватили лишний текст после блока, end < 0 — оставили хвост рекламы.
    """
    starts, ends, n_true = [], [], 0
    for p, y in zip(probs, labels):
        ps = predict_spans(p, threshold, min_len)
        for g in to_spans(y):
            n_true += 1
            best = max(ps, key=lambda s: iou(s, g), default=None)
            if best is not None and iou(best, g) >= 0.5:
                starts.append(best[0] - g[0])
                ends.append(best[1] - g[1])

    def stats(errs):
        if not errs:
            return {}
        s = sorted(errs)
        return {"mean": sum(errs) / len(errs), "median": s[len(s) // 2],
                "mean_abs": sum(map(abs, errs)) / len(errs),
                "exact": sum(e == 0 for e in errs) / len(errs),
                "within_3": sum(abs(e) <= 3 for e in errs) / len(errs),
                "within_10": sum(abs(e) <= 10 for e in errs) / len(errs)}

    return {"matched": len(starts) / max(1, n_true), "start": stats(starts), "end": stats(ends)}


def char_metrics(texts, ad_spans, offsets, probs, threshold=0.5, min_len=1):
    """Что будет, если вырезать предсказанную рекламу из текста (считаем непробельные символы).

    ad_leakage — доля рекламных символов, которые остались в тексте;
    text_loss  — доля нормальных символов, которые вырезали по ошибке.
    Предсказанный спан покрывает символы от начала первого токена до конца последнего.
    """
    ad_total = ad_left = text_total = text_cut = 0
    for text, spans, offs, p in zip(texts, ad_spans, offsets, probs):
        is_ad = [False] * len(text)
        for a, b in spans:
            is_ad[a:b] = [True] * (b - a)
        cut = [False] * len(text)
        for a, b in predict_spans(p, threshold, min_len):
            s, e = offs[a][0], offs[b - 1][1]
            cut[s:e] = [True] * (e - s)
        for ch, ad, c in zip(text, is_ad, cut):
            if ch.isspace():
                continue
            if ad:
                ad_total += 1
                ad_left += not c
            else:
                text_total += 1
                text_cut += c
    return {"ad_leakage": ad_left / max(1, ad_total), "text_loss": text_cut / max(1, text_total)}


# Сетка декодера: общая для evaluate.py и для выбора гиперпараметров baselines.
THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]   # 0.10 … 0.90
# в токенах; настоящие блоки рекламы в train не короче 23 токенов (1% короче 37),
# поэтому дальше 40 не идём, иначе начнём выбрасывать настоящую рекламу
MIN_LENS = [1, 2, 3, 5, 10, 20, 30, 40]


def best_span_f1(probs, labels, iou_thr=0.5):
    """Лучший span_f1@iou_thr по сетке THRESHOLDS x MIN_LENS — то же, что даст общий декодер.

    Быстрая версия: спаны для каждого порога строятся один раз, min_len только фильтрует их.
    Возвращает (лучший span_f1, порог, min_len).
    """
    import numpy as np

    gts = [to_spans(y) for y in labels]
    n_true = sum(map(len, gts))
    best = (-1.0, None, None)
    for t in THRESHOLDS:
        all_pred = []
        for p in probs:
            x = np.concatenate([[0], (np.asarray(p) > t).astype(np.int8), [0]])
            d = np.flatnonzero(np.diff(x))
            all_pred.append(list(zip(d[::2], d[1::2])))
        for m in MIN_LENS:
            tp = n_pred = 0
            for ps, gs in zip(all_pred, gts):
                ps = [s for s in ps if s[1] - s[0] >= m]
                n_pred += len(ps)
                tp += sum(any(iou(a, g) >= iou_thr for a in ps) for g in gs)
            score = f1(tp, n_pred, n_true)
            if score > best[0]:
                best = (score, t, m)
    return best


def detection(pos_probs, neg_probs, threshold, min_len):
    """Есть ли в документе реклама: документы с рекламой (pos) против документов без неё (neg).

    found_*  — доля документов, где после порога и min_len найден хотя бы один спан
               (found_neg — доля ложных срабатываний, found_pos — доля найденных);
    flagged_neg_tokens — доля токенов документов без рекламы, помеченных как реклама;
    doc_roc_auc / doc_pr_auc — скор документа = максимальная вероятность по токенам (без порога);
    tpr_at_fpr_* — какую долю документов с рекламой ловим, если допускаем 1% / 5% ложных срабатываний.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    def found(probs):
        return sum(bool(predict_spans(p, threshold, min_len)) for p in probs) / max(1, len(probs))

    def flagged(probs):
        spans = [predict_spans(p, threshold, min_len) for p in probs]
        return sum(b - a for s in spans for a, b in s) / max(1, sum(len(p) for p in probs))

    score = lambda p: max(p) if len(p) else 0.0  # noqa: E731
    y = [1] * len(pos_probs) + [0] * len(neg_probs)
    s = [score(p) for p in pos_probs] + [score(p) for p in neg_probs]
    out = {"n_pos": len(pos_probs), "n_neg": len(neg_probs),
           "found_pos": found(pos_probs), "found_neg": found(neg_probs),
           "flagged_neg_tokens": flagged(neg_probs)}
    if pos_probs and neg_probs:
        fpr, tpr, _ = roc_curve(y, s)
        out.update(doc_roc_auc=roc_auc_score(y, s), doc_pr_auc=average_precision_score(y, s),
                   tpr_at_fpr_1=float(np.interp(0.01, fpr, tpr)), tpr_at_fpr_5=float(np.interp(0.05, fpr, tpr)))
    return out
