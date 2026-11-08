"""Модели baselines: обучаются в scripts/baselines.py, сохраняются в runs/<имя>/model.pkl,
используются в демо. predict(text, offsets) -> вероятность рекламы на каждый токен DeBERTa."""
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .units import paragraphs, sentences, to_tokens, windows


def smooth(p, units, window):
    """Среднее вероятностей по окну соседних единиц внутри документа (window=1 — без сглаживания)."""
    if window == 1:
        return p
    half, out, k = window // 2, np.empty_like(p), 0
    for u in units:
        q = p[k:k + len(u)]
        for i in range(len(u)):
            out[k + i] = q[max(0, i - half):i + half + 1].mean()
        k += len(u)
    return out


def neighbours(cur, units):
    """Матрицы [предыдущая единица] и [следующая] внутри документа; у краёв — нулевой вектор."""
    n = cur.shape[0]
    pad = sp.vstack([cur, sp.csr_matrix((1, cur.shape[1]))]).tocsr()   # строка n — нулевая
    prev_idx, next_idx, k = [], [], 0
    for u in units:
        for i in range(len(u)):
            prev_idx.append(k + i - 1 if i > 0 else n)
            next_idx.append(k + i + 1 if i < len(u) - 1 else n)
        k += len(u)
    return pad[prev_idx], pad[next_idx]


class PositionPrior:
    """Вероятность рекламы = доля рекламных токенов в корзине относительной позиции (по train)."""

    def __init__(self, table):
        self.table = table

    def predict(self, text, offsets):
        n = len(self.table)
        return [self.table[min(n - 1, int(s / max(1, len(text)) * n))] for s, _ in offsets]


class TfidfBaseline:
    """TF-IDF + LogReg по текстовым единицам.

    kind="para"   — абзац;
    kind="sent"   — [предыдущее | текущее | следующее предложение | первое/последнее в абзаце];
    kind="window" — [левый контекст | центр окна | правый контекст], окно обрывается на \\n.
    """

    def __init__(self, kind, center=10, context=20):
        self.kind, self.center, self.context = kind, center, context
        self.vec = self.clf = None
        self.smooth_window = 1

    def units(self, text):
        if self.kind == "para":
            return paragraphs(text)
        if self.kind == "sent":
            return sentences(text)
        return windows(text, self.center, self.context)

    def _texts(self, texts, units, part=0):
        """Тексты единиц; part — индекс начала в кортеже единицы (0 — сама единица/центр окна)."""
        return [t[u[part]:u[part + 1]] for t, us in zip(texts, units) for u in us]

    def fit_vectorizer(self, texts, units):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        self.vec.fit(self._texts(texts, units))
        return self

    def matrix(self, texts, units):
        cur = self.vec.transform(self._texts(texts, units)).tocsr()
        if self.kind == "para":
            return cur
        if self.kind == "sent":
            prev, nxt = neighbours(cur, units)
            flags = sp.csr_matrix(np.array([(float(f), float(l)) for us in units for *_, f, l in us]).reshape(-1, 2))
            return sp.hstack([prev, cur, nxt, flags]).tocsr()
        left = self.vec.transform(self._texts(texts, units, 2))
        right = self.vec.transform(self._texts(texts, units, 4))
        return sp.hstack([left, cur, right]).tocsr()

    def unit_probs(self, X, units):
        return smooth(self.clf.predict_proba(X)[:, 1], units, self.smooth_window)

    def predict(self, text, offsets):
        units = self.units(text)
        if not units:
            return [0.0] * len(offsets)
        return to_tokens(offsets, units, self.unit_probs(self.matrix([text], [units]), [units]))


def save(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)
