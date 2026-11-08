from adblock.data import parse
from adblock.metrics import evaluate, to_spans
from adblock.tokenize import MAX_LEN, STRIDE, label_tokens, make_windows, merge_windows, window_starts


def test_ad_boundaries(tok):
    text, spans = parse("Hello there.\n\n<ad>Try Foo today.</ad>\n\nBye now.")
    _, offsets, labels = label_tokens(tok, text, spans)
    words = [text[s:e] for s, e in offsets]
    assert words == ["Hello", "there.", "Try", "Foo", "today.", "Bye", "now."]
    assert labels == [0, 0, 1, 1, 1, 0, 0]   # ведущий "\n\n" у "Try" не сдвинул границу


def test_windows_cover_everything():
    for n in (10, 510, 511, 2000):
        starts = window_starts(n)
        assert starts[0] == 0 and starts[-1] + MAX_LEN - 2 >= n
        assert all(b - a == MAX_LEN - 2 - STRIDE for a, b in zip(starts, starts[1:]))


def test_window_format(tok):
    ids, _, labels = label_tokens(tok, " ".join(["w"] * 1000), [(0, 5)])
    for w in make_windows(tok, ids, labels):
        assert len(w["input_ids"]) <= MAX_LEN
        assert w["input_ids"][0] == tok.cls_token_id and w["input_ids"][-1] == tok.sep_token_id
        assert w["labels"][0] == w["labels"][-1] == -100


def test_merge_windows_counts_each_token_once(tok):
    ids, _, labels = label_tokens(tok, " ".join(["w"] * 1000), [])
    windows = make_windows(tok, ids, labels)
    # первое окно предсказывает 1.0, второе 0.0 -> в перекрытии должно быть 0.5
    window_probs = [[1.0 if i == 0 else 0.0] * (len(w["input_ids"]) - 2) for i, w in enumerate(windows)]
    probs = merge_windows(len(ids), windows, window_probs)
    assert len(probs) == len(ids)                      # одна вероятность на токен
    overlap = range(windows[1]["start"], windows[0]["start"] + MAX_LEN - 2)
    assert len(overlap) == STRIDE
    assert all(probs[i] == 0.5 for i in overlap)
    assert probs[0] == 1.0 and probs[-1] == 0.0


def test_metrics():
    assert to_spans([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    y = [0, 1, 1, 1, 1, 0]
    assert evaluate([[0, .9, .9, .9, .9, 0]], [y])["span_f1@0.5"] == 1.0
    r = evaluate([[0, .9, .9, 0, 0, 0]], [y])          # IoU = 0.5
    assert r["span_f1@0.3"] == 1.0 and r["span_f1@0.5"] == 1.0 and r["span_f1@0.7"] == 0.0


def test_min_len_drops_short_spans():
    y = [0, 1, 1, 1, 1, 0, 0, 0]
    p = [0, .9, .9, .9, .9, 0, 0, .9]                   # настоящий блок + шумовой токен в конце
    assert evaluate([p], [y], min_len=1)["span_f1@0.5"] < 1.0
    assert evaluate([p], [y], min_len=2)["span_f1@0.5"] == 1.0
