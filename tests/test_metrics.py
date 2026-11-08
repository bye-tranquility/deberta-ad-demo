from adblock.metrics import auc_metrics, boundary_errors, char_metrics

Y = [0, 0, 1, 1, 1, 1, 0, 0]


def test_auc_perfect_and_random():
    perfect = auc_metrics([[0, 0, .9, .9, .9, .9, 0, 0]], [Y])
    assert perfect["pr_auc"] == 1.0 and perfect["roc_auc"] == 1.0
    flat = auc_metrics([[.5] * 8], [Y])
    assert flat["roc_auc"] == 0.5


def test_boundary_errors_sign():
    # предсказание начинается на 1 токен раньше и заканчивается на 1 раньше
    r = boundary_errors([[0, .9, .9, .9, .9, 0, 0, 0]], [Y])
    assert r["matched"] == 1.0
    assert r["start"]["mean"] == -1 and r["end"]["mean"] == -1
    assert r["start"]["within_3"] == 1.0 and r["start"]["exact"] == 0.0


def test_boundary_errors_unmatched():
    r = boundary_errors([[0, 0, .9, 0, 0, 0, 0, 0]], [Y])   # IoU 0.25 < 0.5
    assert r["matched"] == 0.0 and r["start"] == {}


def test_char_leakage_and_loss():
    text = "aa bb CC DD ee ff"
    #       0123456789...
    ad = [(6, 11)]                                   # "CC DD"
    offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]
    # модель вырезала "bb CC": одно рекламное слово осталось, одно нормальное вырезано
    r = char_metrics([text], [ad], [offsets], [[0, .9, .9, 0, 0, 0]])
    assert r["ad_leakage"] == 0.5     # DD (2 из 4 рекламных символов)
    assert r["text_loss"] == 0.25     # bb (2 из 8 нормальных символов)


def test_detection():
    from adblock.metrics import detection
    pos = [[0, .9, .9, .9, 0], [0, .8, .8, .8, 0]]
    neg = [[0, 0, 0, 0, 0], [0, .9, 0, 0, 0]]         # во втором — одиночный шумовой токен
    r = detection(pos, neg, threshold=0.5, min_len=2)
    assert r["found_pos"] == 1.0 and r["found_neg"] == 0.0   # шум отброшен min_len
    assert detection(pos, neg, 0.5, 1)["found_neg"] == 0.5
    assert r["doc_roc_auc"] == 0.625   # пары: 1 + 0.5 (ничья 0.9 vs 0.9) + 1 + 0 (0.8 < 0.9) = 2.5 из 4
