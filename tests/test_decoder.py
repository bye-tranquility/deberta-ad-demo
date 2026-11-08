from adblock.decoder import decode, merge_close, sentence_boundaries, snap_spans

P = [0, .9, .9, 0, 0, .9, .9, .9, 0, 0, 0, 0, .9, 0]


def test_old_decoder_is_special_case():
    from adblock.metrics import predict_spans
    for thr in (0.5, 0.95):
        for ml in (1, 2, 3):
            assert decode(P, thr, ml) == predict_spans(P, thr, ml)


def test_merge_before_min_len():
    assert decode(P, 0.5, min_len=1, merge_gap=2) == [(1, 8), (12, 13)]
    # куски по 2 и 3 токена по отдельности отбрасываются min_len=5, склеенные — нет
    assert decode(P, 0.5, min_len=5) == []
    assert decode(P, 0.5, min_len=5, merge_gap=2) == [(1, 8)]
    assert merge_close([(0, 2), (10, 12)], 5) == [(0, 2), (10, 12)]


def test_snap_to_sentences():
    bounds = [0, 4, 9, 20]
    assert snap_spans([(5, 8)], bounds, 2) == [(4, 9)]        # обе границы в пределах 2 токенов
    assert snap_spans([(5, 8)], bounds, 0) == [(5, 8)]        # выключено
    assert snap_spans([(6, 15)], bounds, 3) == [(4, 15)]      # конец дальше 3 от границ — не трогаем
    assert snap_spans([(1, 3), (5, 8)], bounds, 5) == [(0, 9)]  # после сдвига слились


def test_sentence_boundaries():
    text = "One two. Three four five. Six."
    offsets = [(0, 3), (4, 8), (9, 14), (15, 19), (20, 25), (26, 30)]
    assert sentence_boundaries(text, offsets) == [0, 2, 5, 6]
