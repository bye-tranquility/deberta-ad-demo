from adblock.units import paragraphs, sentences, to_tokens, unit_labels

TEXT = "Intro one. Intro two!\n\n  \nSpeaking of X, try Foo. Foo is great…\nBye."


def test_paragraphs_skip_empty_lines():
    assert [TEXT[a:b] for a, b in paragraphs(TEXT)] == [
        "Intro one. Intro two!", "Speaking of X, try Foo. Foo is great…", "Bye."]


def test_sentences_and_flags():
    s = sentences(TEXT)
    assert [TEXT[a:b] for a, b, *_ in s] == [
        "Intro one.", "Intro two!", "Speaking of X, try Foo.", "Foo is great…", "Bye."]
    assert [(f, l) for *_, f, l in s] == [(True, False), (False, True), (True, False), (False, True), (True, True)]


def test_unit_labels_majority_of_nonspace_chars():
    text = "aaaa bbbb cc"
    assert unit_labels(text, [(5, 12)], [(0, 12)]) == [1]   # 6 из 10 непробельных — реклама
    assert unit_labels(text, [(10, 12)], [(0, 12)]) == [0]  # 2 из 10


def test_to_tokens():
    units = [(0, 4), (6, 10)]
    offsets = [(0, 2), (2, 4), (4, 5), (6, 10)]       # третий токен между единицами
    assert to_tokens(offsets, units, [0.1, 0.9]) == [0.1, 0.1, 0.0, 0.9]


def test_windows_cut_at_newline_and_cover_all_words():
    from adblock.units import windows
    text = "w1 w2 w3 w4 w5\nv1 v2 v3"
    ws = windows(text, center=2, context=2)
    centers = [text[a:b] for a, b, *_ in ws]
    assert centers == ["w1 w2", "w3 w4", "w5", "v1 v2", "v3"]       # центр не переходит через \n
    left = [text[la:lb] for _, _, la, lb, _, _ in ws]
    right = [text[ra:rb] for *_, ra, rb in ws]
    assert left == ["", "w1 w2", "w3 w4", "", "v1 v2"]              # контекст тоже обрывается на \n
    assert right == ["w3 w4", "w5", "", "v3", ""]
