from adblock.data import normalize_name, parse


def ads(response):
    text, spans = parse(response)
    assert "<ad>" not in text and "</ad>" not in text
    return [text[s:e] for s, e in spans]


def test_one_block():
    assert ads("Intro.\n\n<ad>Buy Foo.</ad>\n\nOutro.") == ["Buy Foo."]


def test_several_blocks():
    assert ads("a <ad>one</ad> b <ad>two</ad>") == ["one", "two"]


def test_block_at_start_and_end():
    assert ads("<ad>Start.</ad> body") == ["Start."]
    assert ads("body <ad>End.</ad>") == ["End."]


def test_whitespace_trimmed():
    assert ads("x<ad>\n  Ad.  \n</ad>y") == ["Ad."]


def test_broken_tags():
    assert parse("a <ad>b") is None
    assert parse("a </ad> b") is None


def test_normalize_name():
    assert normalize_name("Luxury Finder.com") == normalize_name("Luxuryfinder.com")


def test_nested_balanced_tags_give_one_block():
    assert ads("a <ad>b <ad>c</ad> d</ad> e") == ["b c d"]
    assert parse("a <ad>b <ad>c</ad> d") is None      # вложенные, но не закрытые


def test_strip_ads():
    from adblock.data import strip_ads
    text, spans = parse("Intro.\n\n<ad>Buy Foo.</ad>\n\nOutro.")
    assert strip_ads(text, spans) == "Intro.\n\nOutro."
    text, spans = parse("One two. <ad>Buy Foo.</ad> Three.")   # реклама внутри абзаца
    assert strip_ads(text, spans) == "One two. Three."
    text, spans = parse("<ad>Buy Foo.</ad>\n\nOutro.")
    assert strip_ads(text, spans) == "Outro."
