import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class FakeTokenizer:
    """Токен = слово. Оффсет включает ведущий пробел — как худший случай у SentencePiece."""
    cls_token_id, sep_token_id = 1, 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        offsets = [m.span() for m in re.finditer(r"\s*\S+", text)]
        return {"input_ids": list(range(10, 10 + len(offsets))), "offset_mapping": offsets}


@pytest.fixture
def tok():
    return FakeTokenizer()
