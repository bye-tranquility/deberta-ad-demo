"""Предсказание DeBERTa: вероятность рекламы на каждый токен документа."""
import torch

from .tokenize import label_tokens, make_windows, merge_windows

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_deberta(path):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    # dtype=float32: без него transformers 5 грузит DeBERTa в fp16
    model = AutoModelForTokenClassification.from_pretrained(path, dtype=torch.float32).to(DEVICE).eval()
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, text, spans=()):
    """Возвращает (вероятности по токенам, метки по токенам). Перекрытия окон усреднены."""
    ids, _, labels = label_tokens(tokenizer, text, spans)
    windows = make_windows(tokenizer, ids, labels)
    window_probs = []
    for w in windows:
        x = torch.tensor([w["input_ids"]], device=DEVICE)
        window_probs.append(model(input_ids=x).logits.softmax(-1)[0, 1:-1, 1].tolist())  # без [CLS]/[SEP]
    return merge_windows(len(ids), windows, window_probs), labels
