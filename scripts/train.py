"""Шаг 4. Обучение DeBERTa размечать токены рекламы.

  uv run python scripts/train.py --smoke     # быстрая проверка: 500 текстов, 1 эпоха
  uv run python scripts/train.py             # полное обучение
Модель сохраняется в runs/<name>/model, история loss — в runs/<name>/log.json.
На GPU (Modal) запускается через scripts/modal_train.py.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock.data import PROCESSED, RUNS, read_jsonl  # noqa: E402
from adblock.tokenize import MODEL, label_tokens, make_windows  # noqa: E402


def build_dataset(tokenizer, rows):
    windows = []
    for r in rows:
        ids, _, labels = label_tokens(tokenizer, r["text"], r["ad_spans"])
        for w in make_windows(tokenizer, ids, labels):
            windows.append({"input_ids": w["input_ids"], "labels": w["labels"]})
    return windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16, help="эффективный батч (на Mac набирается накоплением по 4)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    train = read_jsonl(PROCESSED / "train.jsonl")
    val = read_jsonl(PROCESSED / "val.jsonl")
    epochs = args.epochs
    if args.smoke:
        train, val, epochs = train[:500], val[:200], 1
    args.name = args.name or ("smoke" if args.smoke else "deberta")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds, val_ds = build_dataset(tokenizer, train), build_dataset(tokenizer, val)
    print(f"окон: train={len(train_ds)}, val={len(val_ds)}")

    # dtype=float32 обязателен: без него transformers 5 грузит веса DeBERTa в fp16,
    # и AdamW после первого же шага превращает их в nan
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, num_labels=2, id2label={0: "O", 1: "AD"}, label2id={"O": 0, "AD": 1}, dtype=torch.float32)

    out = RUNS / args.name
    gpu = torch.cuda.is_available()
    if gpu:   # CUDA (Modal): весь батч сразу помещается в память
        batch, accum, checkpointing = args.batch_size, 1, False
    else:     # Mac: батч набирается накоплением по 4, gradient checkpointing экономит память
        batch, accum, checkpointing = 4, max(1, args.batch_size // 4), True
    print(f"gpu={gpu} batch={batch}x{accum} lr={args.lr} epochs={epochs} gradient_checkpointing={checkpointing}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch,
            gradient_accumulation_steps=accum,
            per_device_eval_batch_size=batch * 2,
            gradient_checkpointing=checkpointing,
            learning_rate=args.lr,
            warmup_steps=0.1,  # дробь = доля шагов (10% warmup)
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",            # чекпойнт после каждой эпохи…
            save_total_limit=1,
            load_best_model_at_end=True,      # …и в конце берём лучший по loss на val
            metric_for_best_model="eval_loss",
            logging_steps=10 if args.smoke else 50,
            bf16=gpu and torch.cuda.is_bf16_supported(),  # иначе fp32; fp16 не используем
            dataloader_num_workers=3 if gpu else 0,   # на Modal контейнеру выделено 4 ядра
            # окна похожей длины в одном батче: меньше паддинга (без этого ~30% вычислений впустую)
            train_sampling_strategy="group_by_length" if gpu else "random",
            seed=42,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out / "model"))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)   # лучшая модель уже в model/
    (out / "log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    print("сохранено в", out / "model")


if __name__ == "__main__":
    main()
