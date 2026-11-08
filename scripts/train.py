"""Шаг 4. Обучение DeBERTa размечать токены рекламы.

  uv run python scripts/train.py --smoke     # быстрая проверка: 500 текстов, 1 эпоха
  uv run python scripts/train.py             # полное обучение
  uv run python scripts/train.py --name deberta_human_fold0 --init runs/deberta/model --human_fold 0 --lr 1e-5
        # дообучение готовой модели на фолде NaiAD_human (+ часть синтетики, чтобы её не забыть)
Модель сохраняется в runs/<name>/model, история loss — в runs/<name>/log.json.
Если runs/<name>/model уже есть, скрипт останавливается (перезапись — только с --overwrite).
На GPU (Modal) запускается через scripts/modal_train.py.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adblock.data import PROCESSED, RUNS, normalize_row, read_jsonl  # noqa: E402
from adblock.human import load_cv  # noqa: E402
from adblock.metrics import best_span_f1  # noqa: E402
from adblock.tokenize import MODEL, label_tokens, make_windows, merge_windows  # noqa: E402


def build_dataset(tokenizer, rows):
    windows = []
    for r in rows:
        ids, _, labels = label_tokens(tokenizer, r["text"], r["ad_spans"])
        for w in make_windows(tokenizer, ids, labels):
            windows.append({"input_ids": w["input_ids"], "labels": w["labels"]})
    return windows


def build_eval(tokenizer, rows):
    """Окна val + привязка к документам, чтобы после предсказания собрать вероятность на каждый токен."""
    windows, meta, labels = [], [], []
    for d, r in enumerate(rows):
        ids, _, y = label_tokens(tokenizer, r["text"], r["ad_spans"])
        labels.append(y)
        for w in make_windows(tokenizer, ids, y):
            windows.append({"input_ids": w["input_ids"], "labels": w["labels"]})
            meta.append((d, w["start"], len(w["input_ids"]) - 2))
    return windows, meta, labels


def span_f1_metric(meta, labels):
    """compute_metrics для Trainer: лучший span_f1@0.5 на val по сетке порогов и min_len."""
    def compute(ep):
        preds = ep.predictions
        docs = {}
        for i, (d, start, n) in enumerate(meta):
            docs.setdefault(d, []).append(({"start": start}, preds[i, 1:1 + n].tolist()))
        probs = [merge_windows(len(labels[d]), [w for w, _ in docs[d]], [p for _, p in docs[d]])
                 for d in range(len(labels))]
        return {"span_f1": best_span_f1(probs, labels)[0]}
    return compute


class OrderedEvalTrainer(Trainer):
    """Оценка всегда по порядку окон: span_f1 склеивает окна в документы по их позиции.

    В transformers 5 при train_sampling_strategy="group_by_length" Trainer перемешивает и val,
    и тогда предсказания окон попадают не в свои документы.
    """

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset) if eval_dataset is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16, help="эффективный батч (на Mac набирается накоплением по 4)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--init", default=MODEL, help="с каких весов начинать: имя на HF или путь к сохранённой модели")
    ap.add_argument("--human_fold", type=int, default=None, help="фолд NaiAD_human: train/val из human_cv.jsonl")
    ap.add_argument("--synth_n", type=int, default=4000, help="сколько синтетических текстов train подмешать к human")
    ap.add_argument("--human_repeat", type=int, default=4, help="сколько раз повторить human-тексты в train")
    ap.add_argument("--overwrite", action="store_true", help="разрешить перезаписать runs/<name>/model")
    ap.add_argument("--normalize", action="store_true", help="числа и ссылки -> NUMBER_TOKEN / URL_TOKEN, как в human")
    ap.add_argument("--select", choices=["loss", "span_f1"], default="loss", help="по чему выбирать лучшую эпоху на val")
    ap.add_argument("--human_refit", action="store_true",
                    help="учить на train+val фолдах human ровно --epochs эпох, без val и выбора эпохи")
    args = ap.parse_args()
    args.name = args.name or ("smoke" if args.smoke else "deberta")
    out = RUNS / args.name
    if (out / "model").exists() and not args.overwrite:
        sys.exit(f"{out / 'model'} уже существует — выберите другое --name или добавьте --overwrite")

    epochs = args.epochs
    if args.human_fold is not None:
        # human (3 фолда, повторённые human_repeat раз) + случайная часть синтетики; val — отдельный фолд human
        synth = random.Random(42).sample(read_jsonl(PROCESSED / "train.jsonl"), args.synth_n)
        human_train = load_cv(args.human_fold, "trainval" if args.human_refit else "train")
        train = synth + human_train * args.human_repeat
        val = [] if args.human_refit else load_cv(args.human_fold, "val")
        print(f"фолд {args.human_fold}: human train {len(human_train)} x{args.human_repeat}, синтетика {len(synth)}, "
              f"human val {len(val)}")
    else:
        train = read_jsonl(PROCESSED / "train.jsonl")
        val = read_jsonl(PROCESSED / "val.jsonl")
    if args.smoke:
        train, val, epochs = train[:500], val[:200], 1
    if args.normalize:
        train, val = [normalize_row(r) for r in train], [normalize_row(r) for r in val]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    train_ds = build_dataset(tokenizer, train)
    val_ds, val_meta, val_labels = build_eval(tokenizer, val)
    print(f"окон: train={len(train_ds)}, val={len(val_ds)}")

    # dtype=float32 обязателен: без него transformers 5 грузит веса DeBERTa в fp16,
    # и AdamW после первого же шага превращает их в nan
    model = AutoModelForTokenClassification.from_pretrained(
        args.init, num_labels=2, id2label={0: "O", 1: "AD"}, label2id={"O": 0, "AD": 1}, dtype=torch.float32)
    print("стартовые веса:", args.init)
    gpu = torch.cuda.is_available()
    if gpu:   # CUDA (Modal): весь батч сразу помещается в память
        batch, accum, checkpointing = args.batch_size, 1, False
    else:     # Mac: батч набирается накоплением по 4, gradient checkpointing экономит память
        batch, accum, checkpointing = 4, max(1, args.batch_size // 4), True
    print(f"gpu={gpu} batch={batch}x{accum} lr={args.lr} epochs={epochs} gradient_checkpointing={checkpointing}")

    trainer = OrderedEvalTrainer(
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
            eval_strategy="no" if not val else "epoch",
            save_strategy="no" if not val else "epoch",   # чекпойнт после каждой эпохи…
            save_total_limit=1,
            load_best_model_at_end=bool(val),             # …и в конце берём лучший на val
            metric_for_best_model="eval_span_f1" if args.select == "span_f1" else "eval_loss",
            greater_is_better=args.select == "span_f1",
            logging_steps=10 if args.smoke or args.human_fold is not None else 50,
            bf16=gpu and torch.cuda.is_bf16_supported(),  # иначе fp32; fp16 не используем
            dataloader_num_workers=3 if gpu else 0,   # на Modal контейнеру выделено 4 ядра
            # окна похожей длины в одном батче: меньше паддинга (без этого ~30% вычислений впустую)
            train_sampling_strategy="group_by_length" if gpu else "random",
            seed=42,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=val_ds or None,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=span_f1_metric(val_meta, val_labels) if args.select == "span_f1" and val else None,
        preprocess_logits_for_metrics=(lambda logits, labels: logits.float().softmax(-1)[..., 1])
        if args.select == "span_f1" and val else None,
    )
    trainer.train()
    trainer.save_model(str(out / "model"))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)   # лучшая модель уже в model/
    (out / "log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    evals = [e for e in trainer.state.log_history if "eval_loss" in e]
    if evals:
        key = "eval_span_f1" if args.select == "span_f1" else "eval_loss"
        best = (max if args.select == "span_f1" else min)(evals, key=lambda e: e[key])
        args.best_epoch = round(best["epoch"])
        print("эпохи на val:", [(round(e["epoch"]), round(e[key], 4)) for e in evals], "-> лучшая", args.best_epoch)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=1))
    print("сохранено в", out / "model")


if __name__ == "__main__":
    main()
