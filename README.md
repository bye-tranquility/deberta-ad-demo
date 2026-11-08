# AdBlock: поиск рекламы в ответах LLM

Модель (DeBERTa) смотрит на каждый токен ответа и говорит: реклама (1) или нет (0).
Данные — NaiAD, в ответах реклама размечена тегами `<ad>…</ad>`.

## Как устроено (по шагам)

| Шаг | Файл | Что делает |
|---|---|---|
| 1 | `src/adblock/data.py` | убирает теги `<ad>`, запоминает, где была реклама (позиции символов) |
| 2 | `src/adblock/split.py` | делит на train/val/test **по брендам**, чтобы модель не запомнила бренды |
| 3 | `src/adblock/tokenize.py` | режет текст на токены, ставит метку 0/1, нарезает окна по 512 токенов |
| 4 | `scripts/train.py` | обучает модель |
| 5 | `scripts/evaluate.py`, `src/adblock/metrics.py` | подбирает порог на val, считает F1 на test |

## Запуск

```bash
uv add torch transformers sentencepiece protobuf scikit-learn accelerate
uv add --dev pytest
uv run pytest
uv run python -m adblock.split
uv run python scripts/train.py --smoke
uv run python scripts/evaluate.py --name smoke
```

Если smoke отработал и loss падает — `scripts/train.py` без `--smoke`.

## Обучение на GPU (Modal)

Один раз:

```bash
uv add --dev modal
uv run modal token new
uv run modal volume create adblock
uv run modal volume put adblock data/processed processed
```

Проверка окружения (500 текстов, ~5 минут):

```bash
uv run modal run scripts/modal_train.py --smoke
```

Полное обучение (`--detach`: можно закрыть терминал, прогресс на modal.com/apps):

```bash
uv run modal run --detach scripts/modal_train.py --name deberta
```

Забрать результаты:

```bash
uv run modal volume get adblock runs/deberta/metrics.json runs/deberta/
```

Модель (~700 МБ) лежит в `runs/<name>/model` на volume; скачивать её не обязательно.
Если `data/processed` пересобирался, загрузите заново с `--force`.

## Демо (Streamlit)

```bash
uv run modal volume get adblock runs/deberta/model runs/deberta/   # DeBERTa, ~700 МБ, один раз
uv run python scripts/baselines.py                                  # baselines -> runs/baseline_*/model.pkl
uv run streamlit run app.py
```

Слева выбор модели (DeBERTa и все baselines), порог и `min_len` (по умолчанию значения, подобранные на val),
настоящая разметка и тепловая карта. Текст: случайный пример из test, из `NaiAD_human` или свой.
Вкладки: подсветка (совпало / лишнее / пропущено), текст без рекламы, график вероятностей по токенам.
Модели, которых нет в `runs/`, в списке не показываются.

## Метрики

- **token_f1** — насколько точно размечены отдельные токены.
- **span_f1** — найден ли блок рекламы целиком (засчитывается, если пересечение с настоящим блоком IoU ≥ 0.5).

## Что можно добавить потом (из `prompts/02_token_tagging_simple.md`)

Маркер переноса строки, вес класса в loss, baselines, проверка на текстах без рекламы,
OOD на `NaiAD_human.jsonl`, доверительные интервалы, разбивки по стратегиям.
