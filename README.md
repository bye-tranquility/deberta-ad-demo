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
uv run streamlit run demo/app.py
```

Модели берутся из локальной `runs/`, если она есть, иначе с Hugging Face
([`bye-tranquility/deberta-ad-tagger`](https://huggingface.co/bye-tranquility/deberta-ad-tagger)):
DeBERTa — из корня репозитория, baselines — из `baselines/<имя>.pkl`. Модели, которых нет ни там, ни там,
в списке не показываются. Порог и `min_len` по умолчанию — подобранные на val (`demo/config.json`).
Примеры: 500 текстов test (`demo/examples_test.jsonl`) и NaiAD_human (скачивается с Hugging Face).

Слева: выбор модели, порог, `min_len`, настоящая разметка, тепловая карта.
Вкладки: подсветка (совпало / лишнее / пропущено), текст без рекламы, график вероятностей по токенам.

### Деплой на Streamlit Community Cloud

1. Модель на Hugging Face (один раз): `uv run hf upload bye-tranquility/deberta-ad-tagger runs/deberta/model .`;
   baselines — `uv run hf upload bye-tranquility/deberta-ad-tagger runs/<имя>/model.pkl baselines/<имя>.pkl`.
2. Код на GitHub; зависимости демо — `demo/requirements.txt` (CPU-сборка torch). Он лежит рядом с `app.py`
   и поэтому важнее корневого `uv.lock`, в котором torch с CUDA.
3. share.streamlit.io → New app → репозиторий, ветка `main`, файл `demo/app.py`, Python 3.12.

## Данные

NaiAD — [`MaxAcand/NaiAD`](https://huggingface.co/datasets/MaxAcand/NaiAD), лицензия MIT.
`demo/examples_test.jsonl` — 500 текстов из test-части нашего сплита этого датасета.

## Метрики

- **token_f1** — насколько точно размечены отдельные токены.
- **span_f1** — найден ли блок рекламы целиком (засчитывается, если пересечение с настоящим блоком IoU ≥ 0.5).

## Что можно добавить потом (из `prompts/02_token_tagging_simple.md`)

Маркер переноса строки, вес класса в loss, baselines, проверка на текстах без рекламы,
OOD на `NaiAD_human.jsonl`, доверительные интервалы, разбивки по стратегиям.
