"""Обучение и оценка DeBERTa на GPU в Modal.

Запускает те же scripts/train.py и scripts/evaluate.py, что и локально, только на GPU.
Данные и результаты лежат на Modal Volume "adblock":
    /vol/processed   — train/val/test.jsonl (загружаются один раз, см. README)
    /vol/runs/<name> — модель, log.json, probs_*.json, metrics.json
    /vol/hf          — кэш скачанной deberta-v3-base

  modal run scripts/modal_train.py --smoke --gpu L40S              # проверка, результаты в runs/smoke-L40S
  modal run --detach scripts/modal_train.py --name deberta_b32 --batch-size 32 --lr 3e-5
  modal run --detach scripts/modal_train.py --cv
        # кросс-валидация на NaiAD_human: 5 фолдов дообучения runs/deberta/model (runs/deberta_human_fold<k>)
        # и оценка той же runs/deberta без дообучения на тех же фолдах; всё параллельно
  modal run --detach scripts/modal_train.py --cv2
        # вторая схема: нормализация чисел/ссылок, 2 повтора human, эпоха по span_f1 на val (runs/deberta_human3_fold<k>),
        # затем переобучение на train+val с найденным числом эпох (runs/deberta_human3_refit_fold<k>)
"""
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
# L40S (48 ГБ, ~$1.95/ч) или A100 (40 ГБ, ~$2.10/ч): какая выгоднее, видно по smoke-запускам.
# L4 дешевле за час, но в несколько раз медленнее, так что итоговая стоимость выше.
DEFAULT_GPU = "L40S"
VOL = "/vol"

volume = modal.Volume.from_name("adblock", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # те же версии, что в uv.lock
    .pip_install(
        "torch==2.15.0",
        "transformers==5.22.2",
        "accelerate==1.15.0",
        "sentencepiece==0.2.3",
        "protobuf==7.36.2",
        "scikit-learn==1.9.1",
        "numpy==2.5.4",
    )
    .env({
        "ADBLOCK_PROCESSED": f"{VOL}/processed",
        "ADBLOCK_RUNS": f"{VOL}/runs",
        "HF_HOME": f"{VOL}/hf",
    })
    # код проекта копируется при каждом запуске, поэтому правки подхватываются сразу
    .add_local_dir(ROOT / "src", "/root/project/src")
    .add_local_dir(ROOT / "scripts", "/root/project/scripts")
)

app = modal.App("adblock", image=image)


# timeout — страховка от зависшего запуска: дольше 3 часов GPU не оплачиваем
# cpu=4: при ~1 ядре по умолчанию загрузка данных не успевала за GPU (провалы загрузки до 14%)
@app.function(gpu=DEFAULT_GPU, cpu=4, volumes={VOL: volume}, timeout=3 * 60 * 60)
def train_and_evaluate(name: str, smoke: bool, train_args: list[str]):
    def run(*args):
        cmd = ["python", *args]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd="/root/project")

    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
    run("scripts/train.py", "--name", name, *(["--smoke"] if smoke else []), *train_args)
    volume.commit()                     # сохранить модель, даже если оценка упадёт
    run("scripts/evaluate.py", "--name", name)
    volume.commit()


@app.function(gpu=DEFAULT_GPU, cpu=4, volumes={VOL: volume}, timeout=3 * 60 * 60)
def cv_fold(fold: int, finetune: bool, lr: float, epochs: int):
    """Фолд кросс-валидации: дообучить runs/deberta на human (или взять как есть) и оценить фолд."""
    def run(*args):
        cmd = ["python", *args]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd="/root/project")

    name = f"deberta_human_fold{fold}" if finetune else "deberta"
    if finetune:
        run("scripts/train.py", "--name", name, "--init", f"{VOL}/runs/deberta/model", "--human_fold", str(fold),
            "--lr", str(lr), "--epochs", str(epochs))
        volume.commit()
    run("scripts/evaluate_cv.py", "--run", name, "--fold", str(fold))
    volume.commit()


@app.function(gpu=DEFAULT_GPU, cpu=4, volumes={VOL: volume}, timeout=3 * 60 * 60)
def cv2_fold(fold: int, lr: float, epochs: int, repeat: int, prefix: str):
    """Вторая схема дообучения на фолде: выбор эпохи по span_f1, затем переобучение на train+val."""
    import json

    def run(*args):
        cmd = ["python", *args]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd="/root/project")

    common = ["--init", f"{VOL}/runs/deberta/model", "--human_fold", str(fold), "--lr", str(lr),
              "--human_repeat", str(repeat), "--normalize"]
    select = f"{prefix}_fold{fold}"
    run("scripts/train.py", "--name", select, "--epochs", str(epochs), "--select", "span_f1", *common)
    volume.commit()
    run("scripts/evaluate_cv.py", "--run", select, "--fold", str(fold), "--normalize")
    volume.commit()
    best_epoch = json.loads(open(f"{VOL}/runs/{select}/train_config.json").read())["best_epoch"]
    refit = f"{prefix}_refit_fold{fold}"
    run("scripts/train.py", "--name", refit, "--epochs", str(best_epoch), "--human_refit", *common)
    volume.commit()
    run("scripts/evaluate_cv.py", "--run", refit, "--fold", str(fold), "--normalize")
    volume.commit()


@app.local_entrypoint()
def main(name: str = "deberta", smoke: bool = False, gpu: str = DEFAULT_GPU,
         batch_size: int = 16, lr: float = 2e-5, epochs: int = 3, cv: bool = False, cv_lr: float = 1e-5,
         cv2: bool = False, cv2_repeat: int = 2, cv2_prefix: str = "deberta_human3"):
    if cv2:
        for _ in cv2_fold.starmap([(k, cv_lr, epochs, cv2_repeat, cv2_prefix) for k in range(5)], return_exceptions=True):
            pass
        print(f"Готово: runs/{cv2_prefix}_fold<k>, runs/{cv2_prefix}_refit_fold<k>")
        return
    if cv:
        jobs = [(k, True, cv_lr, epochs) for k in range(5)] + [(k, False, cv_lr, epochs) for k in range(5)]
        for _ in cv_fold.starmap(jobs, return_exceptions=True):
            pass
        print("Готово. Забрать: modal volume get adblock 'runs/deberta*/cv_*' (см. README)")
        return
    if smoke:
        name = f"smoke-{gpu}"
    train_args = ["--batch_size", str(batch_size), "--lr", str(lr), "--epochs", str(epochs)]
    train_and_evaluate.with_options(gpu=gpu).remote(name, smoke, train_args)
    print(f"\nГотово. Забрать результаты (без модели):\n"
          f"  modal volume get adblock runs/{name}/metrics.json runs/{name}/\n"
          f"  modal volume get adblock runs/{name}/log.json runs/{name}/")
