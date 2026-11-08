"""Скачивает MaxAcand/NaiAD в ./data/naiad (рядом с проектом).

Запуск (на вашей машине, не в песочнице):
    pip install datasets huggingface_hub
    python scripts/download_naiad.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(__file__).resolve().parent.parent / "data" / "naiad"
target.mkdir(parents=True, exist_ok=True)

path = snapshot_download(
    repo_id="MaxAcand/NaiAD",
    repo_type="dataset",
    local_dir=target,
)
print("Скачано в:", path)

# Быстрый обзор: подвыборки, сплиты, колонки
from datasets import get_dataset_config_names, load_dataset

for cfg in get_dataset_config_names("MaxAcand/NaiAD"):
    ds = load_dataset("MaxAcand/NaiAD", cfg)
    for split, d in ds.items():
        print(f"{cfg}/{split}: {len(d)} строк, колонки: {d.column_names}")
