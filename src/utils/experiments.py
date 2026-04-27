"""
src/utils/experiments.py
========================
Append-only experiment log so every model run gets recorded with the same
schema. Read this CSV at the end of the project to build the comparison
tables for the report.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib  import Path
from typing   import Any


# Canonical column order — every notebook writes these fields
EXPERIMENT_COLUMNS = [
    'timestamp',
    'task',
    'model_name',
    'features',
    'split_strategy',
    'n_train',
    'n_val',
    'n_test',
    'n_classes_modelable',
    'n_classes_excluded',
    'accuracy',
    'macro_f1',
    'weighted_f1',
    'top_10_macro_f1',
    'train_time_s',
    'inference_ms_per_row',
    'model_size_mb',
    'notes',
]


def log_experiment(results_path: str | Path, **kwargs: Any) -> None:
    """Append one row to the experiments CSV. Creates header if file is new.

    Any keys not in EXPERIMENT_COLUMNS are silently dropped (so notebooks can
    pass dicts directly without filtering). Missing keys default to ''.
    """
    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    kwargs.setdefault('timestamp', datetime.utcnow().isoformat(timespec='seconds'))

    # Coerce list fields to strings so csv.writer handles them
    row = {}
    for col in EXPERIMENT_COLUMNS:
        v = kwargs.get(col, '')
        if isinstance(v, (list, tuple)):
            v = ';'.join(str(x) for x in v)
        elif isinstance(v, float):
            v = round(v, 6)
        row[col] = v

    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def load_experiments(results_path: str | Path):
    """Load the experiment log as a DataFrame. Returns empty DF if file missing."""
    import pandas as pd  # local import keeps module lightweight
    path = Path(results_path)
    if not path.exists():
        return pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    return pd.read_csv(path)
