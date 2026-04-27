"""
src/evaluation/metrics.py
=========================
Unified evaluation logic so every model in the comparison is judged identically.

Three primary functions:
  - `evaluate(...)`            — standard classification metrics + top-K F1
  - `confidence_fallback(...)` — apply confidence threshold for hierarchical fallback
  - `hierarchical_evaluate(...)` — compute F1 at multiple hierarchy levels
"""
from __future__ import annotations

from collections import Counter
from typing      import Callable, Optional

import numpy  as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────
# 1. Headline evaluation
# ─────────────────────────────────────────────────────────────────────────
def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k:      int = 10,
) -> dict:
    """Compute the standard classification metric bundle.

    Returns a dict with:
      - accuracy, macro_f1, weighted_f1
      - top_k_macro_f1: macro-F1 restricted to the top-K most frequent classes
                       (the project's success criterion #2)
      - top_k_classes: the actual top-K class labels (for the report)
      - n_classes_observed: how many distinct classes are in y_true
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1    = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy    = accuracy_score(y_true, y_pred)

    # Top-K macro F1: restrict to top-K classes by frequency in y_true
    class_counts  = Counter(y_true.tolist())
    top_k_classes = [c for c, _ in class_counts.most_common(k)]

    # Compute F1 only on rows where y_true is in top_k_classes
    mask = np.isin(y_true, top_k_classes)
    if mask.sum() > 0:
        top_k_macro_f1 = f1_score(
            y_true[mask], y_pred[mask],
            labels=top_k_classes, average='macro', zero_division=0,
        )
    else:
        top_k_macro_f1 = 0.0

    return {
        'accuracy':           float(accuracy),
        'macro_f1':           float(macro_f1),
        'weighted_f1':        float(weighted_f1),
        f'top_{k}_macro_f1':  float(top_k_macro_f1),
        f'top_{k}_classes':   top_k_classes,
        'n_classes_observed': int(len(class_counts)),
    }


def per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a DataFrame with per-class precision/recall/F1/support.

    Useful for confusion analysis and finding which classes the model fails on.
    """
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for cls, m in rep.items():
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        rows.append({
            'class':     cls,
            'precision': m['precision'],
            'recall':    m['recall'],
            'f1':        m['f1-score'],
            'support':   m['support'],
        })
    return pd.DataFrame(rows).sort_values('support', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# 2. Confidence-thresholded hierarchical fallback
# ─────────────────────────────────────────────────────────────────────────
def confidence_fallback(
    y_proba:    np.ndarray,
    classes:    np.ndarray,
    threshold:  float,
    rollup_fn:  Callable[[str], str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply confidence-thresholded hierarchical fallback.

    For each row:
      - If max(p) >= threshold:  output the model's top class (full granularity)
      - Else:                    output rollup_fn(top_class) (parent level)

    This mirrors how the model would be deployed in production.

    Parameters
    ----------
    y_proba : (n_samples, n_classes) probability matrix
    classes : array of class labels in the order matching y_proba columns
    threshold : float in [0, 1]
    rollup_fn : maps a fine-grained class label to its parent label
                (e.g. SubIndustry → parent_industry_code → str(c)[:8])

    Returns
    -------
    predictions   : (n_samples,) — fine-grained where confident, rolled up otherwise
    top_prob      : (n_samples,) — the model's max probability per row
    is_confident  : (n_samples,) bool — True if prediction was confident
    """
    classes  = np.asarray(classes)
    top_idx  = y_proba.argmax(axis=1)
    top_prob = y_proba.max(axis=1)
    top_pred = classes[top_idx]

    is_confident = top_prob >= threshold
    fallback     = np.array([rollup_fn(c) for c in top_pred])
    predictions  = np.where(is_confident, top_pred, fallback)

    return predictions, top_prob, is_confident


def evaluate_with_fallback(
    y_true:           np.ndarray,
    y_proba:          np.ndarray,
    classes:          np.ndarray,
    threshold:        float,
    rollup_fn:        Callable[[str], str],
    rollup_y_true_fn: Optional[Callable[[str], str]] = None,
    k:                int = 10,
) -> dict:
    """End-to-end evaluation with confidence fallback applied.

    Computes:
      - Coverage: % of rows where the model was confident enough for fine-grained
      - F1 metrics on the *resulting* predictions (mix of fine + rolled-up)
      - F1 metrics on the *confident subset only* at fine granularity

    Parameters
    ----------
    rollup_y_true_fn : optional. If provided, applied to y_true so the rolled-up
                       predictions can be compared against rolled-up truth.
                       Defaults to rollup_fn.
    """
    if rollup_y_true_fn is None:
        rollup_y_true_fn = rollup_fn

    y_pred, top_prob, is_confident = confidence_fallback(
        y_proba, classes, threshold, rollup_fn,
    )

    # Build "rolled-up y_true" — for confident rows keep original; otherwise roll up
    y_true_eval = np.array([
        t if c else rollup_y_true_fn(t)
        for t, c in zip(y_true, is_confident)
    ])

    # Headline metrics on the merged predictions
    overall = evaluate(y_true_eval, y_pred, k=k)

    # Metrics on the confident subset (pure fine-grained performance)
    if is_confident.sum() > 0:
        confident_only = evaluate(y_true[is_confident], y_pred[is_confident], k=k)
    else:
        confident_only = {'accuracy': 0.0, 'macro_f1': 0.0, 'weighted_f1': 0.0,
                          f'top_{k}_macro_f1': 0.0, 'n_classes_observed': 0}

    return {
        'threshold':              threshold,
        'coverage':               float(is_confident.mean()),
        'n_confident':            int(is_confident.sum()),
        'n_total':                int(len(y_true)),
        'overall_macro_f1':       overall['macro_f1'],
        'overall_top_k_macro_f1': overall[f'top_{k}_macro_f1'],
        'confident_macro_f1':     confident_only['macro_f1'],
        'confident_top_k_macro_f1': confident_only.get(f'top_{k}_macro_f1', 0.0),
    }


# ─────────────────────────────────────────────────────────────────────────
# 3. Hierarchical evaluation — F1 at every level of the GECS hierarchy
# ─────────────────────────────────────────────────────────────────────────
def hierarchical_evaluate(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    rollups:   dict[str, Callable[[str], str]],
) -> pd.DataFrame:
    """Compute macro-F1 at multiple hierarchy levels by rolling up predictions.

    Parameters
    ----------
    rollups : dict mapping level name → function that converts a label to its
              parent-level code, e.g.
              {
                  'subindustry':     lambda x: str(x),
                  'parent_industry': lambda x: str(x)[:8],
                  'industry_group':  lambda x: str(x)[:5],
                  'sector':          lambda x: str(x)[:3],
                  'super_sector':    lambda x: str(x)[:1],
              }

    Returns
    -------
    DataFrame with one row per level: macro_f1, weighted_f1, accuracy, n_classes.
    """
    rows = []
    for level, fn in rollups.items():
        yt = np.array([fn(t) for t in y_true])
        yp = np.array([fn(p) for p in y_pred])
        rows.append({
            'level':       level,
            'accuracy':    accuracy_score(yt, yp),
            'macro_f1':    f1_score(yt, yp, average='macro',    zero_division=0),
            'weighted_f1': f1_score(yt, yp, average='weighted', zero_division=0),
            'n_classes':   len(set(yt.tolist())),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# 4. Convenience hierarchy rollup factories for the GECS taxonomy
# ─────────────────────────────────────────────────────────────────────────
def gecs_rollups_task1() -> dict[str, Callable[[str], str]]:
    """Hierarchy rollups for 8-digit MstarGlobal codes (Task 1)."""
    return {
        'industry':     lambda x: str(x),         # 8 digits — full label
        'industry_group': lambda x: str(x)[:5],   # 5 digits
        'sector':       lambda x: str(x)[:3],     # 3 digits
        'super_sector': lambda x: str(x)[:1],     # 1 digit
    }


def gecs_rollups_task2() -> dict[str, Callable[[str], str]]:
    """Hierarchy rollups for 10-digit SubIndustry codes (Task 2)."""
    return {
        'subindustry':     lambda x: str(x),       # 10 digits — full label
        'parent_industry': lambda x: str(x)[:8],   # 8 digits
        'industry_group':  lambda x: str(x)[:5],   # 5 digits
        'sector':          lambda x: str(x)[:3],   # 3 digits
        'super_sector':    lambda x: str(x)[:1],   # 1 digit
    }
