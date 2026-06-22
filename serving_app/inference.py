"""
GECS Industry & Sub-Industry Inference Pipeline
================================================

Task 1: FLANG-BERT champion (Meet's v10) for 145 GECS industries
Task 2: DeBERTa-v3-small cross-encoder (Anas) for 374+ sub-industries

Architecture:
    Input segment → Task1FlangBertRanker → top-K industries with scores
                  → Task2SubindustryRanker (taxonomy expansion + cross-encoder)
                  → top-K sub-industries with combined scores

Weights expected at:
    weights/task1/best_model_state.pt              (FLANG-BERT champion, 418 MB)
    weights/task1/label_encoder.pkl                (145 industry codes)
    weights/task1/label_encoder_sector.pkl         (11 sectors)
    weights/task1/label_encoder_group.pkl          (55 groups)
    weights/task1/numeric_scaler.pkl               (StandardScaler for 6 continuous feats)
    weights/task1/industry_lookup.csv              (code → name)
    weights/task2/cross_encoder/best_state.pt      (DeBERTa-v3-small, 539 MB)
    weights/task2/taxonomy_table.csv               (industry → sub-industry mapping)
"""

from __future__ import annotations

import html
import json
import os
import pickle
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import log_softmax, softmax
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TASK1_MODEL_NAME = "SALT-NLP/FLANG-BERT"
TASK2_CROSS_ENCODER_MODEL = "microsoft/deberta-v3-small"

TASK1_MAX_LEN = 512
TASK2_MAX_LEN = 256

# Numeric features used by Task 1 model (10 total, in exact training order)
NUMERIC_COLS = [
    "revenue_share", "is_largest_bin", "log_revenue",
    "log_total_revenue", "n_segments", "herfindahl_index",
    "report_quarter", "lp_short_flag", "sd_short_flag", "sn_short_flag",
]
CONTINUOUS_COLS = [
    "revenue_share", "log_revenue", "log_total_revenue",
    "n_segments", "herfindahl_index", "report_quarter",
]

# Routing thresholds
LOW_CONFIDENCE_THRESHOLD = 0.35
PARENT_TOPK_FOR_SUBINDUSTRY = 10

# Text normalization
SMART_TRANS = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\xa0": " ", "\ufeff": "",
})
WS_RE = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WS_RE.sub(" ", html.unescape(str(value)).translate(SMART_TRANS)).strip()


def clean_code(value: Any, width: int | None = None) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.sub(r"\D+", "", text)
    if width and digits:
        return digits.zfill(width)
    return digits


def first_words(value: Any, n: int = 60) -> str:
    return " ".join(str(value).split()[:n])


def torch_load_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

class FLANGMultiTask(nn.Module):
    """Task 1 champion architecture (matches training notebook).

    Multi-task heads: leaf (industry), sector, group.
    Concatenates 10 numeric features to BERT pooled output before leaf head.
    """

    def __init__(self, model_name: str, n_leaf: int, n_sector: int, n_group: int,
                 num_features: int = 10, dropout: float = 0.1, local_files_only: bool = True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leaf_head = nn.Linear(h + num_features, n_leaf)
        self.sector_head = nn.Linear(h, n_sector)
        self.group_head = nn.Linear(h, n_group)

    def forward(self, input_ids, attention_mask, numeric_feats):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = self.dropout(pooled)
        combined = torch.cat([pooled, numeric_feats.to(pooled.dtype)], dim=-1)
        return self.leaf_head(combined), self.sector_head(pooled), self.group_head(pooled)


class SegmentTaxonomyCrossEncoder(nn.Module):
    """Task 2 architecture — DeBERTa-v3-small with binary classification head."""

    def __init__(self, model_name: str, local_files_only: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = masked_mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        pooled = pooled.to(self.classifier.weight.dtype)
        return self.classifier(pooled).squeeze(-1)


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: AutoTokenizer, max_length: int):
        self.texts = list(map(str, texts))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Path discovery
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelPaths:
    package_root: Path
    weights_root: Path
    task1_state: Path
    task1_label_encoder: Path
    task1_label_encoder_sector: Path
    task1_label_encoder_group: Path
    task1_numeric_scaler: Path
    task1_industry_lookup: Path
    task2_cross_encoder_state: Path
    task2_taxonomy_table: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ModelPaths":
        """Walk up from this file until we find the weights/ directory."""
        start = (start or Path(__file__).resolve()).resolve()
        candidates = [
            start, *start.parents,
            Path.cwd().resolve(),
        ]
        for root in candidates:
            weights = root / "weights"
            if (weights / "task1" / "best_model_state.pt").exists() and \
               (weights / "task2" / "cross_encoder" / "best_state.pt").exists():
                t1 = weights / "task1"
                t2 = weights / "task2"
                return cls(
                    package_root=root,
                    weights_root=weights,
                    task1_state=t1 / "best_model_state.pt",
                    task1_label_encoder=t1 / "label_encoder.pkl",
                    task1_label_encoder_sector=t1 / "label_encoder_sector.pkl",
                    task1_label_encoder_group=t1 / "label_encoder_group.pkl",
                    task1_numeric_scaler=t1 / "numeric_scaler.pkl",
                    task1_industry_lookup=t1 / "industry_lookup.csv",
                    task2_cross_encoder_state=t2 / "cross_encoder" / "best_state.pt",
                    task2_taxonomy_table=t2 / "taxonomy_table.csv",
                )
        raise FileNotFoundError(
            "Could not locate weights/. Expected layout:\n"
            "  weights/task1/best_model_state.pt\n"
            "  weights/task2/cross_encoder/best_state.pt\n"
            "  weights/task2/taxonomy_table.csv\n"
            "Run: python scripts/download_weights.py"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — converts raw input to the 10 numeric features
# ─────────────────────────────────────────────────────────────────────────────

def compute_numeric_features(frame: pd.DataFrame, scaler) -> np.ndarray:
    """Compute the 10 numeric features used by the FLANG-BERT model.

    Required input columns (with sensible defaults if missing):
      - Revenue, total_revenue_company_as_of, revenue_share
      - is_largest_share_segment
      - SegmentName, SegmentDescription, LongProfile
      - AsOfDate (for report_quarter)

    Groups by CompanyId+AsOfDate to compute n_segments and herfindahl when
    multiple rows are provided. Falls back gracefully on single-row input.
    """
    df = frame.copy()

    def to_float(s):
        try:
            return float(s)
        except (TypeError, ValueError):
            return 0.0

    df["Revenue"] = df.get("Revenue", 0).apply(to_float)
    df["total_revenue_company_as_of"] = df.get("total_revenue_company_as_of", 0).apply(to_float)
    df["revenue_share"] = df.get("revenue_share", 0).apply(to_float)

    # log_revenue, log_total_revenue (safe log, zero floor)
    df["log_revenue"] = np.log1p(df["Revenue"].clip(lower=0))
    df["log_total_revenue"] = np.log1p(df["total_revenue_company_as_of"].clip(lower=0))

    # is_largest_bin: 1 if revenue_share is the max within company-date group
    def _largest_flag(val):
        if isinstance(val, str):
            return 1 if val.lower() in ("true", "1", "yes") else 0
        return int(bool(val))

    if "is_largest_share_segment" in df.columns:
        df["is_largest_bin"] = df["is_largest_share_segment"].apply(_largest_flag)
    else:
        df["is_largest_bin"] = 0

    # n_segments & herfindahl_index — compute per company-date group when available
    if "CompanyId" in df.columns and "AsOfDate" in df.columns:
        group_cols = ["CompanyId", "AsOfDate"]
        df["n_segments"] = df.groupby(group_cols)["SegmentName"].transform("count").astype(float)
        df["herfindahl_index"] = df.groupby(group_cols)["revenue_share"].transform(
            lambda x: float((x ** 2).sum())
        )
    else:
        df["n_segments"] = 1.0
        df["herfindahl_index"] = (df["revenue_share"] ** 2).fillna(0.0)

    # report_quarter: extract quarter (1-4) from AsOfDate
    def _quarter(d):
        try:
            month = int(str(d)[5:7])
            return (month - 1) // 3 + 1
        except (ValueError, IndexError):
            return 0

    df["report_quarter"] = df.get("AsOfDate", "").apply(_quarter).astype(float)

    # Short flags: 1 if text is short/missing
    def _short(text, threshold=10):
        if pd.isna(text):
            return 1
        return 1 if len(str(text).split()) < threshold else 0

    df["lp_short_flag"] = df.get("LongProfile", "").apply(lambda t: _short(t, 20))
    df["sd_short_flag"] = df.get("SegmentDescription", "").apply(lambda t: _short(t, 8))
    df["sn_short_flag"] = df.get("SegmentName", "").apply(lambda t: _short(t, 3))

    # Fill any remaining NaNs
    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)

    # Apply scaler ONLY to continuous columns (matches training)
    feats = df[NUMERIC_COLS].copy()
    feats[CONTINUOUS_COLS] = scaler.transform(feats[CONTINUOUS_COLS])
    return feats[NUMERIC_COLS].values.astype(np.float32)


def build_task1_text(frame: pd.DataFrame, sibling_words: int = 25, lp_words: int = 100) -> list[str]:
    """Build the text input for FLANG-BERT, matching training format.

    Format:
        [] [] [year] [PRIMARY] {seg_name} [SEP] {seg_desc}
        [SEG {rev_pct}%] {sibling_desc} ...
        [LP] {long_profile_short}

    Note: sector/group hint tokens are left empty at inference since we don't
    yet know the answer. The year token is preserved.
    """
    df = frame.copy()
    df["_seg_name"] = df.get("SegmentName", "").fillna("").astype(str).str.strip()
    df["_seg_desc"] = df.get("SegmentDescription", "").fillna("").astype(str).str.strip()
    df["_long_p"] = df.get("LongProfile", "").fillna("").astype(str).str.strip()
    df["_year"] = df.get("AsOfDate", "").fillna("").astype(str).str[:4]

    # Sibling lookup
    has_groups = "CompanyId" in df.columns and "AsOfDate" in df.columns
    sibling_lookup: dict[tuple, list[int]] = {}
    if has_groups:
        for key, sub in df.groupby(["CompanyId", "AsOfDate"]):
            sibling_lookup[key] = sub.index.tolist()

    texts = []
    for idx in df.index:
        row = df.loc[idx]
        seg_name = row["_seg_name"]
        seg_desc = row["_seg_desc"] if row["_seg_desc"] else seg_name
        long_p = row["_long_p"]
        year = row["_year"]

        # Sibling context
        sib_parts = []
        if has_groups:
            key = (row["CompanyId"], row["AsOfDate"])
            for sib_idx in sibling_lookup.get(key, []):
                if sib_idx == idx:
                    continue
                sib = df.loc[sib_idx]
                rev_pct = round(float(sib.get("revenue_share", 0) or 0) * 100, 1)
                sib_desc = sib["_seg_desc"]
                sib_short = " ".join(sib_desc.split()[:sibling_words])
                if sib_short:
                    sib_parts.append(f"[SEG {rev_pct}%] {sib_short}")

        # LongProfile (truncated)
        lp_short = " ".join(long_p.split()[:lp_words]) if long_p else ""

        parts = [
            "[]", "[]", f"[{year}]", "[PRIMARY]",
            seg_name, "[SEP]", seg_desc,
        ]
        if sib_parts:
            parts.extend(sib_parts)
        if lp_short:
            parts.append(f"[LP] {lp_short}")

        texts.append(" ".join(parts))

    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — FLANG-BERT industry ranker
# ─────────────────────────────────────────────────────────────────────────────

class Task1FlangBertRanker:
    """Loads the FLANG-BERT v10 champion and ranks 145 GECS industries."""

    def __init__(self, paths: ModelPaths, device: torch.device, local_files_only: bool):
        self.paths = paths
        self.device = device
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self._loaded = False

        self.label_encoder = None
        self.sector_encoder = None
        self.group_encoder = None
        self.numeric_scaler = None
        self.industry_codes: np.ndarray | None = None
        self.tokenizer = None
        self.model: FLANGMultiTask | None = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            with open(self.paths.task1_label_encoder, "rb") as f:
                self.label_encoder = pickle.load(f)
            with open(self.paths.task1_label_encoder_sector, "rb") as f:
                self.sector_encoder = pickle.load(f)
            with open(self.paths.task1_label_encoder_group, "rb") as f:
                self.group_encoder = pickle.load(f)
            with open(self.paths.task1_numeric_scaler, "rb") as f:
                self.numeric_scaler = pickle.load(f)

            # Industry codes in label_encoder.classes_ order — these are what model heads predict
            raw_classes = np.asarray(self.label_encoder.classes_).astype(str)
            self.industry_codes = np.array([clean_code(c, width=8) for c in raw_classes])

            n_leaf = len(self.label_encoder.classes_)
            n_sec = len(self.sector_encoder.classes_)
            n_grp = len(self.group_encoder.classes_)

            self.tokenizer = AutoTokenizer.from_pretrained(
                TASK1_MODEL_NAME, local_files_only=self.local_files_only,
            )
            model = FLANGMultiTask(
                TASK1_MODEL_NAME, n_leaf, n_sec, n_grp,
                num_features=len(NUMERIC_COLS), dropout=0.1,
                local_files_only=self.local_files_only,
            ).to(self.device)
            state = torch_load_state(self.paths.task1_state, self.device)
            model.load_state_dict(state)
            model.eval()
            self.model = model
            self._loaded = True

    @torch.no_grad()
    def log_scores(self, frame: pd.DataFrame) -> np.ndarray:
        self.ensure_loaded()
        assert self.model is not None and self.tokenizer is not None

        texts = build_task1_text(frame)
        numeric = compute_numeric_features(frame, self.numeric_scaler)

        ds = TextDataset(texts, self.tokenizer, TASK1_MAX_LEN)
        batch_size = 16 if self.device.type == "cuda" else 4
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        numeric_t = torch.from_numpy(numeric)
        all_logits = []
        cursor = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            bs = batch["input_ids"].shape[0]
            num_batch = numeric_t[cursor:cursor + bs].to(self.device)
            cursor += bs
            leaf_logits, _, _ = self.model(
                batch["input_ids"], batch["attention_mask"], num_batch,
            )
            all_logits.append(leaf_logits.float().cpu().numpy())
        logits = np.vstack(all_logits)
        return log_softmax(logits, axis=1)

    def top_industries(self, frame: pd.DataFrame, top_k: int = 5) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
        log_p = self.log_scores(frame)
        probs = softmax(log_p, axis=1)
        assert self.industry_codes is not None
        order = np.argsort(-probs, axis=1)
        rows: list[list[dict[str, Any]]] = []
        for i in range(len(frame)):
            row = []
            for rank, class_idx in enumerate(order[i, :top_k], start=1):
                row.append({
                    "rank": rank,
                    "industry_code": str(self.industry_codes[class_idx]),
                    "score": float(probs[i, class_idx]),
                })
            rows.append(row)
        return rows, probs


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Sub-industry cross-encoder reranker (from Anas's pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class Task2SubindustryRanker:
    def __init__(self, paths: ModelPaths, device: torch.device, local_files_only: bool):
        self.paths = paths
        self.device = device
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self._loaded = False
        self.tokenizer = None
        self.model: SegmentTaxonomyCrossEncoder | None = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self.tokenizer = AutoTokenizer.from_pretrained(
                TASK2_CROSS_ENCODER_MODEL, local_files_only=self.local_files_only,
            )
            model = SegmentTaxonomyCrossEncoder(
                TASK2_CROSS_ENCODER_MODEL, self.local_files_only,
            ).to(self.device)
            model.load_state_dict(torch_load_state(self.paths.task2_cross_encoder_state, self.device))
            model.eval()
            self.model = model
            self._loaded = True

    @staticmethod
    def build_pair_text(segment_text: str, taxonomy_text: str) -> str:
        return (
            f"[SEGMENT] {normalize_text(segment_text)} "
            f"[CANDIDATE_TAXONOMY] {normalize_text(taxonomy_text)} "
            "[QUESTION] Does this segment belong to this subindustry?"
        ).strip()

    @torch.no_grad()
    def score_pairs(self, pair_texts: list[str]) -> np.ndarray:
        self.ensure_loaded()
        assert self.tokenizer is not None and self.model is not None
        ds = TextDataset(pair_texts, self.tokenizer, TASK2_MAX_LEN)
        batch_size = int(os.environ.get(
            "SERVING_TASK2_BATCH_SIZE", "16" if self.device.type == "cuda" else "4",
        ))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        chunks = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch["input_ids"], batch["attention_mask"])
            chunks.append(out.float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level prediction service
# ─────────────────────────────────────────────────────────────────────────────

class PredictionService:
    def __init__(
        self,
        package_root: Path | None = None,
        local_files_only: bool | None = None,
        alpha: float = 7.5,
    ):
        self.paths = ModelPaths.discover(package_root)
        self.local_files_only = bool(
            int(os.environ.get("SERVING_LOCAL_FILES_ONLY", "0"))
            if local_files_only is None else local_files_only
        )
        self.alpha = float(os.environ.get("SERVING_TASK2_ALPHA", alpha))

        device_pref = os.environ.get("SERVING_DEVICE", "cpu").strip().lower()
        if device_pref == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load taxonomy (industry → sub-industry mapping)
        self.taxonomy = pd.read_csv(self.paths.task2_taxonomy_table, dtype=str).fillna("")
        self.taxonomy["industry_code"] = self.taxonomy["industry_code"].map(
            lambda x: clean_code(x, width=8)
        )
        self.taxonomy["subindustry_code"] = self.taxonomy["subindustry_code"].map(
            lambda x: clean_code(x, width=10)
        )
        self.taxonomy_text_by_leaf = self.taxonomy.set_index("subindustry_code")["taxonomy_text"].to_dict()
        self.leaf_name = self.taxonomy.set_index("subindustry_code")["subindustry_name"].to_dict()
        self.parent_name = (
            self.taxonomy.drop_duplicates("industry_code")
            .set_index("industry_code")["industry_name"].to_dict()
        )
        self.leaves_by_parent = (
            self.taxonomy.groupby("industry_code")["subindustry_code"]
            .apply(lambda x: list(dict.fromkeys(x))).to_dict()
        )

        # Load industry lookup for the /api/industries endpoint
        try:
            self.industry_lookup = pd.read_csv(self.paths.task1_industry_lookup, dtype=str).fillna("")
            self.industry_lookup["industry_code"] = self.industry_lookup["industry_code"].map(
                lambda x: clean_code(x, width=8)
            )
        except FileNotFoundError:
            self.industry_lookup = None

        self.industry_ranker = Task1FlangBertRanker(self.paths, self.device, self.local_files_only)
        self.subindustry_ranker = Task2SubindustryRanker(self.paths, self.device, self.local_files_only)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "device": str(self.device),
            "task1_model": TASK1_MODEL_NAME,
            "task1_champion": "FLANG-BERT v10 (Meet)",
            "task2_model": TASK2_CROSS_ENCODER_MODEL,
            "task2_alpha": self.alpha,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "package_root": str(self.paths.package_root),
            "taxonomy_subindustries": int(self.taxonomy["subindustry_code"].nunique()),
            "taxonomy_industries": int(self.taxonomy["industry_code"].nunique()),
        }

    def industry_options(self) -> list[dict[str, str]]:
        if self.industry_lookup is None:
            # Fallback: derive from taxonomy
            unique = self.taxonomy.drop_duplicates("industry_code")[["industry_code", "industry_name"]]
            return [
                {"industry_code": row["industry_code"], "industry_name": row["industry_name"]}
                for _, row in unique.iterrows()
            ]
        return self.industry_lookup.to_dict("records")

    def _records_to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        if not records:
            raise ValueError("`records` is empty.")
        df = pd.DataFrame(records)
        for col in ["SegmentName", "SegmentDescription", "LongProfile", "AsOfDate",
                    "CompanyId", "MstarGlobal", "Revenue",
                    "total_revenue_company_as_of", "revenue_share",
                    "is_largest_share_segment"]:
            if col not in df.columns:
                df[col] = ""
        return df

    def _row_meta(self, records, i, known_industry_code=None):
        rec = records[i]
        return {
            "row_id": i + 1,
            "segment_name": rec.get("SegmentName"),
            "company_id": rec.get("CompanyId"),
            "as_of_date": rec.get("AsOfDate"),
            "revenue": rec.get("Revenue"),
            "total_revenue_company_as_of": rec.get("total_revenue_company_as_of"),
            "revenue_share": rec.get("revenue_share"),
            "is_largest_share_segment": rec.get("is_largest_share_segment"),
            "known_industry_used": bool(known_industry_code),
        }

    def predict_industry(self, records: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
        df = self._records_to_frame(records)
        rows, probs = self.industry_ranker.top_industries(df, top_k=top_k)
        out = []
        for i, candidates in enumerate(rows):
            top_prob = candidates[0]["score"] if candidates else 0.0
            confidence_flag = "high" if top_prob >= LOW_CONFIDENCE_THRESHOLD else "low"
            for cand in candidates:
                cand["industry_name"] = self.parent_name.get(cand["industry_code"], "")
            row = self._row_meta(records, i)
            row.update({
                "industry_candidates": [{"rank":c["rank"],"industry_code":c["industry_code"],"industry_name":c["industry_name"],"confidence":c["score"],"score":c["score"]} for c in candidates],
                "subindustry_candidates": [],
                "top1_confidence": top_prob,
                "confidence_flag": confidence_flag,
                "recommendation": "auto_accept" if confidence_flag == "high" else "analyst_review",
            })
            out.append(row)
        return {"task": "industry", "results": out}

    def predict_subindustry(
        self,
        records: list[dict[str, Any]],
        top_k: int = 5,
        known_industry_code: str | None = None,
    ) -> dict[str, Any]:
        df = self._records_to_frame(records)

        # Stage 1: Get parent industry candidates
        if known_industry_code:
            code = clean_code(known_industry_code, width=8)
            if code not in self.leaves_by_parent:
                raise ValueError(f"Known industry code {code} not in taxonomy.")
            parent_candidates_per_row = [[(code, 1.0)]] * len(df)
            top1_confidence_per_row = [1.0] * len(df)
        else:
            log_p = self.industry_ranker.log_scores(df)
            probs = softmax(log_p, axis=1)
            order = np.argsort(-probs, axis=1)
            codes = self.industry_ranker.industry_codes
            parent_candidates_per_row = []
            top1_confidence_per_row = []
            for i in range(len(df)):
                top1_confidence_per_row.append(float(probs[i, order[i, 0]]))
                cands = []
                for class_idx in order[i, :PARENT_TOPK_FOR_SUBINDUSTRY]:
                    parent_code = str(codes[class_idx])
                    if parent_code in self.leaves_by_parent:
                        cands.append((parent_code, float(probs[i, class_idx])))
                parent_candidates_per_row.append(cands)

        # Stage 2 & 3: Expand to leaves, score with cross-encoder, combine with parent prior
        results = []
        for i in range(len(df)):
            row = df.iloc[i]
            seg_text = f"[SEGMENT_NAME] {row['SegmentName']} [SEGMENT_DESCRIPTION] {row['SegmentDescription']}".strip()

            parents = parent_candidates_per_row[i]
            pair_texts, leaf_codes, parent_priors = [], [], []
            for parent_code, parent_prob in parents:
                for leaf_code in self.leaves_by_parent.get(parent_code, []):
                    tax_text = self.taxonomy_text_by_leaf.get(leaf_code, "")
                    pair_texts.append(Task2SubindustryRanker.build_pair_text(seg_text, tax_text))
                    leaf_codes.append(leaf_code)
                    parent_priors.append(parent_prob)

            if not pair_texts:
                row = self._row_meta(records, i, known_industry_code)
                row.update({"industry_candidates":[],"subindustry_candidates":[],"top1_confidence":top1_confidence_per_row[i],"confidence_flag":"low","recommendation":"analyst_review"})
                results.append(row)
                continue

            ce_logits = self.subindustry_ranker.score_pairs(pair_texts)
            parent_log_priors = np.log(np.array(parent_priors) + 1e-12)
            combined = ce_logits + self.alpha * parent_log_priors

            from scipy.special import softmax as _sm
            order = np.argsort(-combined)
            combined_probs = _sm(combined)
            top = []
            for rank, idx in enumerate(order[:top_k], start=1):
                code = leaf_codes[idx]
                top.append({"rank":rank,"subindustry_code":code,"subindustry_name":self.leaf_name.get(code,""),"industry_code":code[:8],"industry_name":self.parent_name.get(code[:8],""),"confidence":float(combined_probs[idx]),"cross_encoder_score":float(ce_logits[idx]),"combined_score":float(combined[idx])})

            top1_conf = top1_confidence_per_row[i]
            confidence_flag = "high" if top1_conf >= LOW_CONFIDENCE_THRESHOLD else "low"
            row = self._row_meta(records, i, known_industry_code)
            row.update({"industry_candidates":[],"subindustry_candidates":top,"top1_confidence":top1_conf,"confidence_flag":confidence_flag,"recommendation":"auto_accept" if confidence_flag=="high" else "analyst_review"})
            results.append(row)

        return {"task": "subindustry", "alpha": self.alpha, "results": results}

    def predict_both(
        self,
        records: list[dict[str, Any]],
        top_k: int = 5,
        known_industry_code: str | None = None,
    ) -> dict[str, Any]:
        industry_result = self.predict_industry(records, top_k=top_k)
        subindustry_result = self.predict_subindustry(records, top_k=top_k, known_industry_code=known_industry_code)
        merged = []
        for ind, sub in zip(industry_result["results"], subindustry_result["results"]):
            row = dict(ind)
            row["subindustry_candidates"] = sub.get("subindustry_candidates", [])
            row["known_industry_used"] = sub.get("known_industry_used", False)
            merged.append(row)
        return {"task": "both", "alpha": self.alpha, "results": merged}


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor (lazy)
# ─────────────────────────────────────────────────────────────────────────────

_SERVICE: PredictionService | None = None
_SERVICE_LOCK = threading.Lock()


def get_service() -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = PredictionService()
    return _SERVICE


def predict_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Unified JSON-in/JSON-out entry point used by fastapi_app.py."""
    service = get_service()
    records = payload.get("records", [])
    top_k = int(payload.get("top_k", 5))
    known = payload.get("known_industry_code") or payload.get("industry_code")
    task = (payload.get("task") or "both").lower().strip()

    if task == "industry":
        return service.predict_industry(records, top_k=top_k)
    if task == "subindustry":
        return service.predict_subindustry(records, top_k=top_k, known_industry_code=known)
    return service.predict_both(records, top_k=top_k, known_industry_code=known)
