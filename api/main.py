"""
GECS Classification API v3
- Industry names loaded from hierarchy CSV
- T2 correlated with T1 prefix
- Claude fallback at 0.10 threshold
- /lookup and /hierarchy endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Union, Dict
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import time
import logging
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoConfig
from anthropic import Anthropic

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GECS Classification API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Paths ─────────────────────────────────────────────────
ARTIFACTS_DIR  = Path(__file__).parent / "artifacts"
T1_ARTIFACTS   = ARTIFACTS_DIR / "task1"
T2_ARTIFACTS   = ARTIFACTS_DIR / "task2"
GECS_DEFS_PATH = ARTIFACTS_DIR / "gecs_definitions.json"
HIER_PATH      = ARTIFACTS_DIR / "hierarchy.json"   # built on startup

# ── Config ────────────────────────────────────────────────
MODEL_NAME        = "SALT-NLP/FLANG-BERT"
MAX_LEN           = 512
NUM_FEATURES_T1   = 10
T1_CONF_HIGH      = 0.70
T1_CONF_MEDIUM    = 0.10   # Claude kicks in above 10%
T2_CONF_THRESHOLD = 0.30
CLAUDE_MODEL      = "claude-sonnet-4-6"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Pydantic ──────────────────────────────────────────────
class SegmentInput(BaseModel):
    name          : str
    description   : Optional[str] = ""
    revenue_share : Optional[float] = 0.0
    is_largest    : Optional[bool] = False

class CompanyRequest(BaseModel):
    company_id    : str
    long_profile  : Optional[str] = ""
    segments      : List[SegmentInput]
    as_of_date    : Optional[str] = "2024"
    total_revenue : Optional[float] = 0.0
    n_segments    : Optional[int] = None

class BatchRequest(BaseModel):
    companies     : List[CompanyRequest]

class ExplainRequest(BaseModel):
    segment_text  : str
    top3_preds    : List[dict]
    context       : Optional[str] = "industry"

class SegmentPrediction(BaseModel):
    segment_name           : str
    industry_code          : str
    industry_name          : str
    industry_confidence    : float
    industry_top5          : List[dict]
    subindustry_code       : str
    subindustry_name       : str
    subindustry_confidence : float
    subindustry_top5       : List[dict]
    route                  : str
    needs_review           : bool
    latency_ms             : float

class CompanyResponse(BaseModel):
    company_id       : str
    predictions      : List[SegmentPrediction]
    total_latency_ms : float

class BatchResponse(BaseModel):
    results          : List[CompanyResponse]
    total_companies  : int
    total_segments   : int
    total_latency_ms : float
    routes_summary   : dict

# ── Model architectures ───────────────────────────────────
class FLANGMultiTaskT1(nn.Module):
    def __init__(self, model_name, n_leaf, n_sector, n_group, num_features=10, dropout=0.1):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(model_name)
        h                = self.bert.config.hidden_size
        self.dropout     = nn.Dropout(dropout)
        self.leaf_head   = nn.Linear(h + num_features, n_leaf)
        self.sector_head = nn.Linear(h, n_sector)
        self.group_head  = nn.Linear(h, n_group)

    def forward(self, input_ids, attention_mask, numeric_feats):
        out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask     = attention_mask.unsqueeze(-1).float()
        pooled   = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled   = self.dropout(pooled)
        combined = torch.cat([pooled, numeric_feats.to(pooled.dtype)], dim=-1)
        return self.leaf_head(combined), self.sector_head(pooled), self.group_head(pooled)

class FLANGMultiTaskT2(nn.Module):
    def __init__(self, n_subind, n_industry, n_sector, hidden_size=768, num_layers=12, dropout=0.1):
        super().__init__()
        config                     = AutoConfig.from_pretrained(MODEL_NAME)
        config.hidden_size         = hidden_size
        config.num_hidden_layers   = num_layers
        config.intermediate_size   = hidden_size * 4
        config.num_attention_heads = max(1, hidden_size // 64)
        self.bert        = AutoModel.from_config(config)
        self.dropout     = nn.Dropout(dropout)
        self.subind_head = nn.Linear(hidden_size, n_subind)
        self.ind_head    = nn.Linear(hidden_size, n_industry)
        self.sector_head = nn.Linear(hidden_size, n_sector)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        pooled = self.dropout(pooled)
        return self.subind_head(pooled), self.ind_head(pooled), self.sector_head(pooled)

# ── Global state ──────────────────────────────────────────
class AppState:
    t1_model         = None
    t2_model         = None
    tokenizer        = None
    le_t1            = None
    le_t2            = None
    le_t1_sector     = None
    le_t1_group      = None
    scaler_t1        = None
    gecs_defs        = {}
    # Name lookups: code (str) → name (str)
    industry_names   = {}   # 8-digit → name
    subind_names     = {}   # 10-digit → name
    sector_names     = {}   # 3-digit → name
    group_names      = {}   # 5-digit → name
    # Full hierarchy tree for /hierarchy endpoint
    hierarchy_tree   = {}
    t2_prefix_index  = {}
    anthropic_client = None
    t2_available     = False
    ready            = False

state = AppState()

def _load_hierarchy():
    """Load industry names from hierarchy CSV or GECS Activities file."""
    # Try hierarchy CSV first
    hier_csv = ARTIFACTS_DIR / "industries_Hierarchy.csv"
    gecs_csv  = ARTIFACTS_DIR / "GECS_Activities2026.csv"

    ind_names  = {}
    sub_names  = {}
    sec_names  = {}
    grp_names  = {}
    tree       = {}

    # Try GECS Activities CSV
    for csv_path in [gecs_csv, hier_csv]:
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                logger.info(f"Hierarchy columns: {df.columns.tolist()}")
                cols = df.columns.tolist()

                # Try to map columns flexibly
                id_col   = next((c for c in cols if 'activity' in c.lower() and 'id' in c.lower()), None) or \
                           next((c for c in cols if 'industry' in c.lower() and 'id' in c.lower()), None)
                name_col = next((c for c in cols if 'activity' in c.lower() and ('name' in c.lower() or 'defin' in c.lower())), None) or \
                           next((c for c in cols if 'name' in c.lower()), None) or \
                           next((c for c in cols if 'defin' in c.lower()), None)

                if id_col and name_col:
                    for _, row in df.iterrows():
                        try:
                            raw_id = str(row[id_col]).split('.')[0].strip()
                            name   = str(row[name_col]).strip()[:80]
                            if raw_id and raw_id != 'nan' and name and name != 'nan':
                                clean_id = raw_id.replace('.0','')
                                ind_names[clean_id] = name
                                # derive sector/group from id length
                                if len(clean_id) >= 3:
                                    sec_names[clean_id[:3]] = sec_names.get(clean_id[:3], name[:40])
                                if len(clean_id) >= 5:
                                    grp_names[clean_id[:5]] = grp_names.get(clean_id[:5], name[:40])
                        except:
                            pass
                    logger.info(f"Loaded {len(ind_names)} industry names from {csv_path.name}")
                    break
            except Exception as e:
                logger.warning(f"Could not load hierarchy from {csv_path}: {e}")

    # Fallback — build from GECS definitions keys (already loaded)
    if not ind_names and state.gecs_defs:
        for code, defn in state.gecs_defs.items():
            name = defn[:60] if defn else code
            ind_names[str(code)] = name
        logger.info(f"Built {len(ind_names)} names from GECS definitions fallback")

    # Known sector names from GECS taxonomy
    known_sectors = {
        '101':'Basic Materials','102':'Consumer Cyclical','103':'Financial Services',
        '104':'Real Estate','105':'Consumer Defensive','106':'Healthcare',
        '107':'Utilities','108':'Communication Services','109':'Energy',
        '205':'Consumer Staples','206':'Healthcare','207':'Utilities',
        '208':'Communication Services','209':'Energy','251':'Consumer Cyclical',
        '301':'Consumer Staples','302':'Healthcare','303':'Industrials',
        '304':'Technology','305':'Financial Services','306':'Real Estate',
        '307':'Basic Materials','308':'Consumer Cyclical','309':'Energy',
        '310':'Financial Services','311':'Technology','312':'Communication Services',
        '313':'Healthcare','314':'Industrials','315':'Real Estate',
        '316':'Consumer Staples','317':'Utilities','318':'Basic Materials',
    }
    for k, v in known_sectors.items():
        if k not in sec_names:
            sec_names[k] = v

    return ind_names, sub_names, sec_names, grp_names

# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
async def load_models():
    logger.info("Loading models...")
    t0 = time.time()
    try:
        state.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Task 1
        with open(T1_ARTIFACTS / "label_encoder.pkl", "rb") as f:
            state.le_t1 = pickle.load(f)
        with open(T1_ARTIFACTS / "label_encoder_sector.pkl", "rb") as f:
            state.le_t1_sector = pickle.load(f)
        with open(T1_ARTIFACTS / "label_encoder_group.pkl", "rb") as f:
            state.le_t1_group = pickle.load(f)
        with open(T1_ARTIFACTS / "numeric_scaler.pkl", "rb") as f:
            state.scaler_t1 = pickle.load(f)

        NUM_CLASSES_T1 = len(state.le_t1.classes_)
        NUM_SECTORS_T1 = len(state.le_t1_sector.classes_)
        NUM_GROUPS_T1  = len(state.le_t1_group.classes_)

        state.t1_model = FLANGMultiTaskT1(
            MODEL_NAME, NUM_CLASSES_T1, NUM_SECTORS_T1, NUM_GROUPS_T1, NUM_FEATURES_T1
        ).to(DEVICE)
        state.t1_model.load_state_dict(
            torch.load(T1_ARTIFACTS / "best_model_state.pt", map_location=DEVICE), strict=True
        )
        state.t1_model.eval()
        logger.info(f"Task 1 loaded — {NUM_CLASSES_T1} classes")

        # Task 2
        try:
            with open(T2_ARTIFACTS / "label_encoder.pkl", "rb") as f:
                state.le_t2 = pickle.load(f)
            NUM_CLASSES_T2 = len(state.le_t2.classes_)

            t2_cfg_path = T2_ARTIFACTS / "model_config.json"
            if t2_cfg_path.exists():
                with open(t2_cfg_path) as f:
                    t2_cfg = json.load(f)
                hidden_size = t2_cfg.get("hidden_size", 768)
                num_layers  = t2_cfg.get("num_layers", 12)
            else:
                hidden_size, num_layers = 768, 12

            state.t2_model = FLANGMultiTaskT2(
                NUM_CLASSES_T2, NUM_CLASSES_T1, NUM_SECTORS_T1,
                hidden_size=hidden_size, num_layers=num_layers
            ).to(DEVICE)
            t2_weights   = torch.load(T2_ARTIFACTS / "best_model_state.pt", map_location=DEVICE)
            actual_vocab = t2_weights['bert.embeddings.word_embeddings.weight'].shape[0]
            state.t2_model.bert.resize_token_embeddings(actual_vocab)
            state.t2_model.load_state_dict(t2_weights, strict=False)
            state.t2_model.eval()
            state.t2_available = True

            # T2 prefix index
            prefix_index = {}
            for idx, cls in enumerate(state.le_t2.classes_):
                prefix = str(cls)[:8]
                if prefix not in prefix_index:
                    prefix_index[prefix] = []
                prefix_index[prefix].append(idx)
            state.t2_prefix_index = prefix_index
            logger.info(f"Task 2 loaded — {NUM_CLASSES_T2} classes, {len(prefix_index)} prefixes")
        except Exception as e:
            logger.warning(f"Task 2 not loaded: {e}")
            state.t2_available = False

        # GECS definitions
        if GECS_DEFS_PATH.exists():
            with open(GECS_DEFS_PATH) as f:
                state.gecs_defs = json.load(f)

        # Load hierarchy names
        state.industry_names, state.subind_names, state.sector_names, state.group_names = _load_hierarchy()

        # Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if api_key:
            state.anthropic_client = Anthropic(api_key=api_key)
            logger.info("Anthropic client ready")

        # Warmup
        dummy = state.tokenizer("warmup", return_tensors="pt",
                                max_length=32, padding="max_length", truncation=True)
        ids  = dummy["input_ids"].to(DEVICE)
        mask = dummy["attention_mask"].to(DEVICE)
        feat = torch.zeros(1, NUM_FEATURES_T1).to(DEVICE)
        with torch.no_grad():
            state.t1_model(ids, mask, feat)
            if state.t2_available:
                state.t2_model(ids, mask)

        state.ready = True
        logger.info(f"All models ready in {time.time()-t0:.1f}s on {DEVICE}")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# ── Landing ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def landing():
    html_path = Path(__file__).parent / "gecs_platform.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>GECS API v3 Running</h1><p><a href='/docs'>API Docs</a></p>"

# ── Text builders ─────────────────────────────────────────
def build_t1_text(seg, siblings, long_profile, year):
    seg_name = str(seg.name).strip()
    seg_desc = str(seg.description or seg_name).strip()
    lp_short = " ".join(str(long_profile).split()[:100]) if long_profile else ""
    sib_parts = []
    for sib in siblings:
        rev_pct   = round(float(sib.revenue_share or 0) * 100, 1)
        sib_desc  = str(sib.description or sib.name).strip()
        sib_short = " ".join(sib_desc.split()[:25])
        sib_parts.append(f"[SEG {rev_pct}%] {sib_short}")
    parts = ["[PRIMARY]", seg_name, "[SEP]", seg_desc]
    if sib_parts:
        parts.append(" ".join(sib_parts))
    if lp_short:
        parts.append(f"[LP] {lp_short}")
    return " ".join(parts)

def build_t2_text(seg, siblings, industry_prefix=""):
    seg_name = str(seg.name).strip()
    seg_desc = str(seg.description or seg_name).strip()
    sib_parts = []
    for sib in siblings:
        sib_desc  = str(sib.description or sib.name).strip()
        sib_short = " ".join(sib_desc.split()[:25])
        if sib_short:
            sib_parts.append(sib_short)
    prefix_token = f"[{industry_prefix}]" if industry_prefix else ""
    text = f"{prefix_token} [PRIMARY] {seg_name}: {seg_desc}"
    if sib_parts:
        text += " [SIBLINGS] " + " | ".join(sib_parts[:5])
    return text

def build_numeric_features(seg, company):
    n_segs = company.n_segments or len(company.segments)
    rev    = float(seg.revenue_share or 0)
    total  = float(company.total_revenue or 0)
    raw = np.array([[
        rev, 1.0 if seg.is_largest else 0.0,
        np.log1p(total * rev), np.log1p(total), float(n_segs),
        1.0 / max(n_segs, 1), 4.0,
        1.0 if len(str(company.long_profile or "").split()) < 50 else 0.0,
        1.0 if len(str(seg.description or "").split()) < 20 else 0.0,
        1.0 if len(str(seg.name or "").split()) < 2 else 0.0,
    ]], dtype=np.float32)
    if state.scaler_t1:
        CONT = [0, 2, 3, 4, 5, 6]
        raw[0, CONT] = state.scaler_t1.transform(raw[:, CONT])[0]
    return torch.tensor(raw, dtype=torch.float32)

def get_name(code, names_dict, fallback=""):
    code = str(code).strip()
    return names_dict.get(code) or names_dict.get(code.lstrip('0')) or fallback

# ── Inference ─────────────────────────────────────────────
@torch.no_grad()
def run_t1_inference(texts, feats_list):
    enc   = state.tokenizer(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
    ids   = enc["input_ids"].to(DEVICE)
    mask  = enc["attention_mask"].to(DEVICE)
    feats = torch.cat(feats_list, dim=0).to(DEVICE)
    logits, _, _ = state.t1_model(ids, mask, feats)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()

@torch.no_grad()
def run_t2_inference(texts):
    if not state.t2_available:
        return None
    enc  = state.tokenizer(texts, max_length=MAX_LEN, padding=True, truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    logits, _, _ = state.t2_model(ids, mask)
    return torch.softmax(logits.float(), dim=-1).cpu().numpy()

def get_top5(probs, le, names_dict=None):
    top5_idx = np.argsort(probs)[::-1][:5]
    result = []
    for i in top5_idx:
        code = le.inverse_transform([int(i)])[0]
        entry = {"code": code, "probability": round(float(probs[i]), 4)}
        if names_dict:
            entry["name"] = get_name(code, names_dict)
        result.append(entry)
    return result

def get_top5_filtered(probs, le, valid_indices, names_dict=None):
    if not valid_indices:
        return get_top5(probs, le, names_dict)
    filtered = sorted([(probs[i], i) for i in valid_indices], reverse=True)
    result = []
    for p, i in filtered[:5]:
        code = le.inverse_transform([int(i)])[0]
        entry = {"code": code, "probability": round(float(p), 4)}
        if names_dict:
            entry["name"] = get_name(code, names_dict)
        result.append(entry)
    return result

# ── Claude ────────────────────────────────────────────────
async def claude_classify(segment_text, top3):
    if not state.anthropic_client:
        return None
    defs = ""
    for p in top3:
        code = p["code"]
        defn = state.gecs_defs.get(str(code), "No definition available")
        name = get_name(code, state.industry_names)
        defs += f"\n{code} {name} (conf {p['probability']:.2f}): {defn[:200]}\n"

    prompt = f"""You are a Morningstar GECS financial industry classifier.

Segment text: "{segment_text[:500]}"

Top model predictions:
{defs}

Reply with ONLY the 8-digit industry code that best fits. No explanation."""
    try:
        r = state.anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )
        code = r.content[0].text.strip().replace(".", "").replace(" ", "")
        return code if code in state.le_t1.classes_ else None
    except Exception as e:
        logger.warning(f"Claude classify error: {e}")
        return None

# ── Core predict ──────────────────────────────────────────
async def predict_company(company: CompanyRequest) -> CompanyResponse:
    t0   = time.time()
    year = str(company.as_of_date or "2024")[:4]

    t1_texts, t1_feats = [], []
    for i, seg in enumerate(company.segments):
        siblings = [s for j, s in enumerate(company.segments) if j != i]
        t1_texts.append(build_t1_text(seg, siblings, company.long_profile, year))
        t1_feats.append(build_numeric_features(seg, company))

    t1_probs_all = run_t1_inference(t1_texts, t1_feats)

    predictions = []
    for i, seg in enumerate(company.segments):
        seg_t0   = time.time()
        siblings = [s for j, s in enumerate(company.segments) if j != i]

        t1_probs = t1_probs_all[i]
        t1_conf  = float(t1_probs.max())
        t1_class = state.le_t1.inverse_transform([int(t1_probs.argmax())])[0]
        t1_top5  = get_top5(t1_probs, state.le_t1, state.industry_names)

        if t1_conf >= T1_CONF_HIGH:
            route    = "model_auto"
            final_t1 = t1_class
        elif t1_conf >= T1_CONF_MEDIUM and state.anthropic_client:
            claude_code = await claude_classify(t1_texts[i], t1_top5[:3])
            if claude_code:
                final_t1 = claude_code
                route    = "claude_fallback"
            else:
                final_t1 = t1_class
                route    = "model_auto"
        else:
            final_t1 = t1_class
            route    = "human_review"

        ind_name = get_name(final_t1, state.industry_names, final_t1)

        # T2 — correlated with T1
        t2_text      = build_t2_text(seg, siblings, industry_prefix=final_t1)
        t2_probs_arr = run_t2_inference([t2_text])

        if t2_probs_arr is not None:
            t2_probs        = t2_probs_arr[0]
            valid_t2_indices = state.t2_prefix_index.get(str(final_t1), [])
            if valid_t2_indices:
                best_idx = max(valid_t2_indices, key=lambda idx: t2_probs[idx])
                t2_conf  = float(t2_probs[best_idx])
                t2_class = state.le_t2.inverse_transform([best_idx])[0]
                t2_top5  = get_top5_filtered(t2_probs, state.le_t2, valid_t2_indices, state.subind_names)
            else:
                t2_conf  = float(t2_probs.max())
                t2_class = state.le_t2.inverse_transform([int(t2_probs.argmax())])[0]
                t2_top5  = get_top5(t2_probs, state.le_t2, state.subind_names)
            if t2_conf < T2_CONF_THRESHOLD and route != "human_review":
                route = "human_review_subindustry"
            sub_name = get_name(t2_class, state.subind_names, t2_class)
        else:
            t2_conf, t2_class, t2_top5 = 0.0, "unavailable", []
            sub_name = ""

        predictions.append(SegmentPrediction(
            segment_name           = seg.name,
            industry_code          = final_t1,
            industry_name          = ind_name,
            industry_confidence    = round(t1_conf, 4),
            industry_top5          = t1_top5,
            subindustry_code       = t2_class,
            subindustry_name       = sub_name,
            subindustry_confidence = round(t2_conf, 4),
            subindustry_top5       = t2_top5,
            route                  = route,
            needs_review           = route in ("human_review", "human_review_subindustry"),
            latency_ms             = round((time.time() - seg_t0) * 1000, 2),
        ))

    return CompanyResponse(
        company_id       = company.company_id,
        predictions      = predictions,
        total_latency_ms = round((time.time() - t0) * 1000, 2),
    )

# ── Endpoints ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status"             : "ready" if state.ready else "loading",
        "device"             : str(DEVICE),
        "task1_classes"      : len(state.le_t1.classes_) if state.le_t1 else 0,
        "task2_classes"      : len(state.le_t2.classes_) if state.le_t2 else 0,
        "task2_available"    : state.t2_available,
        "t2_prefix_coverage" : len(state.t2_prefix_index),
        "claude_enabled"     : state.anthropic_client is not None,
        "industry_names_loaded": len(state.industry_names),
        "t1_conf_thresholds" : {"high": T1_CONF_HIGH, "medium": T1_CONF_MEDIUM},
    }

@app.get("/lookup/{code}")
async def lookup(code: str):
    name = get_name(code, state.industry_names) or \
           get_name(code, state.subind_names) or \
           get_name(code, state.sector_names) or \
           get_name(code, state.group_names)
    defn = state.gecs_defs.get(code, "")
    return {"code": code, "name": name or "Unknown", "definition": defn}

@app.get("/hierarchy")
async def hierarchy():
    """Return sector/group/industry tree for the hierarchy explorer."""
    tree = {}
    for code in state.le_t1.classes_:
        sec = str(code)[:3]
        grp = str(code)[:5]
        if sec not in tree:
            tree[sec] = {"name": get_name(sec, state.sector_names, sec), "groups": {}}
        if grp not in tree[sec]["groups"]:
            tree[sec]["groups"][grp] = {"name": get_name(grp, state.group_names, grp), "industries": []}
        tree[sec]["groups"][grp]["industries"].append({
            "code": code,
            "name": get_name(code, state.industry_names, code)
        })
    return tree

@app.post("/explain")
async def explain(request: ExplainRequest):
    if not state.anthropic_client:
        raise HTTPException(status_code=503, detail="Claude not configured")
    top3_str = ", ".join([
        f"{p['code']} {get_name(p['code'], state.industry_names)} ({p.get('probability',0)*100:.1f}%)"
        for p in request.top3_preds
    ])
    prompt = f"""You are a Morningstar GECS financial industry classifier expert.

Segment text: "{request.segment_text[:400]}"

Top model predictions: {top3_str}

In 2-3 sentences, explain why the top prediction is correct and how it differs from alternatives. Be concise and financially specific."""
    try:
        r = state.anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return {"explanation": r.content[0].text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/full")
async def predict_full(request: Union[CompanyRequest, BatchRequest]):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    t0 = time.time()
    if isinstance(request, CompanyRequest):
        return await predict_company(request)
    results = []
    for company in request.companies:
        results.append(await predict_company(company))
    routes = {}
    for r in results:
        for p in r.predictions:
            routes[p.route] = routes.get(p.route, 0) + 1
    return BatchResponse(
        results=results, total_companies=len(results),
        total_segments=sum(len(r.predictions) for r in results),
        total_latency_ms=round((time.time()-t0)*1000, 2),
        routes_summary=routes,
    )

@app.post("/predict/industry")
async def predict_industry(request: Union[CompanyRequest, BatchRequest]):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models still loading")
    companies = request.companies if isinstance(request, BatchRequest) else [request]
    results = []
    for company in companies:
        year = str(company.as_of_date or "2024")[:4]
        segs = []
        for i, seg in enumerate(company.segments):
            siblings = [s for j, s in enumerate(company.segments) if j != i]
            text  = build_t1_text(seg, siblings, company.long_profile, year)
            feat  = build_numeric_features(seg, company)
            probs = run_t1_inference([text], [feat])[0]
            conf  = float(probs.max())
            cls   = state.le_t1.inverse_transform([int(probs.argmax())])[0]
            segs.append({
                "segment_name"  : seg.name,
                "industry_code" : cls,
                "industry_name" : get_name(cls, state.industry_names, cls),
                "confidence"    : round(conf, 4),
                "top5"          : get_top5(probs, state.le_t1, state.industry_names),
                "needs_review"  : conf < T1_CONF_HIGH,
            })
        results.append({"company_id": company.company_id, "segments": segs})
    return results[0] if isinstance(request, CompanyRequest) else {"results": results}
