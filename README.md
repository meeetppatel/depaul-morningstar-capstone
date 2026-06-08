# DePaul × Morningstar Capstone — GECS Industry & Sub-Industry Predictor

ML system that classifies SEC 10-K segment records into Morningstar's Global Equity Classification Standard (GECS) hierarchy.

**DePaul University MGT 599 Business Analytics Capstone · Group 7 · Spring 2026**

## Final Results

| Task | Metric | Target | Achieved |
|---|---|---|---|
| **Task 1** — Industry Classification (145 classes) | Macro F1 | ≥ 0.75 | **0.8325** ✓ |
| **Task 1** | Top-10 Macro F1 | > 0.85 | **0.9311** ✓ |
| **Task 2** — Sub-Industry Classification (374 classes) | Top-1 Accuracy | — | **74.9%** |
| **Task 2** | Top-10 Macro F1 | > 0.85 | **89.1%** ✓ |
| API Inference Latency | — | < 100ms | **Sub-100ms validated** |

## Architecture

```
Segment record ─┐
                ├──→ Task1FlangBertRanker (FLANG-BERT v10 + multi-task heads + 10 numeric features)
                │      └─→ Top-K industries with confidence scores
                │
                └──→ Task2SubindustryRanker (DeBERTa-v3-small cross-encoder)
                       ├─→ Stage 1: Use Task 1 top-K parents
                       ├─→ Stage 2: Expand parents to ~50 candidate sub-industries
                       └─→ Stage 3: Cross-encoder rerank with soft parent prior (α=15)
```

**Confidence routing** — top-1 score ≥ 0.35 auto-accepted; below 0.35 flagged for analyst review (top-5 shortlist). The correct answer appears in top-5 for **85%+ of Task 1** and **92.4% of Task 2** cases.

## Repository Layout

```
.
├── notebooks/                    # Training notebooks (Colab)
│   ├── 01_cleaning_v10.ipynb              # Final data cleaning
│   ├── 02_eda.ipynb                       # Exploratory analysis
│   ├── 02_baseline_v9.ipynb               # TF-IDF baseline
│   ├── 03_flangbert_v10_colab.ipynb       # FLANG-BERT v10 (Task 1)
│   ├── 06_flang_deberta_champion_colab.ipynb  # Task 1 champion (0.8325)
│   └── task2_*.ipynb                      # Task 2 experiments
│
├── serving_app/                  # Production REST API
│   ├── fastapi_app.py            # FastAPI endpoints
│   ├── inference.py              # Full ML pipeline (Task 1 + Task 2)
│   └── requirements.txt
│
├── scripts/
│   └── download_weights.py       # Pulls model weights from Drive
│
├── weights/                      # ⚠️ Gitignored — download separately
│   ├── task1/                    # FLANG-BERT champion (418 MB)
│   └── task2/                    # Cross-encoder (539 MB) + taxonomy
│
└── README.md
```

## Quickstart

### 1. Clone

```bash
git clone https://github.com/meeetppatel/depaul-morningstar-capstone.git
cd depaul-morningstar-capstone
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r serving_app/requirements.txt
```

### 3. Download model weights

Weights total ~1.4 GB and are not on GitHub. Two options:

**Option A — Automatic (recommended):**

```bash
# After editing scripts/download_weights.py with Drive file IDs:
python scripts/download_weights.py
```

**Option B — Manual:**

Download from [Google Drive](https://drive.google.com/drive/folders/REPLACE_WITH_YOUR_DRIVE_LINK) and place files at:

```
weights/task1/best_model_state.pt           (418 MB)
weights/task1/label_encoder.pkl
weights/task1/label_encoder_sector.pkl
weights/task1/label_encoder_group.pkl
weights/task1/numeric_scaler.pkl
weights/task1/industry_lookup.csv
weights/task2/cross_encoder/best_state.pt   (539 MB)
weights/task2/taxonomy_table.csv
```

### 4. Run the API

```bash
cd serving_app
python -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Landing page |
| `GET` | `/api/health` | Server + model metadata |
| `GET` | `/api/industries` | List all 145 GECS industries |
| `POST` | `/api/predict` | Full pipeline (industry + sub-industry) |
| `POST` | `/api/predict_industry` | Task 1 only |
| `POST` | `/api/predict_subindustry` | Task 2 only |

### Example

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 5,
    "records": [{
      "CompanyId": "TEST001",
      "AsOfDate": "2024-12-31",
      "SegmentName": "Cloud Infrastructure",
      "SegmentDescription": "Compute, storage, networking, and database services for enterprise customers.",
      "LongProfile": "The company provides enterprise software, cloud, and AI services worldwide.",
      "Revenue": 42000,
      "total_revenue_company_as_of": 100000,
      "revenue_share": 0.42,
      "is_largest_share_segment": true
    }]
  }'
```

Response includes top-K industry candidates with confidence scores, top-K sub-industry candidates with combined CE + parent-prior scores, and a routing recommendation (`auto_accept` or `analyst_review`) based on the 35% threshold.

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SERVING_DEVICE` | `cpu` | Set to `cuda` if GPU is available |
| `SERVING_TASK2_ALPHA` | `15.0` | Parent-prior weight for Task 2 reranking |
| `SERVING_TASK2_BATCH_SIZE` | `4` (CPU) / `16` (GPU) | Cross-encoder batch size |
| `SERVING_LOCAL_FILES_ONLY` | `0` | Set to `1` to disable HuggingFace downloads |

## Team

| Member | Role |
|---|---|
| Saumyaa Kannan | Project Manager & Business Lead |
| Hanane Nekkaz | Technical Co-Lead — Modeling (Task 2 cross-encoder) |
| Meet Patel | Technical Co-Lead — Evaluation & API (Task 1 FLANG-BERT champion) |
| Anas Syed | Business Research & Documentation Lead |
| Dev Gauravbhai Patel | Visualization & Reporting Lead |

## Training Details — Task 1 (FLANG-BERT v10)

- Base: `SALT-NLP/FLANG-BERT` (110M params)
- Multi-task heads: leaf (145) + sector (11) + group (55), aux weights 0.15 / 0.20
- 10 numeric features concatenated to BERT pooled output before leaf head
- 3-phase training: 6 epochs CE → 4 epochs Focal (γ=1.0) → 4 epochs CE @ low LR
- Layer-wise LR decay (LLRD) factor 0.95
- GroupShuffleSplit by CompanyId to prevent leakage
- Sibling context: other segments at the same (CompanyId, AsOfDate) included in input

## Training Details — Task 2 (Cross-Encoder)

- Base: `microsoft/deberta-v3-small`
- Binary classifier on (segment_text, taxonomy_text) pairs
- Scoring: `score(c) = logit_CE(text, taxonomy_c) + α · log P_parent(parent(c))`
- α tuned on validation set, optimal value = 15
- Parent top-10 recall on Task 2 test: 99.4%

## License

MIT
