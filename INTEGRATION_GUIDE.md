# Integration Guide — Step by Step

Follow these steps to merge this package into your `depaul-morningstar-capstone` repo.

## Step 1 — Copy files into your repo

From the unzipped `gecs_integration/` folder, copy these into your repo:

```bash
# From the integration package into your repo root
cp -r serving_app/    /path/to/depaul-morningstar-capstone/
cp -r scripts/        /path/to/depaul-morningstar-capstone/
cp    README.md       /path/to/depaul-morningstar-capstone/      # ⚠️ overwrites existing
cp    .gitignore      /path/to/depaul-morningstar-capstone/      # ⚠️ overwrites existing
```

## Step 2 — Verify file layout

```bash
cd /path/to/depaul-morningstar-capstone
find . -type d -not -path './.git*' -not -path './notebooks*' | sort
```

You should see:
```
.
./scripts
./serving_app
./weights
./weights/task1
./weights/task2
./weights/task2/cross_encoder
```

## Step 3 — Test locally

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r serving_app/requirements.txt

# Run the server
cd serving_app
python -m uvicorn fastapi_app:app --host 0.0.0.0 --port 8000
```

First load will take 2–3 minutes (downloads HuggingFace tokenizers + loads 1 GB of weights into RAM).

In a separate terminal:

```bash
# Quick smoke test
curl http://localhost:8000/api/health

# Full prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 5,
    "records": [{
      "SegmentName": "Cloud Infrastructure",
      "SegmentDescription": "Compute, storage, networking, and database services.",
      "LongProfile": "Enterprise software, cloud, AI services."
    }]
  }'
```

## Step 4 — Upload weights to Drive (for distribution)

GitHub can't host the 1 GB of weights. Upload to Drive:

1. Create a new Drive folder, e.g. "GECS_Capstone_Weights"
2. Upload these files (preserving names):
   - `weights/task1/best_model_state.pt`
   - `weights/task1/label_encoder.pkl`
   - `weights/task1/label_encoder_sector.pkl`
   - `weights/task1/label_encoder_group.pkl`
   - `weights/task1/numeric_scaler.pkl`
   - `weights/task1/industry_lookup.csv`
   - `weights/task2/cross_encoder/best_state.pt`
   - `weights/task2/taxonomy_table.csv`
3. For each file: right-click → Share → "Anyone with the link" → copy the link
4. Extract the file ID from each link:
   - URL: `https://drive.google.com/file/d/ABC123XYZ/view?usp=sharing`
   - File ID: `ABC123XYZ`
5. Open `scripts/download_weights.py` and replace each `REPLACE_WITH_DRIVE_FILE_ID` with the actual ID

## Step 5 — Push to GitHub

```bash
cd /path/to/depaul-morningstar-capstone

# Stage everything (weights/ is gitignored)
git add .

# Verify weights/ NOT staged
git status | grep -i weights   # should be empty

# Commit and push
git commit -m "Integrate FLANG-BERT Task 1 + DeBERTa-v3 Task 2 production API"
git push origin main
```

## Step 6 — Deploy to Hugging Face Spaces (optional)

For a hosted demo:

1. Sign up at huggingface.co (free)
2. Create new Space → Type: **Gradio** or **Docker** → Hardware: **CPU basic (free)**
3. Clone the space repo, copy your code, and add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY serving_app/ ./serving_app/
COPY weights/ ./weights/
RUN pip install --no-cache-dir -r serving_app/requirements.txt
CMD ["python", "-m", "uvicorn", "serving_app.fastapi_app:app", "--host", "0.0.0.0", "--port", "7860"]
```

4. Push and wait for build (~10 min). Your API is live.

## Troubleshooting

### "Could not locate weights/"
Check files exist at the expected paths:
```bash
ls weights/task1/best_model_state.pt
ls weights/task2/cross_encoder/best_state.pt
ls weights/task2/taxonomy_table.csv
```

### Model loads slowly / OOM
On CPU expect 30–60 seconds for first prediction (model is lazy-loaded).
Minimum RAM: 4 GB. Recommended: 8 GB.

### "RuntimeError: Error(s) in loading state_dict"
The label encoder class count must match the saved model's leaf head. If you changed `label_encoder.pkl`, you must retrain.

### "Could not load FLANG-BERT tokenizer"
First run needs internet. After that, weights cache locally. To force offline mode after first run:
```bash
export SERVING_LOCAL_FILES_ONLY=1
```

## Architectural Notes

### What changed from Anas's serving_app

**Replaced:**
- `Task1IndustryRanker` (six-stream HGBT + SEC-BERT) → `Task1FlangBertRanker` (your FLANG-BERT champion)
- Path discovery (`final_best_models_package/...`) → simpler `weights/` layout

**Kept:**
- `Task2SubindustryRanker` (DeBERTa-v3-small cross-encoder)
- Taxonomy expansion logic (Stage 1 → 2 → 3)
- Soft parent prior with α=15
- 35% confidence routing threshold
- API endpoint signatures

### Sector/Group hint tokens

Your training notebook builds text with `[sector] [group]` tokens derived from `MstarGlobal`
(the label). At inference we leave these empty (`[] []`), since the answer is unknown.
The model has learned to use these as soft hints when available, but works without them.

### Numeric features at inference

When only a single segment record is provided, `n_segments=1` and `herfindahl_index = revenue_share²`.
For best accuracy, submit all segments of a company in one batch (multiple records) — the API
groups by `CompanyId + AsOfDate` to compute proper features.
