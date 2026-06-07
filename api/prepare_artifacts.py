"""
Run this script locally to prepare artifacts folder for API deployment.
Copies model weights and encoders from Drive to artifacts/ folder.
"""
import shutil
from pathlib import Path

# ── Update these paths to match your local Drive location ──
DRIVE_BASE = Path.home() / "Documents/capstone/depaul-morningstar-capstone"

T1_SRC = DRIVE_BASE / "CAPSTONE/flangbert_champion_v2_artifacts"
T2_SRC = DRIVE_BASE / "CAPSTONE/task_2/flangbert_artifacts"
GECS   = DRIVE_BASE / "data/raw/GECS_Activities2026.csv"

# ── Destination ─────────────────────────────────────────────
ART_DIR = Path("artifacts")
T1_DST  = ART_DIR / "task1"
T2_DST  = ART_DIR / "task2"
T1_DST.mkdir(parents=True, exist_ok=True)
T2_DST.mkdir(parents=True, exist_ok=True)

# ── Copy Task 1 artifacts ───────────────────────────────────
for fname in ["best_model_state.pt", "label_encoder.pkl",
              "label_encoder_sector.pkl", "label_encoder_group.pkl",
              "numeric_scaler.pkl"]:
    src = T1_SRC / fname
    if src.exists():
        shutil.copy(src, T1_DST / fname)
        print(f"  Copied {fname}")
    else:
        print(f"  NOT FOUND: {src}")

# ── Copy Task 2 artifacts ───────────────────────────────────
for fname in ["best.pt", "label_encoder.pkl"]:
    src = T2_SRC / fname
    if src.exists():
        shutil.copy(src, T2_DST / fname)
        print(f"  Copied {fname}")
    else:
        print(f"  NOT FOUND: {src}")

# ── Build GECS definitions JSON ─────────────────────────────
import pandas as pd, json
if GECS.exists():
    gecs = pd.read_csv(GECS)
    gecs["id_clean"] = gecs["Industry ID"].dropna().astype(float).astype(int).astype(str)
    defs = gecs.groupby("id_clean")["Activity Definition"].first().to_dict()
    with open(ART_DIR / "gecs_definitions.json", "w") as f:
        json.dump(defs, f, indent=2)
    print(f"  GECS definitions: {len(defs)} entries saved")

print("\nArtifacts ready. Run: uvicorn main:app --reload")
