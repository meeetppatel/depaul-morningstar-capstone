"""
Download the model weights from Google Drive into weights/.

Edit the FILE_IDS dict below after uploading your weights to Drive.

Usage:
    python scripts/download_weights.py
"""

import os
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Edit these after uploading your weights to Google Drive
# Get file IDs by sharing a file → "Anyone with the link" → copy ID from URL
# Example: https://drive.google.com/file/d/<ID>/view → use <ID>
# ─────────────────────────────────────────────────────────────────────────────

FILE_IDS = {
    # Task 1 — FLANG-BERT champion
    "weights/task1/best_model_state.pt":         "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task1/label_encoder.pkl":           "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task1/label_encoder_sector.pkl":    "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task1/label_encoder_group.pkl":     "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task1/numeric_scaler.pkl":          "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task1/industry_lookup.csv":         "REPLACE_WITH_DRIVE_FILE_ID",

    # Task 2 — DeBERTa-v3-small cross-encoder
    "weights/task2/cross_encoder/best_state.pt": "REPLACE_WITH_DRIVE_FILE_ID",
    "weights/task2/taxonomy_table.csv":          "REPLACE_WITH_DRIVE_FILE_ID",
}


def download_file(file_id: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed. Run: pip install gdown")

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Already exists: {dest}")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  ↓ Downloading {dest.name} ...")
    gdown.download(url, str(dest), quiet=False)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    missing_ids = [path for path, fid in FILE_IDS.items() if "REPLACE" in fid]
    if missing_ids:
        print("ERROR: Edit scripts/download_weights.py and set Drive file IDs for:")
        for p in missing_ids:
            print(f"  - {p}")
        print()
        print("Alternative: download manually from your Drive and place files at:")
        for p in FILE_IDS:
            print(f"  {p}")
        return 1

    print("Downloading model weights ...")
    for relative_path, file_id in FILE_IDS.items():
        dest = repo_root / relative_path
        download_file(file_id, dest)

    print()
    print("Done. Verify layout:")
    print(f"  ls -lh {repo_root}/weights/task1/")
    print(f"  ls -lh {repo_root}/weights/task2/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
