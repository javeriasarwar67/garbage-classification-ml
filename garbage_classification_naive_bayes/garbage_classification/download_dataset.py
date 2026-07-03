"""
download_dataset.py
───────────────────
Downloads the Garbage Classification dataset from Kaggle.

Dataset  : mostafaabla/garbage-classification
URL      : https://www.kaggle.com/datasets/mostafaabla/garbage-classification
Classes  : cardboard, glass, metal, paper, plastic, trash
Size     : ~400 MB (image dataset)

SETUP INSTRUCTIONS
──────────────────
1. Create a Kaggle account at https://www.kaggle.com
2. Go to: Account → Settings → API → Create New Token
3. Download 'kaggle.json' and place it at:
       Linux/Mac : ~/.kaggle/kaggle.json
       Windows   : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
4. Run:  python download_dataset.py
"""

import os
import sys
import json
import shutil
import zipfile
import subprocess
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────
DATASET_SLUG  = "mostafaabla/garbage-classification"
DOWNLOAD_DIR  = "data/raw"
OUTPUT_CSV    = "data/garbage_dataset.csv"
CATEGORIES    = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# ─────────────────────────────────────


def check_kaggle_credentials():
    """Verify kaggle.json exists."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_path.exists():
        print("❌ kaggle.json not found!")
        print(f"   Expected location: {kaggle_path}")
        print("\n📋 HOW TO GET IT:")
        print("   1. Go to https://www.kaggle.com → Account → Settings → API")
        print("   2. Click 'Create New Token' → downloads kaggle.json")
        print(f"   3. Move it to: {kaggle_path}")
        print("   4. Run this script again.\n")
        return False

    # Validate JSON
    with open(kaggle_path) as f:
        creds = json.load(f)
    if "username" not in creds or "key" not in creds:
        print("❌ kaggle.json is malformed. Re-download it from Kaggle.")
        return False

    os.chmod(kaggle_path, 0o600)   # required by kaggle library
    print(f"✅ Kaggle credentials found for user: {creds['username']}")
    return True


def install_kaggle():
    """Install kaggle package if missing."""
    try:
        import kaggle
    except ImportError:
        print("📦 Installing kaggle package …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        import kaggle
    return True


def download_dataset():
    """Download the dataset using the Kaggle API."""
    import kaggle

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"\n⬇️  Downloading dataset: {DATASET_SLUG}")
    print("   This may take a few minutes (~400 MB) …\n")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        DATASET_SLUG,
        path=DOWNLOAD_DIR,
        unzip=True,
        quiet=False,
    )
    print(f"\n✅ Dataset downloaded to '{DOWNLOAD_DIR}/'")


def build_csv_from_images():
    """
    Walk the downloaded image folders and build a CSV with:
    filename | category | split (train/test)
    """
    rows = []
    raw_path = Path(DOWNLOAD_DIR)

    for split in ["train", "test", ""]:  # handle different folder layouts
        base = raw_path / split if split else raw_path
        for category in CATEGORIES:
            cat_dir = base / category
            if not cat_dir.exists():
                continue
            for img_file in cat_dir.iterdir():
                if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    rows.append({
                        "filename":   str(img_file),
                        "category":   category,
                        "split":      split if split else "all",
                    })

    if not rows:
        print("⚠️  Could not locate image folders. Checking directory structure …")
        for p in sorted(raw_path.rglob("*"))[:30]:
            print("  ", p)
        raise RuntimeError("Please inspect the downloaded folder structure above "
                           "and adjust CATEGORIES or the path logic accordingly.")

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n📄 CSV saved → '{OUTPUT_CSV}'")
    print(f"   Total samples : {len(df)}")
    print(f"   Category counts:")
    print(df["category"].value_counts().to_string(indent=5))
    return df


def main():
    print("=" * 52)
    print("  Garbage Classification — Kaggle Downloader")
    print("=" * 52)

    if not check_kaggle_credentials():
        sys.exit(1)

    install_kaggle()
    download_dataset()
    df = build_csv_from_images()

    print("\n🎉 Done! Now run the classifier:")
    print("   python naive_bayes_classifier.py data/garbage_dataset.csv")


if __name__ == "__main__":
    main()
