"""
prepare_dataset.py
------------------
Robust dataset preparation and validation pipeline for Brain Tumor MRI Detection.
Creates a reproducible 70% Training / 15% Validation / 15% Testing split.

Classes:
- glioma
- meningioma
- notumor
- pituitary
"""

import os
import sys
import shutil
import hashlib
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

# Standard classes required for the project
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file content to detect duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_valid_image(filepath: Path) -> bool:
    """Verify that an image is uncorrupted and has valid pixel dimensions."""
    try:
        if filepath.suffix.lower() not in VALID_EXTENSIONS:
            return False
        if filepath.stat().st_size == 0:
            return False
        with Image.open(filepath) as img:
            img.verify()
        # Re-open to check if image can be converted/read into array
        with Image.open(filepath) as img:
            _ = np.array(img)
        return True
    except Exception:
        return False


def generate_synthetic_samples(target_dir: Path, samples_per_class: int = 20):
    """
    Generate realistic synthetic MRI brain scan samples for testing the pipeline
    when raw datasets have not yet been downloaded.
    """
    print(f"\n[INFO] Creating {samples_per_class} synthetic MRI sample images per class in '{target_dir}'...")
    raw_dir = target_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            # Create synthetic skull/brain MRI slice
            img = np.zeros((224, 224), dtype=np.uint8)
            y, x = np.ogrid[:224, :224]
            # Brain outline (ellipse)
            mask = ((x - 112)**2 / 75**2 + (y - 112)**2 / 90**2) <= 1
            # Brain tissue background texture
            noise = np.random.normal(90, 15, (224, 224)).clip(40, 140).astype(np.uint8)
            img[mask] = noise[mask]
            
            # Add synthetic tumor characteristics if not 'notumor'
            if cls == "glioma":
                tumor_mask = ((x - 85)**2 / 20**2 + (y - 95)**2 / 25**2) <= 1
                img[tumor_mask] = np.random.normal(190, 15, (224, 224)).clip(160, 235).astype(np.uint8)[tumor_mask]
            elif cls == "meningioma":
                tumor_mask = ((x - 145)**2 / 18**2 + (y - 80)**2 / 18**2) <= 1
                img[tumor_mask] = np.random.normal(210, 10, (224, 224)).clip(180, 245).astype(np.uint8)[tumor_mask]
            elif cls == "pituitary":
                tumor_mask = ((x - 112)**2 / 15**2 + (y - 150)**2 / 15**2) <= 1
                img[tumor_mask] = np.random.normal(200, 12, (224, 224)).clip(170, 240).astype(np.uint8)[tumor_mask]
            
            # Save as valid RGB JPEG
            pil_img = Image.fromarray(img).convert("RGB")
            pil_img.save(cls_dir / f"sample_{cls}_{i+1:03d}.jpg", "JPEG", quality=95)
    
    print(f"[SUCCESS] Synthetic dataset created at '{raw_dir}'\n")
    return raw_dir


def print_dataset_summary(summary_counts):
    """Print the exact formatted dataset summary table required."""
    print("\nDataset Summary")
    print("-" * 38)
    print(f"{'Class':<14} {'Train':<7} {'Val':<7} {'Test':<7}")
    
    tot_train, tot_val, tot_test = 0, 0, 0
    for cls in CLASSES:
        tr = summary_counts[cls]["train"]
        vl = summary_counts[cls]["val"]
        ts = summary_counts[cls]["test"]
        tot_train += tr
        tot_val += vl
        tot_test += ts
        print(f"{CLASS_DISPLAY[cls]:<14} {tr:<7} {vl:<7} {ts:<7}")
        
    print("-" * 38)
    print(f"{'Total':<14} {tot_train:<7} {tot_val:<7} {tot_test:<7}\n")


def prepare_dataset(source_dir: Path, target_dir: Path, seed: int = 42, 
                    train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split dataset into 70% Train, 15% Val, 15% Test with deduplication & validation."""
    random.seed(seed)
    np.random.seed(seed)
    
    train_dir = target_dir / "Training"
    val_dir = target_dir / "Validation"
    test_dir = target_dir / "Testing"
    
    # Check if target already has all 3 splits populated
    existing_splits = all((target_dir / split / cls).exists() and 
                          len(list((target_dir / split / cls).glob("*"))) > 0 
                          for split in ["Training", "Validation", "Testing"] 
                          for cls in CLASSES)
    
    if existing_splits and not source_dir.resolve().samefile(target_dir.resolve()):
        print("[INFO] Existing Training, Validation, and Testing directories detected in target.")
        print("[INFO] Validating existing splits without duplicating files...")
        summary = {cls: {"train": 0, "val": 0, "test": 0} for cls in CLASSES}
        for cls in CLASSES:
            summary[cls]["train"] = len([f for f in (train_dir / cls).glob("*") if f.suffix.lower() in VALID_EXTENSIONS])
            summary[cls]["val"] = len([f for f in (val_dir / cls).glob("*") if f.suffix.lower() in VALID_EXTENSIONS])
            summary[cls]["test"] = len([f for f in (test_dir / cls).glob("*") if f.suffix.lower() in VALID_EXTENSIONS])
        print_dataset_summary(summary)
        return

    # Create directory structure
    for split_dir in [train_dir, val_dir, test_dir]:
        for cls in CLASSES:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)
            
    print(f"[INFO] Scanning images from: {source_dir}")
    seen_hashes = set()
    corrupted_count = 0
    duplicate_count = 0
    unsupported_count = 0
    
    summary_counts = {cls: {"train": 0, "val": 0, "test": 0} for cls in CLASSES}
    class_totals = {}

    for cls in CLASSES:
        cls_source = source_dir / cls
        if not cls_source.exists():
            # Try case-insensitive matching
            matched = [d for d in source_dir.iterdir() if d.is_dir() and d.name.lower() == cls.lower()]
            if matched:
                cls_source = matched[0]
            else:
                print(f"[WARNING] Missing class folder for '{cls}' in '{source_dir}'")
                class_totals[cls] = []
                continue

        valid_files = []
        for file_path in cls_source.rglob("*"):
            if not file_path.is_file():
                continue
            
            ext = file_path.suffix.lower()
            if ext not in VALID_EXTENSIONS:
                unsupported_count += 1
                continue
                
            if not is_valid_image(file_path):
                corrupted_count += 1
                continue
                
            file_hash = compute_file_hash(file_path)
            if file_hash in seen_hashes:
                duplicate_count += 1
                continue
                
            seen_hashes.add(file_hash)
            valid_files.append(file_path)
            
        class_totals[cls] = valid_files
        print(f"  • Class '{CLASS_DISPLAY[cls]}': Found {len(valid_files)} valid unique images")

    # Audit reports
    if corrupted_count > 0:
        print(f"[AUDIT] Excluded {corrupted_count} corrupted / unreadable images.")
    if duplicate_count > 0:
        print(f"[AUDIT] Excluded {duplicate_count} exact duplicate images.")
    if unsupported_count > 0:
        print(f"[AUDIT] Skipped {unsupported_count} files with unsupported extensions.")
        
    # Check class imbalance
    counts = [len(f) for f in class_totals.values()]
    if any(c == 0 for c in counts):
        print("\n[ERROR] One or more classes have 0 images. Cannot complete split.")
        return

    min_c, max_c = min(counts), max(counts)
    if max_c > 2 * min_c:
        print(f"\n[NOTICE] Class imbalance detected (Min: {min_c}, Max: {max_c}). Model training may benefit from class weights.")

    # Perform stratified split
    for cls, files in class_totals.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Ensure test gets the remaining files to avoid rounding loss
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        
        # Verify isolation: sets must be completely disjoint
        assert len(set(train_files).intersection(set(test_files))) == 0, "Test set contaminated with training samples!"
        assert len(set(val_files).intersection(set(test_files))) == 0, "Test set contaminated with validation samples!"
        
        summary_counts[cls]["train"] = len(train_files)
        summary_counts[cls]["val"] = len(val_files)
        summary_counts[cls]["test"] = len(test_files)
        
        # Copy to destinations
        for f in train_files:
            dest = train_dir / cls / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
        for f in val_files:
            dest = val_dir / cls / f.name
            if not dest.exists():
                shutil.copy2(f, dest)
        for f in test_files:
            dest = test_dir / cls / f.name
            if not dest.exists():
                shutil.copy2(f, dest)

    print_dataset_summary(summary_counts)
    print(f"[SUCCESS] Dataset successfully partitioned into: 70% Train, 15% Val, 15% Test (Seed={seed})")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate Brain Tumor MRI dataset into 70/15/15 splits.")
    parser.add_argument("--source_dir", type=str, default="data/raw", help="Path to raw unsplit data directory")
    parser.add_argument("--target_dir", type=str, default="data", help="Output directory for Training/Validation/Testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--create_samples", action="store_true", help="Generate synthetic MRI scans if no raw data is present")
    
    args = parser.parse_args()
    target_path = Path(args.target_dir)
    source_path = Path(args.source_dir)
    
    # Check if source exists or alternative paths exist
    if not source_path.exists():
        # Check if user already placed data in data/Training without splits
        alt_raw = Path("data/Training")
        if alt_raw.exists() and all((alt_raw / c).exists() for c in CLASSES) and not (target_path / "Testing").exists():
            print(f"[INFO] Using existing '{alt_raw}' as raw source directory.")
            source_path = alt_raw
        else:
            print(f"[WARNING] Source directory '{source_path}' does not exist.")
            # Automatically create synthetic samples if requested or empty
            source_path = generate_synthetic_samples(target_path, samples_per_class=30)
            
    prepare_dataset(source_path, target_path, seed=args.seed)


if __name__ == "__main__":
    main()
