import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.filters import preprocess
from src.segment import segment
from src.morphology import clean_mask
from src.descriptors import connected_components, extract_features
from src.classify import (
    classify_all,
    fit_prototypes,
    DEFAULT_OIL_PROTOTYPE,
    DEFAULT_LOOKALIKE_PROTOTYPE,
)

ROOT = Path("~/PycharmProjects/PythonProject").expanduser()
DATA_DIR = Path("~/PycharmProjects/PythonProject/data_dir/data").expanduser()
CLASS_OIL_DIR = Path("~/PycharmProjects/PythonProject/data_dir/data/Class_1").expanduser()
CLASS_NONOIL_DIR = Path("~/PycharmProjects/PythonProject/data_dir/data/Class_0").expanduser()
SAMPLE_DIR = Path("~/PycharmProjects/PythonProject/data_dir/data/sample").expanduser()
METADATA_DIR = Path("~/PycharmProjects/PythonProject/data_dir/metadata").expanduser()
PROTOTYPE_FILE = METADATA_DIR / "prototypes.json"
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def run_on_chip(img: np.ndarray, seg_method: str = "otsu") -> dict:
    """
    Full Phase A -> D pipeline for a single grayscale SAR chip.
    Returns a result dict with the binary decision, IoU-ready mask, and
    per-blob feature/classification detail for debugging.
    """
    t0 = time.perf_counter()

    pre = preprocess(img)                       # Phase A
    mask = segment(pre, method=seg_method)       # Phase B (threshold)
    mask = clean_mask(mask)                      # Phase B (morphology)
    blobs = connected_components(mask)           # Phase C.1
    features = extract_features(pre, blobs)      # Phase C.2
    results = classify_all(features)             # Phase D

    elapsed_ms = (time.perf_counter() - t0) * 1000

    is_oil_detected = any(r.is_oil for r in results)

    # Build a combined mask of only the blobs classified as oil, for IoU
    # comparison against ground truth.
    final_mask = np.zeros_like(mask)
    for blob_mask, r in zip(blobs, results):
        if r.is_oil:
            final_mask = cv2.bitwise_or(final_mask, blob_mask)

    return {
        "is_oil_detected": is_oil_detected,
        "num_candidate_blobs": len(blobs),
        "num_oil_blobs": sum(r.is_oil for r in results),
        "processing_ms": round(elapsed_ms, 2),
        "final_mask": final_mask,
        "preprocessed": pre,
        "raw_mask": mask,
        "features": features,
        "classifications": results,
    }


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Ch success metric: Intersection over Union against ground truth."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / union) if union > 0 else 0.0


def fit_and_save_prototypes():
    """
    Reads labeled chips from Class_0 (non-oil / look-alike) and Class_1
    (oil), extracts blob features from each
    """
    feature_lists = {"lookalike": [], "oil": []}

    for class_name, folder in [("lookalike", CLASS_NONOIL_DIR), ("oil", CLASS_OIL_DIR)]:
        if not folder.exists():
            print(f"[fit] skipping missing folder: {folder}")
            continue
        for img_path in folder.iterdir():
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            img = load_grayscale(img_path)
            pre = preprocess(img)
            mask = clean_mask(segment(pre))
            blobs = connected_components(mask)
            feats = extract_features(pre, blobs)
            if feats:
                # take the largest blob per chip as the representative one
                largest = max(feats, key=lambda f: f.area)
                feature_lists[class_name].append(largest)

    if not feature_lists["oil"] or not feature_lists["lookalike"]:
        print("[fit] not enough labeled data found — keeping default prototypes.")
        return

    prototypes = fit_prototypes(feature_lists)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROTOTYPE_FILE, "w") as f:
        json.dump({k: v.tolist() for k, v in prototypes.items()}, f, indent=2)
    print(f"[fit] saved prototypes to {PROTOTYPE_FILE}")


def load_prototypes():
    if PROTOTYPE_FILE.exists():
        with open(PROTOTYPE_FILE) as f:
            data = json.load(f)
        return np.array(data["oil"]), np.array(data["lookalike"])
    return DEFAULT_OIL_PROTOTYPE, DEFAULT_LOOKALIKE_PROTOTYPE


def process_folder(folder: Path):
    oil_proto, lookalike_proto = load_prototypes()
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        img = load_grayscale(img_path)
        result = run_on_chip(img)
        # re-classify with loaded (possibly fitted) prototypes
        results = classify_all(
            result["features"],
            oil_prototype=oil_proto,
            lookalike_prototype=lookalike_proto,
        )
        is_oil = any(r.is_oil for r in results)
        print(
            f"{img_path.name:30s} "
            f"oil={is_oil!s:5s} "
            f"blobs={result['num_candidate_blobs']:2d} "
            f"time={result['processing_ms']:6.1f}ms"
        )


def process_single_image(path: Path, out_dir: Path = ROOT / "debug_out"):
    img = load_grayscale(path)
    result = run_on_chip(img)

    out_dir.mkdir(exist_ok=True)
    stem = path.stem
    cv2.imwrite(str(out_dir / f"{stem}_preprocessed.png"), result["preprocessed"])
    cv2.imwrite(str(out_dir / f"{stem}_raw_mask.png"), result["raw_mask"])
    cv2.imwrite(str(out_dir / f"{stem}_final_mask.png"), result["final_mask"])

    print(json.dumps({
        "image": str(path),
        "is_oil_detected": result["is_oil_detected"],
        "num_candidate_blobs": result["num_candidate_blobs"],
        "num_oil_blobs": result["num_oil_blobs"],
        "processing_ms": result["processing_ms"],
    }, indent=2))
    print(f"Debug images written to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="SAR Oil Spill Detection Pipeline")
    parser.add_argument("--fit", action="store_true", help="fit classifier prototypes from Class_0/Class_1")
    parser.add_argument("--image", type=str, default=None, help="process a single image and save debug output")
    args = parser.parse_args()

    if args.fit:
        fit_and_save_prototypes()
        return

    if args.image:
        process_single_image(Path(args.image))
        return

    if not SAMPLE_DIR.exists() or not any(SAMPLE_DIR.iterdir()):
        print(f"No images found in {SAMPLE_DIR}. Pass --image <path> to test on a single chip.")
        return

    process_folder(SAMPLE_DIR)


if __name__ == "__main__":
    main()
