"""
Phase D: Decision (Recognition)
Minimum Distance Classifier

Not in your original module list but split out here rather than bolted onto
descriptors.py, since "describing a blob" and "deciding if it's oil" are
different responsibilities and you'll likely swap this module out first once you have labeled data.
"""

from dataclasses import dataclass

import numpy as np

from src.descriptors import BlobFeatures


@dataclass
class ClassResult:
    label_id: int
    is_oil: bool
    distance_oil: float
    distance_lookalike: float


# Prototype (mean) feature vectors for the two classes, in
# (smoothness, circularity, log_area) space. These are PLACEHOLDER values —
# replace with means computed from your Class_0 / Class_1 labeled chips
# (see pipeline.py's `fit_prototypes` helper) before trusting the output.
DEFAULT_OIL_PROTOTYPE = np.array([0.85, 0.35, 6.5])
DEFAULT_LOOKALIKE_PROTOTYPE = np.array([0.40, 0.75, 5.5])


def _feature_vector(f: BlobFeatures) -> np.ndarray:
    log_area = np.log(f.area + 1.0)
    return np.array([f.smoothness, f.circularity, log_area])


def fit_prototypes(feature_lists: dict[str, list[BlobFeatures]]) -> dict[str, np.ndarray]:
    """
    Build the class prototypes (mean feature vectors) from
    labeled training blobs.

    feature_lists: {"oil": [BlobFeatures, ...], "lookalike": [BlobFeatures, ...]}
    Run this once over your Class_0/Class_1 labeled data (see pipeline.py)
    and hardcode the result, or persist it to metadata/ and load it here.
    """
    prototypes = {}
    for class_name, features in feature_lists.items():
        vectors = np.stack([_feature_vector(f) for f in features])
        prototypes[class_name] = vectors.mean(axis=0)
    return prototypes


def classify_blob(
    f: BlobFeatures,
    oil_prototype: np.ndarray = DEFAULT_OIL_PROTOTYPE,
    lookalike_prototype: np.ndarray = DEFAULT_LOOKALIKE_PROTOTYPE,
    min_area: float = 50.0,
) -> ClassResult:
    """
    Minimum Distance Classifier.
    Assigns the blob to whichever class prototype it's Euclidean-closer to
    in (smoothness, circularity, log_area) feature space. A hard area gate
    is applied first: below min_area, a blob is too small to trust texture
    stats on and is auto-rejected regardless of distance.
    """
    if f.area < min_area:
        return ClassResult(f.label_id, False, float("inf"), 0.0)

    vec = _feature_vector(f)
    d_oil = float(np.linalg.norm(vec - oil_prototype))
    d_lookalike = float(np.linalg.norm(vec - lookalike_prototype))

    return ClassResult(
        label_id=f.label_id,
        is_oil=d_oil < d_lookalike,
        distance_oil=d_oil,
        distance_lookalike=d_lookalike,
    )


def classify_all(features: list[BlobFeatures], **kwargs) -> list[ClassResult]:
    return [classify_blob(f, **kwargs) for f in features]
