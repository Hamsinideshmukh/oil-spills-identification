"""
Phase C: Feature Extraction (Description)
Connected Components
Regional Descriptors (shape + texture)
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlobFeatures:
    label_id: int
    area: float
    perimeter: float
    circularity: float          # shape compactness
    smoothness: float           # statistical texture moment
    mean_intensity: float
    bbox: tuple                 # (x, y, w, h)
    centroid: tuple             # (cx, cy)


def connected_components(mask: np.ndarray, min_area: int = 30) -> list[np.ndarray]:
    """
    Connected Components:
    Labels every separate dark blob in the cleaned mask. Returns a list of
    single-blob binary masks (same shape as input), one per component,
    filtered by a minimum area so 1-2 pixel noise fragments that survived
    morphology don't get treated as candidate spills.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    blobs = []
    for label_id in range(1, num_labels):  # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        blob_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
        blobs.append(blob_mask)

    return blobs


def _statistical_smoothness(gray_region: np.ndarray) -> float:
    """
    Statistical texture moment "smoothness":
        R = 1 - 1 / (1 + sigma^2)
    where sigma^2 is the intensity variance within the region (normalized
    to [0,1] first). R -> 1 for very smooth (low-variance) regions, R -> 0
    for high-variance (rough/textured) regions. Oil spills damp the ocean's
    capillary waves, so they read as SMOOTH (high R); look-alikes made of
    choppy water or algae texture read as rougher (lower R).
    """
    if gray_region.size == 0:
        return 0.0
    normalized = gray_region.astype(np.float32) / 255.0
    variance = np.var(normalized)
    return float(1 - 1 / (1 + variance))


def extract_features(gray_img: np.ndarray, blob_masks: list[np.ndarray]) -> list[BlobFeatures]:
    """
    For each connected-component blob mask, compute the regional
    descriptors used in the final decision rule.
    """
    features = []

    for i, blob_mask in enumerate(blob_masks):
        contours, _ = cv2.findContours(
            blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Circularity: 4*pi*Area / Perimeter^2. A perfect circle = 1.0.
        # Real oil slicks get smeared by currents/wind into elongated
        # shapes, so circularity well below 1 is actually a mild positive
        # signal for oil (vs. round look-alikes), and is used alongside
        # area/smoothness rather than as a lone gatekeeper.
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        cx = moments["m10"] / (moments["m00"] + 1e-6)
        cy = moments["m01"] / (moments["m00"] + 1e-6)

        region = gray_img[y:y + h, x:x + w]
        region_mask = blob_mask[y:y + h, x:x + w]
        masked_region = region[region_mask > 0]

        smoothness = _statistical_smoothness(masked_region)
        mean_intensity = float(np.mean(masked_region)) if masked_region.size else 0.0

        features.append(
            BlobFeatures(
                label_id=i,
                area=float(area),
                perimeter=float(perimeter),
                circularity=float(circularity),
                smoothness=smoothness,
                mean_intensity=mean_intensity,
                bbox=(x, y, w, h),
                centroid=(cx, cy),
            )
        )

    return features
