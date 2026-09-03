"""
Phase B: Segmentation (Isolating Dark Patches)
Thresholding
"""

import cv2
import numpy as np


def otsu_threshold(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Otsu's Method:
    Finds the global threshold that minimizes intra-class variance between
    two classes (here: "sea" vs "oil/dark patch"). Oil is DARKER than the
    surrounding sea in a SAR backscatter image, so we invert: pixels BELOW
    the threshold are foreground (oil candidates).

    Returns (binary_mask, threshold_value). binary_mask is uint8 {0, 255}.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh_val, mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return mask, thresh_val


def adaptive_threshold(img: np.ndarray, block_size: int = 51, c: int = 5) -> np.ndarray:
    """
    Local Adaptive Thresholding.
    Necessary when illumination/backscatter intensity is not uniform across
    a large SAR swath (e.g. incidence-angle brightness gradients). Each
    pixel is thresholded against the mean of its own local neighborhood
    rather than one global cutoff.

    block_size must be odd. Larger block_size = smoother local threshold
    surface, more tolerant of gradual brightness change; smaller = more
    reactive to local texture.
    """
    if block_size % 2 == 0:
        block_size += 1

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )
    return mask


def segment(img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Dispatch. 'otsu' for roughly uniform-illumination chips (typical for a
    single Sentinel-1 chip), 'adaptive' for chips spanning a large swath
    with visible brightness gradients.
    """
    if method == "otsu":
        mask, _ = otsu_threshold(img)
        return mask
    elif method == "adaptive":
        return adaptive_threshold(img)
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
