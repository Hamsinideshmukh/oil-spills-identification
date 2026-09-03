"""
Phase B continued-> Morphological Cleaning
Here we perofrm , Morphological Operations
"""

import cv2
import numpy as np


def clean_mask(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 9,
) -> np.ndarray:
    """
    Opening then Closing.

    Opening (erosion -> dilation): strips away small, isolated white
    specks left by residual speckle noise that survived Phase A. These are
    false-positive candidates, not real oil.

    Closing (dilation -> erosion): bridges small gaps/holes WITHIN a single
    real oil patch, so one spill isn't fragmented into several small blobs
    at the connected-components stage.

    Kernel sizes are separate knobs on purpose: open_kernel should be small
    (just enough to kill noise specks), close_kernel can be larger since
    a real oil spill is a much bigger structure than the noise you're
    removing.
    """
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_k)
    return closed
