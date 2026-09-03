"""
Phase A -> Pre-Processing (Restoration & Enhancement)

Here, were applying the concepts of Log Transform, Power-Law/Gamma, Speckle filters

SAR speckle is MULTIPLICATIVE noise: I_observed = I_true * N

A Lee filter estimates local statistics (mean/variance) in a sliding window
and blends the noisy pixel toward the local mean based on how "noisy" that
window looks (high local variance -> trust the pixel less).
"""

def adaptive_median_filter(img: np.ndarray, max_window: int = 7) -> np.ndarray:
    """
    Adaptive Median Filter.
    Grows the filter window until the median is a "good" representative of
    the local neighborhood, or max_window is hit. Preserves edges better than
    a fixed-size median filter because flat, noisy regions get a bigger
    window while edges get a small one.

    img: single-channel uint8 or float32 image.
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    padded = np.pad(img, max_window // 2, mode="reflect")
    out = np.zeros_like(img)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            window_size = 3
            while window_size <= max_window:
                half = window_size // 2
                yy, xx = y + max_window // 2, x + max_window // 2
                window = padded[yy - half:yy + half + 1, xx - half:xx + half + 1]

                z_min, z_max = window.min(), window.max()
                z_med = np.median(window)
                z_xy = img[y, x]

                # Level A: is the median itself noise?
                if z_min < z_med < z_max:
                    # Level B: is the center pixel noise?
                    if z_min < z_xy < z_max:
                        out[y, x] = z_xy
                    else:
                        out[y, x] = z_med
                    break
                else:
                    window_size += 2
                    if window_size > max_window:
                        out[y, x] = z_med
            else:
                out[y, x] = z_med

    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median_filter_fast(img: np.ndarray, max_window: int = 7) -> np.ndarray:
    """
    Vectorized approximation of the adaptive median filter using OpenCV's
    medianBlur at increasing kernel sizes, picking per-pixel the smallest
    window whose median lies strictly between the local min and max.
    Much faster than the pure-Python version above; use this in the
    production pipeline (the <500ms latency budget rules out per-pixel
    Python loops on anything but tiny chips).
    """
    img = img.astype(np.uint8)
    sizes = list(range(3, max_window + 1, 2))
    result = None
    resolved = np.zeros(img.shape, dtype=bool)

    for k in sizes:
        med = cv2.medianBlur(img, k)
        local_min = cv2.erode(img, np.ones((k, k), np.uint8))
        local_max = cv2.dilate(img, np.ones((k, k), np.uint8))

        is_valid = (local_min < med) & (med < local_max)
        newly_resolved = is_valid & (~resolved)

        if result is None:
            result = med.copy()
        result[newly_resolved] = med[newly_resolved]
        resolved |= newly_resolved

    # anything never resolved gets the largest-window median as a fallback
    result[~resolved] = med[~resolved]
    return result


def lee_filter(img: np.ndarray, window_size: int = 7) -> np.ndarray:
    """
    Lee Filter (a locally-adaptive Wiener-style filter for
    MULTIPLICATIVE speckle noise). Standard SAR despeckling filter.

    output(x,y) = mean + W * (pixel(x,y) - mean)
    W = var_local / (var_local + var_noise)

    Where the local window is noisy/flat (low local variance relative to
    noise variance) -> W is small -> pixel is pulled toward the local mean
    (smoothing). Where the window has real structure (an edge) -> W is
    large -> the original pixel is preserved.
    """
    img = img.astype(np.float32)
    mean = cv2.boxFilter(img, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(img * img, -1, (window_size, window_size))
    var_local = mean_sq - mean * mean

    # Estimate the overall (multiplicative) noise variance from the image
    overall_mean = np.mean(img)
    overall_var = np.var(img)
    noise_var = overall_var / (overall_mean ** 2 + 1e-6) * (overall_mean ** 2)
    noise_var = max(noise_var, 1e-6)

    weight = var_local / (var_local + noise_var + 1e-6)
    weight = np.clip(weight, 0, 1)

    out = mean + weight * (img - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def log_transform(img: np.ndarray) -> np.ndarray:
    """
    Log Transformation.
    Converts multiplicative speckle noise into additive noise:
        log(I_true * N) = log(I_true) + log(N)
    which is much easier for a linear filter to remove afterward.
    Also compresses the dynamic range, which helps because oil spills
    (dark, low-value pixels) get relatively expanded.
    """
    img = img.astype(np.float32) + 1.0  # avoid log(0)
    c = 255.0 / np.log(1 + img.max())
    out = c * np.log(img)
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_transform(img: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Power-Law (Gamma) Transform.
    gamma > 1 darkens mid/low tones further, expanding contrast in the
    dark region where the oil spill lives,without the noise amplification
    that histogram equalization would introduce.
    """
    normalized = img.astype(np.float32) / 255.0
    out = np.power(normalized, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(img: np.ndarray, use_fast_filter: bool = True) -> np.ndarray:
    """Phase A pipeline: log -> despeckle -> gamma."""
    logged = log_transform(img)
    despeckled = (
        adaptive_median_filter_fast(logged)
        if use_fast_filter
        else lee_filter(logged)
    )
    contrasted = gamma_transform(despeckled, gamma=1.5)
    return contrasted
