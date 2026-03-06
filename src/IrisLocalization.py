"""
IrisLocalization.py

Locates the pupil (inner boundary) and iris (outer boundary) in a grayscale
iris image.

Method follows Ma et al. 2003, Section 3.2.1:
  1. Project image vertically and horizontally; minima give a coarse pupil
     center (pupil is darker than surroundings).
  2. Binarize a 120x120 region around that center using a histogram-based
     threshold; the centroid of the dark region refines the pupil center.
     Repeat this refinement step twice for a better estimate.
  3. Run Canny edge detection and Hough circle transform in a constrained
     region to precisely fit both the pupil circle and the iris circle.

Returns: (pupil_x, pupil_y, pupil_r, iris_r) — all in pixels.
"""

import cv2
import numpy as np


# ── Step 1 ────────────────────────────────────────────────────────────────────

def _coarse_pupil_center(image):
    """
    Estimate pupil center from horizontal and vertical projection profiles.
    The column/row with the smallest total intensity corresponds to the pupil.
    """
    col_sum = np.sum(image, axis=0)   # sum each column
    row_sum = np.sum(image, axis=1)   # sum each row
    cx = int(np.argmin(col_sum))
    cy = int(np.argmin(row_sum))
    return cx, cy


# ── Step 2 ────────────────────────────────────────────────────────────────────

def _refine_pupil_center(image, cx, cy):
    """
    Binarize a 120x120 region centered at (cx, cy).
    The centroid of the dark (pupil) binary region is a more accurate center.
    Also estimates pupil radius from the area of the binary blob.

    Returns: refined (cx, cy, pupil_r).
    """
    h, w = image.shape
    half = 60  # half of 120

    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)

    region = image[y1:y2, x1:x2]

    # Adaptive threshold: lower 30th percentile separates dark pupil from iris
    thresh = int(np.percentile(region, 30))
    thresh = max(thresh, 50)  # guard against very dark images

    _, binary = cv2.threshold(region, thresh, 255, cv2.THRESH_BINARY_INV)

    # Remove small noise blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    M = cv2.moments(binary)
    if M["m00"] == 0:
        return cx, cy, 30  # fallback if no dark region found

    refined_cx = int(M["m10"] / M["m00"]) + x1
    refined_cy = int(M["m01"] / M["m00"]) + y1
    pupil_r = int(np.sqrt(M["m00"] / np.pi))

    return refined_cx, refined_cy, pupil_r


# ── Step 3 ────────────────────────────────────────────────────────────────────

def _hough_circle(image, cx, cy, r_min, r_max):
    """
    Run Canny + HoughCircles in a window around (cx, cy) to fit a circle
    whose radius is expected to be in [r_min, r_max].

    Returns: (x, y, r) of the best-fitting circle (global image coordinates).
    Falls back to (cx, cy, midpoint) if no circle is detected.
    """
    h, w = image.shape
    margin = r_max + 30

    x1 = max(0, cx - margin)
    x2 = min(w, cx + margin)
    y1 = max(0, cy - margin)
    y2 = min(h, cy + margin)

    region = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, (7, 7), 1.5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=max(region.shape) // 2,  # expect only one circle
        param1=50,   # upper Canny threshold
        param2=30,   # accumulator vote threshold (lower = more permissive)
        minRadius=r_min,
        maxRadius=r_max,
    )

    if circles is not None:
        x, y, r = np.round(circles[0, 0]).astype(int)
        return int(x + x1), int(y + y1), int(r)

    # No circle found: return the center estimate with a mid-range radius
    return cx, cy, (r_min + r_max) // 2


# ── Public API ─────────────────────────────────────────────────────────────────

def localize_iris(image):
    """
    Localize pupil and iris boundaries in a grayscale iris image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image (320x280 for CASIA-V1).

    Returns
    -------
    (pupil_x, pupil_y, pupil_r, iris_r) : tuple[int, int, int, int]
        - (pupil_x, pupil_y): pupil center in pixels
        - pupil_r: pupil radius in pixels
        - iris_r:  outer iris radius in pixels
    """
    # Step 1: coarse center from projection profiles
    cx, cy = _coarse_pupil_center(image)

    # Step 2: refine center twice (paper explicitly states "we perform the
    # second step twice for a reasonably accurate estimate")
    cx, cy, pupil_r = _refine_pupil_center(image, cx, cy)
    cx, cy, pupil_r = _refine_pupil_center(image, cx, cy)

    # Step 3: precise boundary fitting with Canny + Hough
    # Pupil radius range for CASIA-V1: ~20–80 px
    pupil_x, pupil_y, pupil_r = _hough_circle(image, cx, cy, r_min=20, r_max=80)

    # Iris outer boundary: search from just outside the pupil to ~150 px
    # (paper: iris diameter > 200 px, so radius > 100 px)
    _, _, iris_r = _hough_circle(image, pupil_x, pupil_y,
                                  r_min=pupil_r + 20, r_max=150)

    return pupil_x, pupil_y, pupil_r, iris_r


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.dataset_loader import load_dataset

    root = sys.argv[1] if len(sys.argv) > 1 else \
        "./data/CASIA Iris Image Database (version 1"

    train, _ = load_dataset(root)
    subject_id, image = train[0]

    px, py, pr, ir = localize_iris(image)
    print(f"Subject: {subject_id}")
    print(f"Pupil  : center=({px}, {py}), radius={pr}px")
    print(f"Iris   : radius={ir}px")

    # Draw and save a debug image
    debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(debug, (px, py), pr, (0, 255, 0), 2)   # pupil  — green
    cv2.circle(debug, (px, py), ir, (0, 0, 255), 2)   # iris   — red
    cv2.circle(debug, (px, py), 2,  (255, 0, 0), 3)   # center — blue
    cv2.imwrite("figures/localization_debug.png", debug)
    print("Debug image saved to figures/localization_debug.png")
