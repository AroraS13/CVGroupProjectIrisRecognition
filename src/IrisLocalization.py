import cv2
import numpy as np


def rough_pupil_center(image):
    """Coarse pupil center approximation by finding the column/row with the lowest intensity sum."""
    x = np.argmin(np.sum(image, axis=0))
    y = np.argmin(np.sum(image, axis=1))
    return x, y


def find_largest_connected_component(image):
    """Return the largest connected component in a binary image, or None if none found."""
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(image)
    if retval <= 1:
        return None
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    return (labels == largest_label).astype(np.uint8) * 255


def localize_iris(image):
    """
    Localize the pupil and iris boundaries in a grayscale iris image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale iris image (uint8, 2-D).

    Returns
    -------
    pupil_cx : int
        X coordinate of the pupil center.
    pupil_cy : int
        Y coordinate of the pupil center.
    pupil_radius : int
        Estimated radius of the pupil in pixels.
    iris_radius : int or None
        Estimated radius of the iris in pixels, or None if it could not be found.
    """
    # --- Pupil localization ---
    pupil_x, pupil_y = rough_pupil_center(image)

    # Crop a local window around the rough center
    y0 = max(0, pupil_y - 80)
    y1 = min(image.shape[0], pupil_y + 80)
    x0 = max(0, pupil_x - 80)
    x1 = min(image.shape[1], pupil_x + 80)

    relevant_window = image[y0:y1, x0:x1]

    # Threshold the darkest 5% of pixels to isolate the pupil
    threshold = np.percentile(relevant_window, 5)
    dark_region = (relevant_window <= threshold).astype(np.uint8) * 255
    dark_region = cv2.medianBlur(dark_region, 5)

    pupil_region = find_largest_connected_component(dark_region)
    if pupil_region is None:
        return pupil_x, pupil_y, 0, None

    # Refine the center using image moments
    M = cv2.moments(pupil_region)
    if M["m00"] == 0:
        return pupil_x, pupil_y, 0, None

    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    # Convert back to full-image coordinates
    pupil_cx = x0 + centroid_x
    pupil_cy = y0 + centroid_y

    area = cv2.countNonZero(pupil_region)
    pupil_radius = int(np.sqrt(area / np.pi))

    # --- Iris localization ---
    edges = cv2.Canny(image, 50, 150)
    height, width = edges.shape

    radii = []
    for theta in np.linspace(0, 2 * np.pi, 120, endpoint=False):
        for r in range(int(pupil_radius * 2), int(pupil_radius * 4)):
            x = int(round(pupil_cx + r * np.cos(theta)))
            y = int(round(pupil_cy + r * np.sin(theta)))

            if x < 0 or x >= width or y < 0 or y >= height:
                break

            if edges[y, x] > 0:
                radii.append(r)
                break

    iris_radius = int(np.median(radii)) if radii else None

    return pupil_cx, pupil_cy, pupil_radius, iris_radius
