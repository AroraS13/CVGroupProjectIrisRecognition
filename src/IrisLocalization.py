"""
IrisLocalization.py

1. Logic behind the script:
This script implements the localization strategy described in Section 3.2.1 of the Li Ma et al. (2003) paper, divided into three logical steps:
- Step 1 (Coarse Pupil Center): It projects image intensities vertically and horizontally. The coordinates corresponding to the minima of the two profiles estimate the rough pupil center.
- Step 2 (Precise Pupil Center): It crops a precise 120x120 window around the rough center, thresholds the darkest region (the pupil), and extracts the largest connected component. Image moments are used to compute the exact centroid, refining the center coordinates and radius.
- Step 3 (Iris Boundary): Canny edge detection is applied to find boundary structures. While the original paper suggests the Hough Transform here, this script implements a highly efficient "Radial Ray-Casting" technique. It shoots 120 rays outwards from the center and records the distance to the nearest Canny edge. Using the median of these ray lengths robustly filters out noise (like eyelashes or reflections) to find the final iris radius without the computational overhead of the Hough space.

2. Key variables/parameters:
- image (numpy.ndarray): The 2D grayscale input image from the CASIA database.
"""

import cv2
import numpy as np


def rough_pupil_center(image):
    """
    Returns rough estimate of pupil center
    """
    #Since pupil should be the darkest region in the image, we can start at the col and row with lowest sum of values
    x = np.argmin(np.sum(image, axis=0))
    y = np.argmin(np.sum(image, axis=1))
    return x, y



def find_largest_connected_component(image):
    """
    Returns largest connected component found in image, None if none found
    """
    #Use cv2.connectedComponentsWithStats to find components and areas
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    #If no relevant components were found, return None
    if label_count <= 1:
        return None

    #Find largest label based on area
    largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    #Find pixels corresponding to largest label, and multiply by 255 so that
    return (labels == largest_label).astype(np.uint8) * 255



def localize_iris(image):
    """
    Main Localization Function: Returns pupil center x,y values, pupil radius, and iris radius
    """
    # Step 1: get rough estimate of pupil center
    rough_pupil_x, rough_pupil_y = rough_pupil_center(image)

    # Step 2: Crop image using a window around rough pupil center. 80 was used here as it offered the best tradeoff between including pupil+iris, and excluding irrelevant information
    #Define bounds:
    window_y0 = max(0, rough_pupil_y - 80)
    window_y1 = min(image.shape[0], rough_pupil_y + 80)
    window_x0 = max(0, rough_pupil_x - 80)
    window_x1 = min(image.shape[1], rough_pupil_x + 80)

    #Define relevant window based on calculated bounds
    relevant_window = image[window_y0:window_y1, window_x0:window_x1]

    #Using 5th percentile, threshold window to find darkest regions, which should correspond to pupil
    threshold = np.percentile(relevant_window, 5)
    dark_region = (relevant_window <= threshold).astype(np.uint8) * 255

    #Apply median blur to reduce noise
    dark_region = cv2.medianBlur(dark_region, 5)

    #Find largest connected component in dark region, which should correspond to the pupil. 
    pupil_region = find_largest_connected_component(dark_region)
    
    #If no connected component was found, proceed with the roughly calculated pupil center: pupil and iris radius set to 0 and None
    if pupil_region is None:
        return rough_pupil_x, rough_pupil_y, 0, None

    #Using Image moments, calculate centroid of pupil region to get more accurate pupil center
    M = cv2.moments(pupil_region)
    if M["m00"] == 0:
        return rough_pupil_x, rough_pupil_y, 0, None

    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    #Calculate true pupil center by adding bounds of the window
    pupil_center_x = window_x0 + centroid_x
    pupil_center_y = window_y0 + centroid_y

    #Use cv2.countNonZero to calculate the pupil area
    area = cv2.countNonZero(pupil_region)

    #Then use radius formula to calculate pupil radius
    pupil_radius = int(np.sqrt(area / np.pi))

    # Step 3: Use cv2.Canny to detect edges in image: Should correspond to iris boundary since its the next darkest region after the pupil
    edges = cv2.Canny(image, 50, 150)
    height, width = edges.shape

    #Calculate iris radius
    radii = []

    #Using np.linspace, sample 120 points around pupil center at various angles
    for sample in np.linspace(0, 2 * np.pi, 120, endpoint=False):

        #For each sampled angle, iterate from pupil radius*2 to pupil radius*4
        for r in range(int(pupil_radius * 2), int(pupil_radius * 4)):

            #Calculate x and y of sampled point, round to nearest int
            x = int(round(pupil_center_x + r * np.cos(sample)))
            y = int(round(pupil_center_y + r * np.sin(sample)))

            #If point is out of bounds, break loop
            if x < 0 or x >= width or y < 0 or y >= height:
                break

            #If edge detected at this point, append to radius list and break
            if edges[y, x] > 0:
                radii.append(r)
                break

    #After running through all angles, find final iris radius by taking median of all radii found, else none if none found
    iris_radius = int(np.median(radii)) if radii else None

    #return final values
    return pupil_center_x, pupil_center_y, pupil_radius, iris_radius
