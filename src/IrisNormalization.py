"""
IrisNormalization - Mapping the iris from Cartesian coordinates to polar coordinates.
"""

import numpy as np


def normalize_iris(image, radius_pupil, radius_iris, centroid_x, centroid_y, columns, rows):
    normalized_iris = np.zeros((rows, columns), dtype=np.uint8)

    for angle_index in range(columns):
        angle = 2 * np.pi * angle_index / columns

        pupil_x = centroid_x + radius_pupil * np.cos(angle)
        pupil_y = centroid_y + radius_pupil * np.sin(angle)

        iris_x = centroid_x + radius_iris * np.cos(angle)
        iris_y = centroid_y + radius_iris * np.sin(angle)

        for radial_index in range(rows):
            fraction = radial_index / (rows - 1)

            x_sample = (1 - fraction) * pupil_x + fraction * iris_x
            y_sample = (1 - fraction) * pupil_y + fraction * iris_y

            x_pixel = int(round(x_sample))
            y_pixel = int(round(y_sample))

            if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                normalized_iris[radial_index, angle_index] = image[y_pixel, x_pixel]

    return normalized_iris
