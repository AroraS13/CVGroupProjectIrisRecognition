"""
IrisNormalization.py

Normalizes iris region into a rectangular strip given image, pupil raidus, iris radius, pupil center, and desired dimensions of normalized image. 
"""

import numpy as np


def normalize_iris(image, radius_pupil, radius_iris, centroid_x, centroid_y, columns, rows):
    """
    Normalization function: Returns normalized iris image 
    """
    #Create empty array to hold normalized values
    normalized_iris = np.zeros((rows, columns), dtype=np.uint8)
    #In each column
    for i in range(columns):
        #Calculate angle corresponding to column
        angle = 2 * np.pi * i / columns
        #Then, calculate x and y of point corresponding to pupil boundary at that angle
        pupil_x = centroid_x + radius_pupil * np.cos(angle)
        pupil_y = centroid_y + radius_pupil * np.sin(angle)
        #Do the same for iris boundary
        iris_x = centroid_x + radius_iris * np.cos(angle)
        iris_y = centroid_y + radius_iris * np.sin(angle)
        #For each row
        for j in range(rows):
            #Calculate faction between pupil and iris boundary corresponding to row
            fraction = j / (rows - 1)
            #Calculate x and y of point correspondoing to fraction between pupil and iris
            x_sample = (1 - fraction) * pupil_x + fraction * iris_x
            y_sample = (1 - fraction) * pupil_y + fraction * iris_y
            #Round to nearest pixel
            x_pixel = int(round(x_sample))
            y_pixel = int(round(y_sample))
            #If found pixel within bounds of image, add it to normalized_iris
            if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                normalized_iris[j, i] = image[y_pixel, x_pixel]
    #Final np array should contain normalized region, return
    return normalized_iris
