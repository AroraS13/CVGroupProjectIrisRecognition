"""
IrisRecognition - Main function that orchestrates the full iris recognition pipeline.
Uses IrisLocalization, IrisNormalization, ImageEnhancement, FeatureExtraction, IrisMatching, and PerformanceEvaluation.
"""

import os
import cv2
import numpy as np

# import all functionalities that are stored in the other modules
from dataset_loader import load_dataset
from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhanceImage
from FeatureExtraction import extract_features
from IrisMatching import match_iris
from PerformanceEvaluation import evaluate_performance


def process_dataset(data):
    """
    Executes preprocessing and feature extraction on the dataset and returns a list of tuples (subject_id, image)
    which act as features (for successfully processed images only).

    Parameters:
    - data (list): [(subject_id, image_array), ...] as returned by load_dataset
    """
    # accumulate the transformed (subject_id, feature_vector) tuples
    features = []

    # for each raw (subject_id, feature_vector) pair in the dataset
    for subject_id, image in data:
        # step 1: localization the iris
        pupil_cx, pupil_cy, pupil_radius, iris_radius = localize_iris(image)
        # skip images where the iris boundary weren't detected
        if iris_radius is None:
            continue
        # step 2: normalize the iris ring into a fixed-size rectangular strip
        # 512 = width of normalized iris strip, 64 = height of normalized iris strip (both follow the paper)
        iris_normalized = normalize_iris(image, pupil_radius, iris_radius, pupil_cx, pupil_cy, columns=512, rows=64)
        # step 3: enhance the normalized image via illumination correction + histogram equalization
        _, _, _, iris_enhanced = enhanceImage(iris_normalized)
        # step 4: feature extraction --> 1536D feature vector
        feature_vector = extract_features(iris_enhanced)
        # record the transformed iris for future use
        features.append((subject_id, feature_vector))
    return features


def main():
    # load the dataset
    train_data, test_data = load_dataset("data/CASIA Iris Image Database (version 1)")
    # preprossing + feature extraction
    train_features = process_dataset(train_data)
    test_features  = process_dataset(test_data)
    # create vector and label arrays for training + matching
    train_labels  = [s for s, _ in train_features]  # subject's id (string) for each training sample
    train_vectors = np.array([v for _, v in train_features])  # matrix with one row per training image
    test_labels   = [s for s, _ in test_features]  # for testing only
    test_vectors  = np.array([v for _, v in test_features])  # for testing only
    # iris matching: train the LDA model and classify the test set
    lda, predictions = match_iris(train_vectors, train_labels, test_vectors)
    # performance evaluation
    evaluate_performance(train_vectors, train_labels, test_vectors, test_labels, lda, predictions)


if __name__ == "__main__":
    main()
