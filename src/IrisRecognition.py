"""
IrisRecognition - Main function that orchestrates the full iris recognition pipeline.
Uses IrisLocalization, IrisNormalization, ImageEnhancement, FeatureExtraction, IrisMatching, and PerformanceEvaluation.
"""

import os
import cv2
import numpy as np

# import all functionalities that are stored in the other modules
from IrisLocalization import localize_iris
from IrisNormalization import normalize_iris
from ImageEnhancement import enhanceImage
from FeatureExtraction import extract_features
from IrisMatching import match_iris
from PerformanceEvaluation import evaluate_performance

def load_dataset(root_dir):
    """
    Loads dataset from root_dir. Returns test and train data as lists
    This script traverses the hierarchical directory structure of the CASIA Iris Image Database to load images into memory. The CASIA database is organized sequentially: Root -> Subject ID -> Session ID. 
    The outer loop iterates through every subject directory (e.g., "001" to "108"). 
    The inner loop explicitly checks for session folders "1" and "2". Images from session "1" are historically captured earlier and are appended to the training set. Images from session "2" are captured later and are appended to the testing set. OpenCV is used to load valid ".bmp" files directly as grayscale matrices, ensuring they are ready for the preprocessing pipeline.

    Parameters:
    - root_dir (str): The path to the root folder containing all subject directories (e.g., 'data/CASIA Iris Image Database/').

    Returns:
    - train_data (list): Accumulates training samples from Session 1. Formatted as a list of tuples: [(subject_id, image_array), ...].
    - test_data (list): Accumulates testing samples from Session 2. Formatted similarly to train_data.
    """
    train_data = []
    test_data = []

    #Probes root directory for subject folders, and then sorts based on subject id
    for subject_id in sorted(os.listdir(root_dir)):

        #For every subject folder, look for session folders(1 or 2)
        sub_path = os.path.join(root_dir, subject_id)

        if not os.path.isdir(sub_path):
            continue

        #For every session type
        for session in ["1", "2"]:
            sess_path = os.path.join(sub_path, session)
            
            #If session folder is not present, skip 
            if not os.path.isdir(sess_path):
                continue
            
            #For every image found in session folder
            for filename in sorted(os.listdir(sess_path)):

                #If file is not .bmp, skip
                if not filename.lower().endswith(".bmp"):
                    continue
                
                #Create image path
                img_path = os.path.join(sess_path, filename)

                #Read in image as grayscale
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                #If load image fails, skip
                if image is None:
                    print(f"Warning: failed to load {img_path}")
                    continue
                
                #If image is found in session 1, add to train set, test set otherwise
                if session == "1":
                    train_data.append((subject_id, image))
                else:
                    test_data.append((subject_id, image))

    #Return sets
    return train_data, test_data


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
        pupil_cx, pupil_cy, pupil_radius, iris_radius = localize_iris(image, window_size=120, percentile_threshold=5)
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