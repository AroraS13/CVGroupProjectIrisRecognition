"""
PerformanceEvaluation - CRR for identification (L1, L2, Cosine); ROC curve for verification.
Outputs Table 3 & Fig. 10, Table 4 & Fig. 11 (per Ma et al. paper).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# IrisMatching contains functions that PerformanceEvaluation can reuse
from IrisMatching import predict, get_cosine_scores, get_class_centers, get_cosine_distance

# directory for the output figures
OUTPUT = "figures"


def _get_crr(true_labels, predicted_labels):
    """
    Calculate and return the correct recognition rate as a percentage
    (the proportion of test samples whose predicted label matches the true label).

    Parameters:
    - true_labels (list): true subject id per test sample
    - predicted_labels (list): predicted ids from the predictor (functions specifically as a classifier)
    """
    # combine the true and predicted labels together into pairs, then keeps a count of how many pairs actually match
    correct = 0
    for t, p in zip(true_labels, predicted_labels):
        if t == p:
            correct += 1
    # return the percentage of the proportion that actually matches (classification accuracy)
    return 100.0 * correct / len(true_labels)


def create_crr_table(train_vectors, train_labels, test_vectors, test_labels, predictions):
    """
    Table 3:
    Computes and prints the CRR under L1, L2, and cosine distance
    for both the 1536-dim features and the LDA-reduced features.

    Parameters:
    - train_vectors (np.ndarray): raw training features
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): raw test features
    - test_labels (list): true subject id per test sample
    - predictions (dict): predictions in LDA-reduced space {"l1": [...], "l2": [...], "cosine": [...]}
    """
    # compute the CRR for each distance measure method via predictions made in LDA-reduced space
    crr_l1_reduced = _get_crr(test_labels, predictions["l1"])
    crr_l2_reduced = _get_crr(test_labels, predictions["l2"])
    crr_cosine_reduced = _get_crr(test_labels, predictions["cosine"])

    # use helper functions from IrisMatching to predict/classify in the raw feature space without LDA (1536-dim)
    # step 1: build class centers --> get the mean vector per subject from the projected training vectors 
    #                                 & extract the labels and align them with their corresponding centers so they can be iterated together
    class_centers = get_class_centers(train_vectors, train_labels)
    center_labels = list(class_centers.keys())
    center_matrix = np.array([class_centers[l] for l in center_labels])
    predictions = {"l1": [], "l2": [], "cosine": []}  # accumulate predicted labels for each distance measure
    for test_vector in test_vectors:
        # for each test image's projected feature vector, keep track of the best distance for each measure (start with "empty" values)
        best_l1 = (None, float("inf"))
        best_l2 = (None, float("inf"))
        best_cosine = (None, float("inf"))
        for label, center in zip(center_labels, center_matrix):
            # for each class center and its subject id label, compute the distance between the current test vector and the current class center for all methods
            d1 = np.sum(np.abs(test_vector - center))  # l1, manhattan
            d2 = np.sqrt(np.sum((test_vector - center) ** 2))  # l2, euclidean
            d3 = get_cosine_distance(test_vector, center)  # cosine
            # update the best match for each measure if needed
            if d1 < best_l1[1]:
                best_l1 = (label, d1)
            if d2 < best_l2[1]:
                best_l2 = (label, d2)
            if d3 < best_cosine[1]:
                best_cosine = (label, d3)
        # append the winning label for each measure to the predictions list
        predictions["l1"].append(best_l1[0])
        predictions["l2"].append(best_l2[0])
        predictions["cosine"].append(best_cosine[0])
    # now after repeating essentially the same construction process for vectors in the original space, also compute the CRR for each distance measure method via the new predictions
    crr_l1_og     = _get_crr(test_labels, predictions["l1"])
    crr_l2_og     = _get_crr(test_labels, predictions["l2"])
    crr_cosine_og = _get_crr(test_labels, predictions["cosine"])

    # print table 3 from the paper with all of the crr information (original space vs reduced space)
    print("\nTable 3: Correct Recognition Rate (%) by Similarity Measure")
    print("-" * 67)
    print(f"{'Similarity Measure':<25} {'Original Features':>20} {'Reduced Features':>20}")
    print("-" * 67)
    print(f"{'L1 distance':<25} {crr_l1_og:>20.2f} {crr_l1_reduced:>20.2f}")
    print(f"{'L2 distance':<25} {crr_l2_og:>20.2f} {crr_l2_reduced:>20.2f}")
    print(f"{'Cosine similarity':<25} {crr_cosine_og:>20.2f} {crr_cosine_reduced:>20.2f}")


def create_crr_dimensions_plot(train_vectors, train_labels, test_vectors, test_labels):
    """
    Figure 10:
    Create a plot that compares CRR cosine results using features of different LDA dimensions.
    The curve shows how recognition accuracy improves and eventually plateaus as more LDA dimensions are retained.

    Parameters:
    - train_vectors (np.ndarray): raw training features
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): raw test features
    - test_labels (list): true subject id per test sample
    """
    # figure out the max number of dimensions to visualize (same layout as the paper)
    max_components = min(220, len(set(train_labels)) - 1)
    # maintain a list of dimensions tested from 40 to the max, in steps of 20
    dim_range = list(range(40, max_components + 1, 20))
    # ensure the max is included even if it doesn't fall on a step of 20
    if max_components not in dim_range:
        dim_range.append(max_components)

    # prepare to accumulate CRR at each dimension
    crr_values = []
    # fit a fresh LDA model at each dimension and classify the test set using cosine distance --> record the score
    for n_components in dim_range:
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        lda.fit(train_vectors, train_labels)
        predictions = predict(lda, train_vectors, train_labels, test_vectors)
        # use cosine because it performs slightly better than l1 and l2
        crr_values.append(_get_crr(test_labels, predictions["cosine"]))

    # create a plot of the findings and save it to the output directory
    os.makedirs(OUTPUT, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(dim_range, crr_values, marker="o", color="black")
    plt.title("Figure 10: CRR vs Number of LDA Dimensions")
    plt.xlabel("Dimensionality of the feature vector")
    plt.ylabel("Correct recognition rate (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT, "fig10_crr_dimensions.png")
    plt.savefig(path)
    plt.show()


def create_roc_table_and_plot(train_vectors, train_labels, test_vectors, test_labels, lda):
    """
    Table 4:
    FMR and FNMR at three specific threshold values that were used in the paper: 0.446, 0.472, 0.502

    Figure 11:
    Plot the ROC curve (FMR vs FNMR) for verification mode

    Parameters:
    - train_vectors (np.ndarray): raw training features
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): raw test features
    - test_labels (list): true subject id per test sample
    - lda (LinearDiscriminantAnalysis): fitted LDA model
    """
    # obtain actual and assumed score distributions (cosine distances) & utilize many thresholds to compute FMR and FNMR at each point
    actual_scores, assumed_scores = get_cosine_scores(lda, train_vectors, train_labels, test_vectors, test_labels)
    # a threshold value is used across all possible cosine distance values (use all unique score values as candidate thresholds)
    thresholds = np.linspace(min(actual_scores.min(), assumed_scores.min()), max(actual_scores.max(), assumed_scores.max()), 500)

    # compute FMR and FNMR across a range of thresholds to accumulate content for the ROC curve
    # FMR at t (false match rate at each threshold): fraction of assumed scores below t (the assumed points are incorrectly accepted)
    fmr  = np.array([np.mean(assumed_scores <= t) for t in thresholds])
    # FNMR at t (false non-match rate at each threshold): fraction of actual scores above t (the actual subjects are incorrectly rejected)
    fnmr = np.array([np.mean(actual_scores > t) for t in thresholds])

    # print table 4
    target_thresholds = [0.446, 0.472, 0.502]
    print("\nTable 4: FMR and FNMR at Threshold Values")
    print(f"{'Threshold':<15} {'FMR (%)':>12} {'FNMR (%)':>12}")
    print("-" * 41)
    for t in target_thresholds:
        # find the index of the closest threshold we computed and print that result
        index = np.argmin(np.abs(thresholds - t))
        print(f"{t:<15.3f} {fmr[index] * 100:>12.4f} {fnmr[index] * 100:>12.4f}")

    # create a plot of the ROC curve and save it to the output directory
    os.makedirs(OUTPUT, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(fmr * 100, fnmr * 100, color="black")
    plt.title("Figure 11: ROC Curve")
    plt.xlabel("False Match Rate (%)")
    plt.ylabel("False Non-Match Rate (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT, "fig11_roc_curve.png")
    plt.savefig(path)
    plt.show()


def evaluate_performance(train_vectors, train_labels, test_vectors, test_labels, lda, predictions):
    """
    Orchestration function of this module.
    Outputs tables 3 + 4 and figures 10 + 11 from the paper.

    Parameters:
    - train_vectors (np.ndarray): raw training features
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): raw test features
    - test_labels (list): true subject id per test sample
    - lda (LinearDiscriminantAnalysis): fitted LDA model
    - predictions (dict): predictions in LDA-reduced space {"l1": [...], "l2": [...], "cosine": [...]}
    """
    # call the functions to create the four outputs
    create_crr_table(train_vectors, train_labels, test_vectors, test_labels, predictions)
    create_crr_dimensions_plot(train_vectors, train_labels, test_vectors, test_labels)
    create_roc_table_and_plot(train_vectors, train_labels, test_vectors, test_labels, lda)


if __name__ == "__main__":
    print("Run this module through IrisRecognition.py")
