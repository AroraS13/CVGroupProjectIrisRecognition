"""
IrisMatching - Fisher linear discriminant for dimension reduction and nearest center classifier for classification.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# the number of LDA (linear discriminant analysis) dimensions to use (200 is the sweet spot where accuracy plateaus)
N_COMPONENTS = 200


def _get_class_centers(projected_train, train_labels):
    """
    Helper function for predict() and get_cosine_scores().
    Computes the mean feature vector (center) for each class in the projected space
    and returns them stored as a dictionary where {subject_id: mean_vector}.
 
    Parameters:
    - projected_train (np.ndarray): training feature matrix after LDA projection
    - train_labels (list): subject id string per training image
    """
    # maps subject_id --> mean feature vector of all its training samples in the space
    class_centers = {}
    # sorted() ensures deterministic key ordering
    for label in sorted(set(train_labels)):
        # for each unique subject id, find all row indices in projected_train that belong to that subject
        indices = [i for i, l in enumerate(train_labels) if l == label]
        # take these rows to compute their mean vector --> subject's class center
        class_centers[label] = np.mean(projected_train[indices], axis=0)
    return class_centers


def _get_cosine_distance(a, b):
    """
    Helper function for predict() and get_cosine_scores().
    Calculate the cosine distance (1 - cosine similarity) between the two feature vectors, see d3 in equation 8.
    Note: the smaller the better match --> consistent with d1 and d2.

    Parameters:
    - a (np.ndarray): projected test vector 
    - b (np.ndarray): class center
    """
    # compute the magnitude/length of each vector
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    # prevent division by zero
    if a_norm == 0 or b_norm == 0:
        return 1.0
    # compute the dot product and normalize to get the cosine similarity; subtracting converts it to a distance (0: same direction, 2: opposite)
    return 1.0 - np.dot(a, b) / (a_norm * b_norm)


def get_cosine_scores(lda, train_vectors, train_labels, test_vectors, test_labels):
    """
    For ease of PerformanceEvaluation.py implementation, compute and return genuine and impostor cosine distance scores for each test sample for the ROC curve
    --> compute FMR (how often an impostor is accepted) and FNMR (how often a genuine is rejected)
    Genuine score: cosine distance to its own true class center.
    Impostor score: cosine distance to the nearest class center that doesn't belong to its true class.
  
    Parameters:
    - lda (LinearDiscriminantAnalysis): fitted LDA model
    - train_vectors (np.ndarray): matrix with one row per training image
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): matrix with one row per testing image
    - test_labels (list): true subject id per test sample
    """
    # project feature matrices into the reduced space and compute the mean vector per subject
    projected_train_vectors = lda.transform(train_vectors)
    projected_test_vectors = lda.transform(test_vectors)
    class_centers = _get_class_centers(projected_train_vectors, train_labels)
    # record genuine pair and impostor pair distances
    genuine_scores = []
    impostor_scores = []
 
    for test_vector, true_label in zip(projected_test_vectors, test_labels):
        # for each test image paired with its true subject id...
        genuine  = None
        impostor = float("inf")
        for label, center in class_centers.items():
            # for each of the pair's class centers, compute the cosine distance from the current test vector to that center
            d = _get_cosine_distance(test_vector, center)
            # true_label: the actual subject id of the test image
            # label: the subject id of the center we're currently comparing against
            if label == true_label:
                # center belongs to the test image's true subject
                genuine = d
            else:
                # update the impostor score if the center is closer than the current best --> finds the nearest wrong class
                if d < impostor:
                    impostor = d
        # converts both lists to numpy arrays and returns them as a pair
        genuine_scores.append(genuine)
        impostor_scores.append(impostor)
    return np.array(genuine_scores), np.array(impostor_scores)


def train_lda(train_vectors, train_labels):
    """
    Fits a Fisher Linear Discriminant model on the training feature vectors and returns the fitted sklearn LDA object.
    Works as LDA projects high-dimensional feature vectors into a lower-dimensional space that maximizes class separability
    by increasing between-class variance and reducing within-class variance.
 
    Parameters:
    - train_vectors (np.ndarray): matrix with one row per training image
    - train_labels (list): subject id string per training image
    """
    # select the max allowed dimensions without breaking the math and create the LDA model
    n_components = min(N_COMPONENTS, len(set(train_labels)) - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    # each image produces one feature vector: 3 images per eye --> multiple vectors per subject --> matrix where each row is an image's feature vector
    lda.fit(train_vectors, train_labels)
    # return the fitted model used to project both training and test vectors (into the reduced space)
    return lda


def predict(lda, train_vectors, train_labels, test_vectors):
    """
    Classify the test vectors using a nearest-center classifier in the LDA space
    and return the predictions, which include manhattan, euclidean, and cosine distance measures (corresponds to equation 8 of the paper).
    Project training and test vectors into the reduced space to compute a class center per subject.
    Then, for each test vector: find the closest class center under L1, L2, and cosine distances and assign the corresponding label.
 
    Parameters:
    - lda (LinearDiscriminantAnalysis): fitted LDA model
    - train_vectors (np.ndarray): matrix with one row per training image
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): matrix with one row per testing image
    """
    # LDA-projected training and test vectors with shape (N_train, n_components)
    projected_train_vectors = lda.transform(train_vectors)
    projected_test_vectors = lda.transform(test_vectors)
    # get the mean vector per subject from the projected training vectors
    class_centers = _get_class_centers(projected_train_vectors, train_labels)
    # extract the labels and align them with their corresponding centers so they can be iterated together
    center_labels = list(class_centers.keys())
    center_matrix = np.array([class_centers[l] for l in center_labels])
    # accumulate predicted labels for each distance measure
    predictions = {"l1": [], "l2": [], "cosine": []}
 
    for test_vector in projected_test_vectors:
        # for each test image's projected feature vector, keep track of the best distance for each measure
        best_l1 = (None, float("inf"))
        best_l2 = (None, float("inf"))
        best_cosine = (None, float("inf"))
 
        for label, center in zip(center_labels, center_matrix):
            # for each class center and its subject id label, compute the distance between the current test vector and the current class center for all methods
            d1 = np.sum(np.abs(test_vector - center))  # l1, manhattan
            d2 = np.sqrt(np.sum((test_vector - center) ** 2))  # l2, euclidean
            d3 = _get_cosine_distance(test_vector, center)  # cosine
            # update the best match for each measure if needed
            if d1 < best_l1[1]:
                best_l1 = (label, d1)
            if d2 < best_l2[1]:
                best_l2 = (label, d2)
            if d3 < best_cosine[1]:
                best_cosine = (label, d3)
 
        # append the winning label for each measure to the predictions list.
        predictions["l1"].append(best_l1[0])
        predictions["l2"].append(best_l2[0])
        predictions["cosine"].append(best_cosine[0])
 
    return predictions


def match_iris():
    """
    Orchestration function of this module.
    Trains the LDA model to execute classification
    and returns the fitted model along with its predictions in the format {"l1": [...], "l2": [...], "cosine": [...]}.
 
    Parameters:
    - train_vectors (np.ndarray): matrix with one row per training image
    - train_labels (list): subject id string per training image
    - test_vectors (np.ndarray): matrix with one row per testing image
    """
    # train and predict
    lda = train_lda(train_vectors, train_labels)
    predictions = predict(lda, train_vectors, train_labels, test_vectors)
    return lda, predictions


if __name__ == "__main__":
    match_iris()
