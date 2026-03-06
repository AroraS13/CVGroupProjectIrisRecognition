"""
dataset_loader.py

Loads Image Dataset

Session 1 for training
Session 2 for testing
"""

import os
import cv2


def load_dataset(root_dir):
    """
    Returns two lists:
        train_data: list of (subject_id, image) for session 1
        test_data:  list of (subject_id, image) for session 2
    """
    train_data = []
    test_data = []

    for subject_id in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, subject_id)
        if not os.path.isdir(sub_path):
            continue

        for session in ["1", "2"]:
            sess_path = os.path.join(sub_path, session)
            if not os.path.isdir(sess_path):
                continue

            for filename in sorted(os.listdir(sess_path)):
                if not filename.lower().endswith(".bmp"):
                    continue

                img_path = os.path.join(sess_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"Warning: failed to load {img_path}")
                    continue

                if session == "1":
                    train_data.append((subject_id, image))
                else:
                    test_data.append((subject_id, image))

    return train_data, test_data


if __name__ == "__main__":
    import sys
    from collections import Counter

    root = sys.argv[1] if len(sys.argv) > 1 else "./data/CASIA-IrisV1"
    train, test = load_dataset(root)

    print(f"Training samples : {len(train)}")
    print(f"Testing samples  : {len(test)}")
    print(f"Total subjects   : {len(set(s for s, _ in train))}")

    train_counts = Counter(s for s, _ in train)
    test_counts = Counter(s for s, _ in test)

    bad_train = [s for s, c in train_counts.items() if c != 3]
    bad_test = [s for s, c in test_counts.items() if c != 4]

    if bad_train:
        print("Subjects with unexpected number of training images:", bad_train)
    if bad_test:
        print("Subjects with unexpected number of testing images:", bad_test)