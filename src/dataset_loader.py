"""
dataset_loader.py

Loads Image Dataset

Session 1 for training
Session 2 for testing
"""

import os
import cv2


def load_dataset(dir):
    """
    Returns two lists:
        train_data: list of (id, image) for session 1
        test_data:  list of (id, image) for session 2
    """
    train_data = []
    test_data = []

    for id in sorted(os.listdir(dir)):
        sub_path = os.path.join(dir, id)
        if not os.path.isdir(sub_path):
            continue

        for session in ["1", "2"]:
            sess_path = os.path.join(sub_path, session)
            if not os.path.isdir(sess_path):
                continue

            for filename in sorted(os.listdir(sess_path)):
                if not filename.endswith(".bmp"):
                    continue

                img_path = os.path.join(sess_path, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if session == "1":
                    train_data.append((id, image))
                else:
                    test_data.append((id, image))

    return train_data, test_data


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./data/CASIA Iris Image Database (version 1"
    train, test = load_dataset(root)

    print(f"Training samples : {len(train)}")
    print(f"Testing samples  : {len(test)}")
    print(f"Total subjects   : {len(set(s for s, _ in train))}")
