"""
dataset_loader.py

Loads image dataset into train and test to allow for easy access. 
"""

import os
import cv2


def load_dataset(root_dir):
    """
    Loads dataset from root_dir. Returns test and train data as lists
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


# if __name__ == "__main__":
#     import sys
#     from collections import Counter

#     root = sys.argv[1] if len(sys.argv) > 1 else "./data/CASIA-IrisV1"
#     train, test = load_dataset(root)

#     print(f"Training samples : {len(train)}")
#     print(f"Testing samples  : {len(test)}")
#     print(f"Total subjects   : {len(set(s for s, _ in train))}")

#     train_counts = Counter(s for s, _ in train)
#     test_counts = Counter(s for s, _ in test)

#     bad_train = [s for s, c in train_counts.items() if c != 3]
#     bad_test = [s for s, c in test_counts.items() if c != 4]

#     if bad_train:
#         print("Subjects with unexpected number of training images:", bad_train)
#     if bad_test:
#         print("Subjects with unexpected number of testing images:", bad_test)