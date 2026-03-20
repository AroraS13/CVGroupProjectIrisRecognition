"""
dataset_loader.py

1. Logic behind the script:
This script traverses the hierarchical directory structure of the CASIA Iris Image Database to load images into memory. The CASIA database is organized sequentially: Root -> Subject ID -> Session ID. 
The outer loop iterates through every subject directory (e.g., "001" to "108"). 
The inner loop explicitly checks for session folders "1" and "2". Images from session "1" are historically captured earlier and are appended to the training set. Images from session "2" are captured later and are appended to the testing set. OpenCV is used to load valid ".bmp" files directly as grayscale matrices, ensuring they are ready for the preprocessing pipeline.

2. Key variables/parameters:
- root_dir (str): The path to the root folder containing all subject directories (e.g., 'data/CASIA Iris Image Database/').
- train_data (list): Accumulates training samples. Formatted as a list of tuples: [(subject_id, image_array), ...].
- test_data (list): Accumulates testing samples. Formatted similarly to train_data.
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