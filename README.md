# Iris Recognition Project

This repository contains the implementation of an automated iris recognition system based on the methodology described in the 2003 paper [*Personal Identification Based on Iris Texture Analysis*](https://ieeexplore.ieee.org/document/1251145) by Li Ma et al.

## Project Structure
```text
├── src/                    # Source code
│   ├── IrisRecognition.py      # Main entry point
│   ├── IrisLocalization.py     # Pupil & iris boundary detection
│   ├── IrisNormalization.py    # Cartesian → polar mapping
│   ├── ImageEnhancement.py     # Normalized image enhancement
│   ├── FeatureExtraction.py    # Filtering & feature extraction
│   ├── IrisMatching.py         # Fisher LDA + nearest center classifier
│   └── PerformanceEvaluation.py # CRR & ROC evaluation
├── data/                   # CASIA Iris Image Database (version 1.0)
├── figures/                # Output figures (ROC curves, tables)
└── README.md
```


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place CASIA Iris Image Database in `data/`. 108 eyes, 7 images per eye (320×280 BMP).


## Logic of the Design
Our system processes raw eye images through a sequential biometric pipeline to identify individuals based on their unique iris textures. The logic is divided into five main stages:

1. **Iris Localization (IrisLocalization.py)**: The system first isolates the region of interest by detecting the inner boundary (pupil-iris) and the outer boundary (iris-sclera) of the eye.

2. **Iris Normalization (IrisNormalization.py)**: Because the size of the iris varies depending on camera distance and pupil dilation, the isolated circular iris ring is unwrapped into a fixed-size rectangular block. This mapping from Cartesian to polar coordinates ensures scale and translation invariance.

3. **Image Enhancement (ImageEnhancement.py)**: The unwrapped image often suffers from low contrast and uneven lighting. We apply background illumination subtraction and histogram equalization to enhance the texture details.

4. **Feature Extraction (FeatureExtraction.py)**: We apply a custom multi-resolution spatial filter with a circularly symmetric modulating sinusoidal function across two frequency channels (f=0.1 and f=0.2). The filtered image is divided into small 8x48 blocks. For each block, we compute the local Mean and Average Absolute Deviation (AAD), yielding a highly discriminatory 1,536-dimensional feature vector.

5. **Matching & Evaluation (IrisMatching.py & PerformanceEvaluation.py)**: To optimize classification, we use the Fisher Linear Discriminant (FLD) to reduce the feature vector's dimensionality from 1536 to 200. Finally, a nearest-center classifier calculates the cosine similarity between vectors to determine matches.

## Limitations of the Current Design
While highly effective on the CASIA dataset, the current design has a few architectural limitations:

1. **Rigid Geometric Assumptions**: The localization step assumes that both the pupil and the iris are perfect circles. In reality, pupils can be slightly elliptical and are rarely perfectly concentric with the outer iris boundary.

2. **Vulnerability to Occlusions**: The current pipeline unwraps the iris but does not actively detect and mask out heavy occlusions caused by eyelids, eyelashes, or specular reflections (light glares). If eyelashes cover a large portion of the iris, their texture will be encoded into the feature vector, introducing noise and potentially causing false rejections.

3. **Sensitivity to Image Quality**: The system currently attempts to process all images fed to it. If an image is heavily out of focus or motion-blurred, the spatial filters will extract flattened statistics, reducing matching accuracy.

## How to Improve It
To build a more robust and production-ready system, the following improvements can be made:

1. **Eyelid/Eyelash Masking**: Introduce a segmentation step (e.g., using parabolic Hough transforms for eyelids or Gabor variance for eyelashes) immediately after localization. Creating a binary noise mask would allow the feature extractor to ignore occluded pixels entirely.

2. **Advanced Boundary Detection**: Replace basic circular edge detection with Active Contour Models (Snakes) or Integro-differential operators to accurately map non-circular and non-concentric pupil shapes.

3. **Pre-Processing Quality Check**: Implement the Image Quality Assessment stage originally mentioned in the Li Ma paper. By using a Support Vector Machine (SVM) to evaluate the low, middle, and high-frequency power of the image's 2D Fourier spectrum, we could automatically discard blurry or heavily occluded images before wasting compute power on feature extraction.