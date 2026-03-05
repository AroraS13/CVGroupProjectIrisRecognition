# Iris Recognition

Columbia University CV Group Project - Spring 2026

Implementation of iris recognition algorithm following Ma et al., 2003.

## Project Structure

```
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

## Pipeline

1. Iris Localization
2. Iris Normalization
3. Image Enhancement
4. Feature Extraction
5. Iris Matching
6. Performance Evaluation

## Dataset

Place CASIA Iris Image Database (version 1.0) in `data/`. 108 eyes, 7 images per eye (320×280 BMP).
