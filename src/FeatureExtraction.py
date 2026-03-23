"""
FeatureExtraction - Filtering the iris and extracting features following '3.3 Feature Extraction' from the paper.
"""

import numpy as np
import cv2

def circularly_symmetric_filter(size, sigma_x, sigma_y, f):
    """
    Generates a circularly symmetric spatial filter as defined in Equation 3 of Ma et al. (2003).
    Unlike standard Gabor filters that capture information only at a specific orientation, this custom filter uses a circularly symmetric sinusoidal modulating function. This allows it to capture rich texture information spreading along the radial direction of the iris regardless of minor orientation differences.

    Parameters:
    - size (int): The size of the kernel (size x size). 
    - sigma_x (float): Standard deviation of the Gaussian envelope along the x-axis.
    - sigma_y (float): Standard deviation of the Gaussian envelope along the y-axis.
    - f (float): The frequency of the circular sinusoidal function.

    Returns:
    - custom_filter (numpy.ndarray): The 2D spatial filter kernel ready for convolution.
    """

    # Define an X-Y grid centered at 0 to evaluate the mathematical functions symmetrically
    half_size = size // 2
    x = np.arange(-half_size, half_size + 1)
    y = np.arange(-half_size, half_size + 1)
    X, Y = np.meshgrid(x, y)

    # Gaussian envelope: Creates the bell-curve shape to localize the filter's effect
    envelope = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(-0.5 * ((X**2 / sigma_x**2) + (Y**2 / sigma_y**2)))
 
    # Circularly symmetric sinusoidal function (M1 in the paper)
    modulator1 = np.cos(2 * np.pi * f * np.sqrt(X**2 + Y**2))

    # Oriented sinusoidal function (M2 in the paper) - Standard Gabor filter
    # modulator2 = np.cos(2 * np.pi * f  * (X*np.cos(theta) + Y*np.sin(theta)))

    # Final custom spatial filter, implementation of Equation 3
    custom_filter = envelope * modulator1

    return custom_filter


def extract_features(normalized_iris, block_size=8):
    """
    Logic behind the script:
    Extracts texture features following Section 3.3.2 of Ma et al. (2003).
    First, it crops the top 48 rows of the normalized image to create a 48x512 Region of Interest (ROI). 
    It convolves this ROI with two spatial filters (2 channels). 
    Finally, it divides each filtered image into 8x8 small blocks, computing the Mean and 
    Average Absolute Deviation (AAD) for each, yielding a 1536-dimensional feature vector.
    
    Parameters:
    - normalized_iris (numpy.ndarray): The preprocessed, rectangular iris image.
    - block_size (int): The size of the square blocks to extract features from (default 8).

    Returns:
    - np.ndarray: The final 1D feature vector of 1536 elements.
    """
    
    # Define filter bank parameters based on the paper's multi-channel approach.
    filters = [
        circularly_symmetric_filter(size=11, sigma_x=3.0, sigma_y=1.5, f=0.1),
        circularly_symmetric_filter(size=11, sigma_x=4.5, sigma_y=1.5, f=0.2)
    ]
    
    features = []

    # Crop the Region of Interest (ROI): top 48 rows as per paper
    roi = normalized_iris[:48, :]

    # Calculate number of vertical and horizontal blocks dynamically based on ROI and block_size
    h, w = roi.shape
    num_blocks_h = h // block_size  # 48 // 8 = 6 blocks
    num_blocks_w = w // block_size  # 512 // 8 = 64 blocks    
    
    # First Loop: Iterate over each spatial filter (multi-resolution channels)
    for filt in filters:
        
        # Convolve the image with the spatial filter, implementation of Equation 4
        filtered_image = cv2.filter2D(roi, cv2.CV_64F, filt)
        
        # Nested Loops: Slide across the filtered image grid block by block
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                
                # Extract the specific 8x8 local block from the filtered image
                block = filtered_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                
                # Compute local mean (m) of absolute values.
                # This represents the average energy of the texture in this specific block.
                mean_val = np.mean(np.abs(block)) 
                
                # Compute average absolute deviation (AAD or sigma).
                # This measures the variance/contrast of the micro-details relative to the mean.
                aad_val = np.mean(np.abs(np.abs(block) - mean_val))
                
                # Append the two extracted statistical features to the main vector
                features.extend([mean_val, aad_val])
                
    return np.array(features)