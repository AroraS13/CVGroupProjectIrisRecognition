"""
FeatureExtraction - Filtering the iris and extracting features following '3.3 Feature Extraction' from the paper.
"""

import numpy as np
import cv2

def circularly_symmetric_filter(size, sigma_x, sigma_y, f):
    """
    Generates a circularly symmetric spatial filter as defined in Equation 3 of Li Ma et al.
    For reference, read '3.3.2 Feature Vector' from the paper.    
    
    Parameters:
    - size (int): The size of the kernel (size x size). 
    - sigma_x (float): Standard deviation of the Gaussian envelope along the x-axis.
    - sigma_y (float): Standard deviation of the Gaussian envelope along the y-axis.
    - f (float): The frequency of the circular sinusoidal function.
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


def extract_features(normalized_iris, num_blocks_h=8, num_blocks_w=48):
    """
    Extracts texture features from the normalized iris image.
    Generates a 1536-dimensional vector using 2 channels, 8x48 blocks, and 2 stats (Mean, AAD).
    For reference, read '3.3.2 Feature Vector' from the paper.
    
    Parameters:
    - normalized_iris (numpy.ndarray): The preprocessed, rectangular iris image.
    - num_blocks_h (int): Number of vertical blocks to divide the image into (default 8).
    - num_blocks_w (int): Number of horizontal blocks to divide the image into (default 48).
    """
    
    # Define filter bank parameters based on the paper's multi-channel approach.
    filters = [
        circularly_symmetric_filter(size=11, sigma_x=3.0, sigma_y=1.5, f=0.1),
        circularly_symmetric_filter(size=11, sigma_x=4.5, sigma_y=1.5, f=0.2)
    ]
    
    features = []
    
    # First Loop: Iterate over each spatial filter (multi-resolution channels)
    for filt in filters:
        
        # Convolve the image with the spatial filter, implementation of Equation 4
        filtered_image = cv2.filter2D(normalized_iris, cv2.CV_64F, filt)
        
        # Determine the pixel height and width of each small local block
        h, w = filtered_image.shape
        block_h = h // num_blocks_h
        block_w = w // num_blocks_w
        
        # Nested Loops: Slide across the filtered image grid block by block
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                
                # Extract the specific local block (patch) from the filtered image
                block = filtered_image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                
                # Compute local mean (m) of absolute values.
                # This represents the average energy of the texture in this specific block.
                mean_val = np.mean(np.abs(block)) 
                
                # Compute average absolute deviation (AAD or sigma).
                # This measures the variance/contrast of the micro-details relative to the mean.
                aad_val = np.mean(np.abs(np.abs(block) - mean_val))
                
                # Append the two extracted statistical features to the main vector
                features.extend([mean_val, aad_val])
                
    return np.array(features)

if __name__ == "__main__":
    # Placeholder 
    pass