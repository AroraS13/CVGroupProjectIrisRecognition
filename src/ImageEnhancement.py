"""
ImageEnhancement.py

Enhances normalized image using various techniques for subsequent feature extraction and matching
"""
import numpy as np
import cv2
def enhanceImage(iris_normalized):
    """
    Image enhancement function: Takes in normalized iris image and returns an enhanced version
    """
    #Get dimensions
    height, width = iris_normalized.shape
    #Set block size: Size of the regions that will be used for calculating illumination. We used 16 here following the paper 
    block_size = 16
    #Calculate # of rough rows and columns based on block size
    rough_rows = height//block_size 
    rough_columns = width//block_size 
    #Create an empty array to hold rough illumination values
    rough_illumination = np.zeros((rough_rows, rough_columns), dtype=np.float32)
    #For each block
    for i in range(rough_rows):
        for j in range(rough_columns):
            #Calculate the bounds of each block 
            #r_start and r_end are start and end of each block in terms of row
            r_start = i*block_size
            r_end = r_start+block_size 
            #col_start and col_end are start and end of each block in terms of column
            col_start= j*block_size
            col_end= col_start+block_size
            #Find block of pixels based on calculated bounds
            block = iris_normalized[r_start:r_end, col_start:col_end]
            #Calculate the mean of the block, and add it to our rough_illumination array
            rough_illumination[i,j]=np.mean(block)
    #Resize the rough illumination array to the same as our original array using bicubic interpolation, following paper
    illumination = cv2.resize(rough_illumination,(width,height),interpolation=cv2.INTER_CUBIC)
    #Subtract illumination from original normalized image to get the adjusted image
    adjusted=iris_normalized.astype(np.float32)-illumination 
    #Normalize the adjusted image to be between 0 and 255
    adjusted= cv2.normalize(adjusted, None, 0, 255, cv2.NORM_MINMAX)
    #Convert to uint8
    adjusted = adjusted.astype(np.uint8)
    #create and empty array to hold enhanced values
    enhanced = np.zeros_like(adjusted)
    #set region size to 32, following paper. 
    region_size = 32
    #For each 32x32 region in the newly adjusted image
    for i in range(0, height, region_size):
        for j in range(0, width, region_size):
            #Calculate the bounds of the region
            r_end = min(i+region_size, height)
            c_end = min(j+region_size, width)
            #Find the region based on calculated bounds
            region = adjusted[i:r_end, j:c_end]
            #Apply histogram equilization
            equalized = cv2.equalizeHist(region)
            #Add equalized region to the enhanced image
            enhanced[i:r_end, j:c_end] = equalized 
    #return rough illumination, illumination, adjusted --> (for testing) and final enhanced image 
    return rough_illumination, illumination, adjusted, enhanced 

