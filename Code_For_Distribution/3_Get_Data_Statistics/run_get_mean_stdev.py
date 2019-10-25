# Import libraries: 
import os
import sys
import time
import numpy as np
# Our code 
import data_loader
import utils
t = time.time()

'''
This script prints the mean and standard deviation of your training data (for both input and output images).

Dependencies: 
1) utils.py
We utilize the file utils.py to keep our data organized for each cell type, magnification, and feature dataset. So, here we demonstrate how to use information from that utils.py file. 
The mean and standard deviation of the training set are ultimately added to this utils.py file. 
2) data_loader.py
We utilize this helper file to load in the images from a given path (taken from utils.py). 

User-defined inputs: 
cell_type - (string) the type of cell used (MDCK-II, Keratinocytes, or HUVEC, in our experiments). 
magnification - (string) the magnification used in the experiment. 
        (For example, our MDCK-II cells were imaged at both 5x and 20x.)
output_var - (string) the type of fluorescent channel for the feature we consider in this experiment. 
        (For example, our MDCK-II 20x experiment involved imaging both in 'DAPI' and 'RFP'.)        
        
Users should remember to update the utils.py folder with the data statistics.        
'''

# Options: 'MDCK', 'KC', 'HUVECS_256'
cell_type = 'KC'
# Options: '5x', '10x', '20x'
magnification = '10x'
# Options: 'DAPI', 'RFP', 'GFP', 'YFP', 'CY5'
output_var = 'DAPI'
# ------------------------------------------------------------------

# Get paths to data and image details (height, width): 
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev = utils.get_normalization_factors(cell_type, magnification, output_var)

# Get data using the data loader: 
X_train, Y_train = data_loader.get_data(TRAIN_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, 0.0, 1.0, 0.0, 1.0)

# Compute statistics on the input image collection: 
dmean = np.mean(X_train[:,:,:,0])
dstdev = np.std(X_train[:,:,:,0])

print('----------------------')

print('The input mean is: ')
print(dmean)
print('The input std is: ')
print(dstdev)

print('----------------------')

# Compute statistics on the output image collection: 
dmean = np.mean(Y_train[:,:,:,0])
dstdev = np.std(Y_train[:,:,:,0])

print('The output mean is: ')
print(dmean)
print('The output std is: ')
print(dstdev)

print('----------------------')

elapsed = time.time() - t
print(elapsed)


