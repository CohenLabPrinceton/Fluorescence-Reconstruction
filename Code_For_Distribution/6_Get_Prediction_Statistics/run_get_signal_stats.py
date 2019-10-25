# Import libraries
import os
import sys
import time
import numpy as np
import random
import scipy
from scipy import stats
from sklearn.metrics import mean_squared_error
# Our code 
import data_loader
import utils

'''
This script loads in prediction images and ground truth fluorescent images, and gets statistics comparing them. 
We report on the Pearson correlation coefficient, mean sqaured error, as well as statistics on a reduced subset of the test data (as a function of image intensity). 

Dependencies: 
1) utils.py
We utilize the file utils.py to keep our data organized for each cell type, magnification, and feature dataset. So, here we demonstrate how to use information from that utils.py file (paths to ground truth data, intensity thresholds, etc.). 
2) data_loader.py
We utilize this helper file to load in the images from a given path (taken from utils.py). 


User-defined inputs: 

cell_type - (string) the type of cell used (MDCK-II, Keratinocytes, or HUVEC, in our experiments). 
magnification - (string) the magnification used in the experiment. 
        (For example, our MDCK-II cells were imaged at both 5x and 20x.)
output_var - (string) the type of fluorescent channel for the feature we consider in this experiment. 
        (For example, our MDCK-II 20x experiment involved imaging both in 'DAPI' and 'RFP'.)      
use_loss - (string) Determines if we use 'mse', the mean squared error loss, or 'p', the Pearson correlation coefficient.   
use_net - (string) Determines if we use a 1-stack U-Net or a 2-stack U-Net.   

save_path - (string) the path to the folder into which output .csvs will be saved. 
pred_dir - (string) the path to the folder from which the prediction images will be read.    
        
Statistics are printed out, and complete Pearsons values are saved to .csv files.       
'''


# ------------------------------------------------------------------
# User-defined parameters:

# Options: 'MDCK', 'KC', 'HUVECS_512', 'HUVECS_256', 'FUCCI'
cell_type = 'KC'
# Options: '5x', '10x', '20x', '60x', '5x_BS'
magnification = '10x'
# Options: 'DAPI', 'RFP', 'GFP', 'YFP', 'CY5'
output_var = 'DAPI'

# Options: 'p', 'mse'
use_loss = 'mse'

# Options: '1stack', '2stack'
use_net = '1stack'

save_path = './violin_plots/'
pred_dir = '/tigress/jl40/predictions_fixed/'
pred_dir = pred_dir + cell_type + '_' + magnification + '/' + output_var + '_' + use_loss + '_' + use_net + '/'
# ------------------------------------------------------------------

# Get data details (path): 
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev = utils.get_normalization_factors(cell_type, magnification, output_var)

# Get prediction and ground truth images (no need to normalize):
Y_pred, Y_gt = data_loader.get_pred_gt_data(pred_dir, TEST_PATH + output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Get the intensity threshold for this experiment. 
# The threshold denotes the intensity value. If images in the ground truth have values above this value, we assume there are meaningful features in that image, and keep the matched pairs of images. 
signal_thres = utils.get_signal_thres(cell_type, magnification, output_var[:-1])
print('Signal thres is: ' + str(signal_thres))

# Store the Pearson correlation coefficient and mean squared error in an array for each image pair.
pears = np.empty(len(Y_pred))
pvals = np.empty(len(Y_pred))
mse_vals = np.empty(len(Y_pred))
indx_store = []
for i in range(len(Y_pred)):
    pear = scipy.stats.pearsonr(np.array(Y_pred[i][:,:,0]).flatten(), np.array(Y_gt[i][:,:,0]).flatten())
    pears[i] = pear[0]
    pvals[i] = pear[1]

    pgt = (np.array(Y_gt[i][:,:,0]).flatten() - output_mean) / output_stdev
    ppred = (np.array(Y_pred[i][:,:,0]).flatten() - output_mean) / output_stdev
    mse_vals[i] = mean_squared_error(pgt, ppred)

    if((Y_gt[i][:,:,0] > signal_thres).any()):
        indx_store.append(i)

# Take statistics and print out the values for each condition. 
pear_reduced_mean = np.mean(pears[indx_store])
pear_reduced_stdev = np.std(pears[indx_store])
mse_reduced_mean = np.mean(mse_vals[indx_store])
mse_reduced_stdev = np.std(mse_vals[indx_store])

# Total Test Set Statistics:
# Get mean and stdev
print('-------------------------')
print('PCC Mean is: ' + str(np.mean(pears)))
print('-------------------------')
print('Stdev is: ' + str(np.std(pears)))

# Get mean and stdev
print('-------------------------')
print('MSE Mean is: ' + str(np.mean(mse_vals)))
print('-------------------------')
print('Stdev is: ' + str(np.std(mse_vals)))

print('----------------------------------------------')
print('----------------------------------------------')

# Reduced Test Set Statistics:
# Get mean and stdev
print('-------------------------')
print('Signal PCC Mean is: ' + str(pear_reduced_mean))
print('-------------------------')
print('Signal Stdev is: ' + str(pear_reduced_stdev))

# Get mean and stdev
print('-------------------------')
print('Signal MSE Mean is: ' + str(mse_reduced_mean))
print('-------------------------')
print('Signal Stdev is: ' + str(mse_reduced_stdev))

print('The number of signal images is: ' + str(len(indx_store)))
print('The total number of images is: ' + str(len(Y_pred)))

# Save the .csv files listing all these statistics (for easier bookkeeping). 
name_ext = cell_type + '_' + magnification + '_' + output_var[:-1] + '_' + use_loss + '_' + use_net 
pear_path = save_path + 'pear_' + name_ext + '.csv'
red_path = save_path + 'reduced_' + name_ext + '.csv'

pears = np.reshape(pears, (-1,1))
red = np.reshape(pears[indx_store], (-1,1))

np.savetxt(pear_path, pears, delimiter=",")
np.savetxt(red_path, red, delimiter=",")



