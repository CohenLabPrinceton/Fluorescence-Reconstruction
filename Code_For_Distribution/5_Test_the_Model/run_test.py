# Import libraries
import os
import sys
import time
import numpy as np
import keras
from keras import optimizers
from keras.models import Model, load_model
from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
from libtiff import TIFF
# Our code 
import data_loader
import models
import utils

t = time.time()

'''
This script loads in your network and weights, and uses it to predict outputs for images in the test set. 
Predicted images are saved to a path of your choosing. 

Dependencies: 
1) utils.py
We utilize the file utils.py to keep our data organized for each cell type, magnification, and feature dataset. So, here we demonstrate how to use information from that utils.py file (paths to ground truth data). 
2) data_loader.py
We utilize this helper file to load in the images from a given path (taken from utils.py). 
3) models.py
This file contains the neural network architectures implemented in TensorFlow/Keras. We pull the model definition from here and begin training. 


User-defined inputs: 

save_dir - (string) the path to the folder into which output images will be saved. 

cell_type - (string) the type of cell used (MDCK-II, Keratinocytes, or HUVEC, in our experiments). 
magnification - (string) the magnification used in the experiment. 
        (For example, our MDCK-II cells were imaged at both 5x and 20x.)
output_var - (string) the type of fluorescent channel for the feature we consider in this experiment. 
        (For example, our MDCK-II 20x experiment involved imaging both in 'DAPI' and 'RFP'.)      
use_loss - (string) Determines if we use 'mse', the mean squared error loss, or 'p', the Pearson correlation coefficient.   
use_net - (string) Determines if we use a 1-stack U-Net or a 2-stack U-Net.      
        
      
'''

# ------------------------------------------------------------------
# User-defined parameters:

save_dir = '/scratch/gpfs/jl40/predictions/'

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
# ------------------------------------------------------------------

# Get data details including path and normalization stats: 
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev = utils.get_normalization_factors(cell_type, magnification, output_var)

# Find weights file to be saved 
if(use_net == '1stack'):
    weights_file = utils.get_weights_file_standard_nonorm(cell_type, magnification, output_var[:-1], use_loss)
if(use_net == '2stack'):
    weights_file = utils.get_weights_file_2stacknet_nonorm(cell_type, magnification, output_var[:-1], use_loss)
weights_file = out_dir + weights_file

# Get test data and normalize input: 
X_test, Y_test = data_loader.get_data_test(TEST_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev)

# Define model: 
if(use_net == '1stack'):
    model = models.get_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
if(use_net == '2stack'):
    model = models.get_2stack_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Define additional model optimization settings: 
adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

if(use_loss == 'p'):
    loss_func = utils.pear_corr
if(use_loss == 'mse'):
    loss_func = 'mean_squared_error'

# Compile model and load existing weights: 
model.compile(loss=loss_func,
              optimizer=adad,
              metrics=['mse'])
# Load weights
model.load_weights(weights_file, by_name=True)

# Print out the test accuracy: 
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('--------------------------------------------------')
print('Test loss: ', test_loss)
print('Test accuracy: ', test_acc)
print('--------------------------------------------------')

# Predict using the model: 
Y_pred = model.predict(X_test, batch_size = 32)

save_dir = save_dir + cell_type + '_' + magnification + '/' + output_var[:-1] + '_' + use_loss + '_' + use_net + '/'

# Make the directory for saving prediction images: 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Loop over the test images, process each and save them as .tif files: 
for i in range(len(Y_pred)):
    Y_pred[i][:,:,0] = ((Y_pred[i][:,:,0]*output_stdev) + (output_mean)).clip(min=0.0)
    img = (Y_pred[i][:,:,0]).astype('uint16')    
    im_name = save_dir + output_var[:-1] + '_prediction_' + str(i).zfill(5) + '.tif'
    tiff = TIFF.open(im_name, mode='w')
    tiff.write_image(img)
    tiff.close()

elapsed = time.time() - t
print(elapsed)
