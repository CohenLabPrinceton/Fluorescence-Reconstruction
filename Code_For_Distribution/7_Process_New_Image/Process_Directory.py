#!/usr/bin/env python
# coding: utf-8

# ## Process Directory
# 
# This script demonstrates how to utilize a trained model and process a new input transmitted-light input image using the saved weights. 
# 
# To keep this demo code simplistic, and to demonstrate the sliding-window approach, we expect input images to be divisible by 256 in width and height. The output image will display the excluded edge boundaries as nan values. 

# 1. Define user input, including input/output paths, and experimental details (cell type, features, etc.).

# Path to large input images (should be divisible by 256 on each side): 
input_dir = './Sample_Images/Phase_Image/'
# Path to save prediction images into: 
output_dir = './Sample_Images/DAPI_Prediction_Image/'
# Path to folder containing weights file (.h5): 
weights_path = './weights_9_4/'

# Define your experimental conditions: 
# Options: 'MDCK', 'KC', 'HUVECS_256'
cell_type = 'KC'
# Options: '5x', '10x', '20x'
magnification = '10x'
# Options: 'DAPI', 'RFP', 'YFP', 'CY5'
output_var = 'DAPI'

# Define model parameters: 
# Options: 'p', 'mse'
use_loss = 'mse'
# Options: '1stack', '2stack'
use_net = '1stack'


# 2. Get the relevant information for applying the model, including the model definition and weights. 

import os
import sys
import random
import warnings
import numpy as np
from PIL import Image
from libtiff import TIFF

from sklearn.metrics import mean_squared_error

import keras
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
import tensorflow as tf
# Our code: 
import utils
import models 

# ------------------------------------------------------------------

# Define name of output .tif files: 
pred_name = 'prediction' + '_' + use_loss + '_' + use_net + '_'
pred_img = output_dir + pred_name

# Data details including normalization stats: 
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev = utils.get_normalization_factors(cell_type, magnification, output_var)

# Define weights file to be read
if(use_net == '1stack'):
    weights_file = utils.get_weights_file_standard_nonorm(cell_type, magnification, output_var[:-1], use_loss)
if(use_net == '2stack'):
    weights_file = utils.get_weights_file_2stacknet_nonorm(cell_type, magnification, output_var[:-1], use_loss)
weights_file = weights_path + weights_file

# Print the weights file path: 
print(weights_file)

# Define model: 
if(use_net == '1stack'):
    model = models.get_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
if(use_net == '2stack'):
    model = models.get_2stack_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
if(use_loss == 'p'):
    loss_func = utils.pear_corr
if(use_loss == 'mse'):
    loss_func = 'mean_squared_error'

# Compile the U-Net model
model.compile(loss=loss_func,
              optimizer=adad,
              metrics=['mse'])

# Load the model weights 
model.load_weights(weights_file, by_name=True)
# If desired, summarize the model: 
#model.summary()


# 3. Process the images in the input folder and save predictions to the output folder. 

Image.MAX_IMAGE_PIXELS = None

# A helper function to get all the files in a directory. 
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.tif'), all_files))
    return data_files

# Helper functions for processing the input image: 
def sliding_window(image, stepSize, windowSize, imgarr, nucmap, covmap, inner_mat, outer_mat):
        # slide a window across the image
        for y in range(0, h-3*stepSize, stepSize):
                for x in range(0, w-3*stepSize, stepSize):
                        # process the current window
                        window_chunk = imgarr[:, y:y + windowSize[1], x:x + windowSize[0], :]
                        predict_chunk = model.predict(window_chunk, verbose=0)[0,:,:,0]
                        predict_reduced = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float64)
                        predict_reduced[int(IMG_HEIGHT/4):int(3*IMG_WIDTH/4), int(IMG_HEIGHT/4):int(3*IMG_WIDTH/4)] = predict_chunk[int(IMG_HEIGHT/4):int(3*IMG_WIDTH/4), int(IMG_HEIGHT/4):int(3*IMG_WIDTH/4)]    
                        nucmap[y:y + windowSize[1], x:x + windowSize[0]] += predict_reduced
                        covmap[y:y + windowSize[1], x:x + windowSize[0]] += outer_mat
        return nucmap, covmap

def process_one_image(i, img, h, w, pred_img, model, stepSize, windowSize, output_mean, output_stdev):
    # Initialize arrays: 
    imgarr = np.zeros((1, h, w, 1), dtype=np.float64)
    imgarr[0,:,:,0] = img
    nucmap = np.zeros((h, w), dtype=np.float64)
    covmap = np.zeros((h, w), dtype=np.float64)
    inner_mat = np.ones((int(IMG_HEIGHT/2), int(IMG_WIDTH/2)))
    outer_mat = np.pad(inner_mat, ((int(IMG_HEIGHT/4), int(IMG_WIDTH/4)), (int(IMG_HEIGHT/4), int(IMG_WIDTH/4))), 'constant', constant_values=(0, 0))

    # Process each patch in a sliding-window fasion:
    nucmap, covmap = sliding_window(img, stepSize, windowSize, imgarr, nucmap, covmap, inner_mat, outer_mat)
    nucmap = np.divide(nucmap, covmap)
    
    # Normalize the output image: 
    nucmap = (nucmap * output_stdev) + output_mean

    # Set the name of the output image: 
    pred_img_temp = pred_img + str(i).zfill(3) + '.tif'  
    # Convert output image to 16-bit type:
    nucmap = nucmap.clip(min=0.0)
    nucmap_arr = nucmap.astype('uint16')
    # Save the output image:
    tiff = TIFF.open(pred_img_temp, mode='w')
    tiff.write_image(nucmap_arr)
    tiff.close()
    return 

# Set stride to 64 pixels in each direction and window (patch) size.
stepSize = 64
windowSize = (256,256)

# Get the files from the input folder and sort them by name.
impathlist = get_file_list_from_dir(input_dir)
impathlist.sort()

for i in range(len(impathlist)):

    print(i)
    
    # Read in and normalize the input image: 
    impath = impathlist[i]
    img = np.array(Image.open(input_dir + impath))
    img = (img - input_mean)/(input_stdev)

    # Process each input image: 
    [h, w] = np.shape(img)
    process_one_image(i, img, h, w, pred_img, model, stepSize, windowSize, output_mean, output_stdev)
        



