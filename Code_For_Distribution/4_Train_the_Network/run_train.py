# Generic dependencies
import os
import sys
import time
import numpy as np

# ML dependencies
import keras
from keras import optimizers
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf

# Our code 
import data_loader
import models
import utils

t = time.time()

'''
This script is used to train the neural network. 
Data is loaded in using the data loader, and models are loaded from models.py. 
Either the mean squared error loss function or the PCC loss function may be used.
The outputs are a weights file (.h5 file), plus a .yaml file which fully stores the model, and a .csv file, which logs the training and validation losses. 


Dependencies: 
1) utils.py
We utilize the file utils.py to keep our data organized for each cell type, magnification, and feature dataset. So, here we demonstrate how to use information from that utils.py file. 
The mean and standard deviation of the training set are ultimately added to this utils.py file. 
2) data_loader.py
We utilize this helper file to load in the images from a given path (taken from utils.py). 
3) models.py
This file contains the neural network architectures implemented in TensorFlow/Keras. We pull the model definition from here and begin training. 


User-defined inputs: 
new_weights_file - (string) the name of the file which defines your trained model weights. 
            Once the network is trained, we record the weights file name in utils.py .
cell_type - (string) the type of cell used (MDCK-II, Keratinocytes, or HUVEC, in our experiments). 
magnification - (string) the magnification used in the experiment. 
        (For example, our MDCK-II cells were imaged at both 5x and 20x.)
output_var - (string) the type of fluorescent channel for the feature we consider in this experiment. 
        (For example, our MDCK-II 20x experiment involved imaging both in 'DAPI' and 'RFP'.)      
use_loss - (string) Determines if we use 'mse', the mean squared error loss, or 'p', the Pearson correlation coefficient.   
use_net - (string) Determines if we use a 1-stack U-Net or a 2-stack U-Net.       
        
Users should remember to update the utils.py folder with the data statistics.        
'''


# ------------------------------------------------------------------
# User-defined parameters:

# Define weights files: 
#old_weights_file = 'None'   # Optional: Can uncomment lines to support loading of existing weights. 
new_weights_file = 'nonorm_kc_10x_dapi_1stack_unet_pear.h5'

# Options: 'MDCK', 'KC', 'HUVECS_512', 'HUVECS_256', 'FUCCI'
cell_type = 'KC'
# Options: '5x', '10x', '20x'
magnification = '10x'
# Options: 'DAPI', 'RFP', 'YFP', 'CY5'
output_var = 'DAPI'

# Options: 'p', 'mse'
use_loss = 'mse'

# Options: '1stack', '2stack'
use_net = '1stack'
# ------------------------------------------------------------------


# Model parameters: 
max_epochs_no = 10000

# Data details including normalization stats: 
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev = utils.get_normalization_factors(cell_type, magnification, output_var)

# Define weights file to be saved 
new_weights_file = out_dir + new_weights_file

# Get Data and Normalize: 
X_train, Y_train = data_loader.get_data(TRAIN_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev)


# Define model: 
if(use_net == '1stack'):
    model = models.get_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
if(use_net == '2stack'):
    model = models.get_2stack_unet_nonorm(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


# Define additional model optimization settings: 
adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# If desired: distribute to multiple GPUs: 
#model = multi_gpu_model(model, 4)

# Define output file names to save .yaml and .csv files 
if(use_loss == 'p'):
    loss_func = utils.pear_corr
    csv_endname = '_pear_1stackunet_training_log_nonorm.csv'
    yaml_endname = '_pear_1stackunet_model_nonorm.yaml'
if(use_loss == 'mse'):
    loss_func = 'mean_squared_error'
    csv_endname = '_mse_1stackunet_training_log_nonorm.csv'
    yaml_endname = '_mse_1stackunet_model_nonorm.yaml'

# Compile model: 
model.compile(loss=loss_func,
              optimizer=adad,
              metrics=['mse'])
# If desired: pretraining...
#model.load_weights(old_weights_file, by_name=True)   # Optional: Can uncomment lines to support loading of existing weights. 

# Stop the model if validation loss has not decreased in the last 100 epochs:
earlystopper = EarlyStopping(patience=100, verbose=1)
# Save Checkpoints: 
checkpointer = ModelCheckpoint(new_weights_file, verbose=1, save_best_only=True)
# Log the losses to a .csv file: 
cLog = CSVLogger(out_dir + output_var[:-1] + csv_endname)

# Save initial model binary to a YAML file: 
yaml_string = model.to_yaml()
with open(out_dir + output_var[:-1] + yaml_endname, 'w') as text_file:
    print(yaml_string, file=text_file)

# Train the model: 
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=max_epochs_no, 
                    callbacks=[earlystopper, checkpointer, cLog])

elapsed = time.time() - t
print(elapsed)
