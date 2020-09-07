
import time
import numpy as np

# ML dependencies
import keras
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K

from libtiff import TIFF
from tqdm import tqdm
import cv2
import os

from source import frm_models, frm_utils

def do_test(working_dir, weights_file):
    t = time.time()
    print('Testing the model....')
    print('-------------------------------------------')

    # Get normalization parameters:
    stats_data = np.load(working_dir + 'stats_data.npy')
    in_mean = stats_data[0]
    in_stdev = stats_data[1]
    out_mean = stats_data[2]
    out_stdev = stats_data[3]

    # Get test data files from in/out image directories and sort by name: 
    input_train_list = frm_utils.get_file_list_from_dir(working_dir + 'Testing/Input_Var/')
    output_train_list = frm_utils.get_file_list_from_dir(working_dir + 'Testing/Output_Var/')
    input_train_list = sorted(input_train_list)  
    output_train_list = sorted(output_train_list)

    # Read in and normalize train images 
    X_test = np.zeros((len(input_train_list), 256, 256, 1), dtype=np.float64)
    Y_test = np.zeros((len(output_train_list), 256, 256, 1), dtype=np.float64)
    print('Getting input test images... ')
    for n, id_ in tqdm(enumerate(input_train_list), total=len(input_train_list)):
        path = working_dir + 'Testing/Input_Var/'
        img = cv2.imread(path + id_, -1)
        X_test[n][:,:,0] = img[:,:]
    print('Getting output test images... ')
    for n, id_ in tqdm(enumerate(output_train_list), total=len(output_train_list)):
        path = working_dir + 'Testing/Output_Var/'
        img = cv2.imread(path + id_, -1)
        Y_test[n][:,:,0] = img[:,:]
    print('-------------------------------------------')

    # Normalize input images: 
    X_test = (X_test - in_mean) / in_stdev

    # Get model:
    model = frm_models.get_unet(256, 256, 1)

    # Model parameters and optimization settings:
    adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    loss_func = 'mean_squared_error'
    weights_file = working_dir + weights_file

    # Compile model: 
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

    # Make the directory for saving prediction images: 
    save_dir = working_dir + 'predictions/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop over the test images, process each and save them as .tif files: 
    for i in range(len(Y_pred)):
        Y_pred[i][:,:,0] = ((Y_pred[i][:,:,0]*out_stdev) + (out_mean)).clip(min=0.0)
        img = (Y_pred[i][:,:,0]).astype('uint16')    
        im_name = save_dir + 'output_prediction_' + str(i).zfill(5) + '.tif'
        tiff = TIFF.open(im_name, mode='w')
        tiff.write_image(img)
        tiff.close()

    elapsed = time.time() - t
    print('-------------------------------------------')
    print('Testing: Elapsed time in seconds: %d' % elapsed)
    print('-------------------------------------------')

    return
