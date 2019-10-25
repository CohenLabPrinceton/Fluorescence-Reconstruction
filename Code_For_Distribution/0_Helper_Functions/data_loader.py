# Import libraries: 
import os
import sys
import numpy as np
import random
import warnings
from tqdm import tqdm
import cv2

seed = 42
test_size = 0.2
random.seed = seed
np.random.seed = seed

# Helper function to get all files in directory: 
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: (file.endswith('.tif') or file.endswith('.tiff')), all_files))
    return data_files

# Read in the entire dataset from the TRAIN_PATH and return as numpy arrays: 
def get_data(TRAIN_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev):

    # Get files from in/out image directories and sort by name: 
    input_train_list = get_file_list_from_dir(TRAIN_PATH + input_var)
    output_train_list = get_file_list_from_dir(TRAIN_PATH + output_var)
    input_train_list = sorted(input_train_list)  
    output_train_list = sorted(output_train_list)

    # Randomly shuffle image names consistently: 
    c = list(zip(input_train_list, output_train_list))
    random.shuffle(c)
    input_train_list,output_train_list = zip(*c)

    # Read in and normalize train images 
    X_train = np.zeros((len(input_train_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float64)
    Y_train = np.zeros((len(output_train_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float64)
    print('Getting and resizing input train images... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(input_train_list), total=len(input_train_list)):
        path = TRAIN_PATH + input_var
        img = cv2.imread(path + id_, -1)
        img = (img - input_mean) / input_stdev
        X_train[n][:,:,0] = img[:,:]
        
    print('Getting and resizing output train images... ')
    for n, id_ in tqdm(enumerate(output_train_list), total=len(output_train_list)):
        path = TRAIN_PATH + output_var
        img = cv2.imread(path + id_, -1)
        img = (img - output_mean) / output_stdev
        Y_train[n][:,:,0] = img[:,:]
    return X_train, Y_train

# Read in a reduced set of images from TRAIN_PATH and return as numpy arrays:
def get_data_frac(frac_in, TRAIN_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev):

    # Get files from in/out image directories and sort by name: 
    input_train_list = get_file_list_from_dir(TRAIN_PATH + input_var)
    output_train_list = get_file_list_from_dir(TRAIN_PATH + output_var)
    input_train_list = sorted(input_train_list)
    output_train_list = sorted(output_train_list)

    # Randomly shuffle image names consistently: 
    c = list(zip(input_train_list, output_train_list))
    random.shuffle(c)
    input_train_list,output_train_list = zip(*c)

    # Take a fraction of the shuffled data: 
    #frac = int(np.floor(frac_in*len(input_train_list)))    # "fraction of data" version 
    frac = frac_in                                          # "up-to-index" version
    input_train_list = input_train_list[:frac]
    output_train_list = output_train_list[:frac]

    # Read in and normalize train images 
    X_train = np.zeros((len(input_train_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float64)
    Y_train = np.zeros((len(output_train_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float64)
    print('Getting and resizing input train images... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(input_train_list), total=len(input_train_list)):
        path = TRAIN_PATH + input_var
        img = cv2.imread(path + id_, -1)
        img = (img - input_mean) / input_stdev
        X_train[n][:,:,0] = img[:,:]

    print('Getting and resizing output train images... ')
    for n, id_ in tqdm(enumerate(output_train_list), total=len(output_train_list)):
        path = TRAIN_PATH + output_var
        img = cv2.imread(path + id_, -1)
        img = (img - output_mean) / output_stdev
        #img = img / 16383.0
        Y_train[n][:,:,0] = img[:,:]

    return X_train, Y_train

# Read in images from TEST_PATH without sorting and return as numpy arrays:
def get_data_test(TEST_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev):

    # Get files from in/out image directories and sort by name: 
    input_train_list = get_file_list_from_dir(TEST_PATH + input_var)
    output_train_list = get_file_list_from_dir(TEST_PATH + output_var)
    input_train_list = sorted(input_train_list)
    output_train_list = sorted(output_train_list)

    # Read in and normalize test images 
    X_train = np.zeros((len(input_train_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float64)
    Y_train = np.zeros((len(output_train_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float64)

    print('Getting and resizing input test images... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(input_train_list), total=len(input_train_list)):
        path = TEST_PATH + input_var
        img = cv2.imread(path + id_, -1)
        img = (img - input_mean) / input_stdev
        X_train[n][:,:,0] = img[:,:]

    print('Getting and resizing output test images... ')
    for n, id_ in tqdm(enumerate(output_train_list), total=len(output_train_list)):
        path = TEST_PATH + output_var
        img = cv2.imread(path + id_, -1)
        img = (img - output_mean) / output_stdev
        Y_train[n][:,:,0] = img[:,:]
    return X_train, Y_train

# Read in images exactly without normalizing them, and return as numpy arrays: 
def get_pred_gt_data(pred_path, gt_path, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):

    # Get files from image directories and sort by name: 
    pred_list = get_file_list_from_dir(pred_path)
    gt_list = get_file_list_from_dir(gt_path)
    pred_list = sorted(pred_list)
    gt_list = sorted(gt_list)

    # Read in images and store as arrays 
    Y_pred = np.zeros((len(pred_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float64)
    Y_gt = np.zeros((len(gt_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float64)

    print('Getting and resizing input train images... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(pred_list), total=len(pred_list)):
        path = pred_path
        img = cv2.imread(path + id_, -1)
        Y_pred[n][:,:,0] = img[:,:]

    print('Getting and resizing output train images... ')
    for n, id_ in tqdm(enumerate(gt_list), total=len(gt_list)):
        path = gt_path
        img = cv2.imread(path + id_, -1)
        Y_gt[n][:,:,0] = img[:,:]
    return Y_pred, Y_gt


if __name__ == '__main__':
    get_data(TRAIN_PATH, input_var, output_var, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, input_mean, input_stdev, output_mean, output_stdev)
    
