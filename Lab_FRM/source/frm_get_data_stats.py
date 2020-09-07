
import os
import sys
import numpy as np
import random
import warnings
from tqdm import tqdm
import cv2
import time

from source import frm_utils

random.seed = 42

def do_get_data_stats(working_dir):
    t = time.time()
    print('Getting training data statistics....')
    print('-------------------------------------------')

    # Get data files from in/out image directories and sort by name: 
    input_train_list = frm_utils.get_file_list_from_dir(working_dir + 'Training/Input_Var/')
    output_train_list = frm_utils.get_file_list_from_dir(working_dir + 'Training/Output_Var/')
    input_train_list = sorted(input_train_list)  
    output_train_list = sorted(output_train_list)

    # Randomly shuffle image names consistently: 
    c = list(zip(input_train_list, output_train_list))
    random.shuffle(c)
    input_train_list,output_train_list = zip(*c)

    # Read in and normalize train images 
    X_train = np.zeros((len(input_train_list), 256, 256, 1), dtype=np.float64)
    Y_train = np.zeros((len(output_train_list), 256, 256, 1), dtype=np.float64)
    print('Getting input train images... ')
    for n, id_ in tqdm(enumerate(input_train_list), total=len(input_train_list)):
        path = working_dir + 'Training/Input_Var/'
        img = cv2.imread(path + id_, -1)
        X_train[n][:,:,0] = img[:,:]
        
    print('Getting output train images... ')
    for n, id_ in tqdm(enumerate(output_train_list), total=len(output_train_list)):
        path = working_dir + 'Training/Output_Var/'
        img = cv2.imread(path + id_, -1)
        Y_train[n][:,:,0] = img[:,:]

    print('-------------------------------------------')

    # Compute statistics on the input image collection: 
    dmean = np.mean(X_train[:,:,:,0])
    dstdev = np.std(X_train[:,:,:,0])

    print('The input mean is: %f' % dmean)
    print('The input std is: %f' % dstdev)
    print('-------------------------------------------')

    # Compute statistics on the input image collection: 
    domean = np.mean(Y_train[:,:,:,0])
    dostdev = np.std(Y_train[:,:,:,0])

    print('The output mean is: %f' % domean)
    print('The output std is: %f' % dostdev)
    print('-------------------------------------------')

    save_name = working_dir + 'stats_data.npy'
    stats_arr = np.array([dmean, dstdev, domean, dostdev])
    np.save(save_name, stats_arr)
    print('Stats data saved to: ' + save_name)
    print('-------------------------------------------')

    elapsed = time.time() - t
    print('Stats Data: Elapsed time in seconds: %d' % elapsed)
    print('-------------------------------------------')


    return X_train, Y_train
