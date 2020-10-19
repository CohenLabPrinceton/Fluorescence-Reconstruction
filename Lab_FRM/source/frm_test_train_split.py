
# Takes the 256x256 pixel^2 images in folders called Split_Input and Split_Output within the working directory, and performs test/train splits. Training and test sets are saved within "Training" and "Testing" within the working directory. Then the split directories are deleted. 

import os
import random
from random import Random
from random import shuffle
from math import floor
from shutil import copyfile, rmtree
import time

from source import frm_utils

# Helper function for performing the test/train split: 
def get_training_and_testing_sets(file_list, split):
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

# Helper function for copying files into a new directory: 
def get_and_copy_files(file_list, old_dir, new_dir):
    for x in file_list:
        src = old_dir + x
        dst = new_dir + x
        copyfile(src, dst)

def do_test_train_split(working_dir):
    t = time.time()
    print('Running test/train split....')
    print('-------------------------------------------')
    split = 0.8

    # Set the paths: 
    input_path = working_dir + 'Split_Input/'
    output_path = working_dir + 'Split_Output/'

    train_in = working_dir + 'Training/Input_Var/'
    test_in = working_dir + 'Testing/Input_Var/'
    train_out = working_dir + 'Training/Output_Var/'
    test_out = working_dir + 'Testing/Output_Var/'

    # If the output paths do not exist, create them: 
    if not os.path.exists(train_in):
        os.makedirs(train_in)
    if not os.path.exists(test_in):
        os.makedirs(test_in)
    if not os.path.exists(train_out):
        os.makedirs(train_out)
    if not os.path.exists(test_out):
        os.makedirs(test_out)

    # Get the input images: 
    input_var_list = frm_utils.get_file_list_from_dir(input_path)
    output_var_list = frm_utils.get_file_list_from_dir(output_path)

    # Sort the images by name: 
    input_var_list = sorted(input_var_list)
    output_var_list = sorted(output_var_list)

    # Randomly shuffle the images in the same way:
    seed = 42
    random.Random(seed).shuffle(input_var_list)
    random.Random(seed).shuffle(output_var_list)

    # Do the test/train splits: 
    input_var_train, input_var_test = get_training_and_testing_sets(input_var_list, split)
    output_var_train, output_var_test = get_training_and_testing_sets(output_var_list, split)

    # Copy the files into the right (test/train) folders. 
    get_and_copy_files(input_var_train, input_path, train_in)
    get_and_copy_files(input_var_test, input_path, test_in)
    get_and_copy_files(output_var_train, output_path, train_out)
    get_and_copy_files(output_var_test, output_path, test_out)

    print('There are: %d images in the input training list.' % len(input_var_train))
    print('There are: %d images in the input testing list.' % len(input_var_test))
    print('There are: %d images in the output training list.' % len(output_var_train))
    print('There are: %d images in the output testing list.' % len(output_var_test))
    print('-------------------------------------------')

    try:
        rmtree(input_path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    try:
        rmtree(output_path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

    elapsed = time.time() - t
    print('Test-Train Split: Elapsed time in seconds: %d' % elapsed)
    print('-------------------------------------------')


    return


