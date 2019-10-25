# Import libraries: 
import os
import random
from random import Random
from random import shuffle
from math import floor
from shutil import copyfile

'''
This script performs test/train splits on your data. 
That is, it takes as input paths to your data, and splits it into a training set and a test set. 
The default setting here is for a random 80% of the data to be used as the training set, 
   with 20% held out for the test set. 

User-defined inputs: 
base_path - (string) the folder containing the input/output image folders. 
        The image folders are named according to their image type: example, ./path/PHASE/ or ./path/DAPI/
input_var - (string) the input variable name. Example: 'PHASE' or 'DIC'
output_var - (string) the output variable name. Example: 'DAPI' or 'RFP'
split - (float) the percentage of the dataset to include in the training set. 
        The remainder will be used in the test set. 
        We recommend 0.8 or 0.9 be used for this value. 
        
Images will be copied into either a Training or Testing folder, accordingly.         
'''

# -------------------------------------------------
# User-defined inputs: 
base_path = '../1_Splitting_Images_Into_Patches/Sample_Images/'
input_var = 'PHASE'
output_var = 'DAPI'
split = 0.5
# -------------------------------------------------

# Set the paths: 
input_path = base_path + input_var + '/'
output_path = base_path + output_var + '/'

train_in = base_path + 'Training/' + input_var + '/'
test_in = base_path + 'Testing/' + input_var + '/'
train_out = base_path + 'Training/' + output_var + '/'
test_out = base_path + 'Testing/' + output_var + '/'

# If the output paths do not exist, create them: 
if not os.path.exists(train_in):
    print('creating directory')
    os.makedirs(train_in)
if not os.path.exists(test_in):
    os.makedirs(test_in)
if not os.path.exists(train_out):
    os.makedirs(train_out)
if not os.path.exists(test_out):
    os.makedirs(test_out)

# Helper function to get files from a directory: 
def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: file.endswith('.tif'), all_files))
    return data_files

# Get the input images: 
phase_list = get_file_list_from_dir(input_path)
dapi_list = get_file_list_from_dir(output_path)

# Sort the images by name: 
phase_list = sorted(phase_list)
dapi_list = sorted(dapi_list)

print(len(phase_list))

# Randomly shuffle the images in the same way:
seed = 42
random.Random(seed).shuffle(phase_list)
random.Random(seed).shuffle(dapi_list)

# Helper function for performing the test/train split: 
def get_training_and_testing_sets(file_list, split):
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

# Do the test/train splits: 
phase_train, phase_test = get_training_and_testing_sets(phase_list, split)
dapi_train, dapi_test = get_training_and_testing_sets(dapi_list, split)

# Helper function for copying files into a new directory: 
def get_and_copy_files(file_list, old_dir, new_dir):
    for x in file_list:
        src = old_dir + x
        dst = new_dir + x
        copyfile(src, dst)

# Copy the files into the right (test/train) folders. 
get_and_copy_files(phase_train, input_path, train_in)
get_and_copy_files(phase_test, input_path, test_in)
get_and_copy_files(dapi_train, output_path, train_out)
get_and_copy_files(dapi_test, output_path, test_out)


