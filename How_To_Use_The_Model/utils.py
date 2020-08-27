# Import libraries: 
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def pear_corr(y_true, y_pred):
    # Defines loss function based on Pearson's correlation coefficients. 
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    true_mean = K.mean(y_true)
    pred_mean = K.mean(y_pred)
    tr_v = y_true - true_mean
    pr_v = y_pred - pred_mean

    num = K.sum(tr_v*pr_v,axis=-1,keepdims=True)
    den = K.sqrt(K.sum(K.square(tr_v)))*K.sqrt(K.sum(K.square(pr_v)))
    return (-1.0*num/den+1.0)/2.0

def get_normalization_factors(cell_type, magnification, output_var):
    # Get mean and standard deviation for various datasets. 

    if(cell_type == 'MDCK'):
        # 256x256
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 1
        input_var = 'PHASE/'
        if(magnification == '5x'):
            TRAIN_PATH = '/scratch/gpfs/jl40/Nuclei5x/Training/'
            TEST_PATH = '/scratch/gpfs/jl40/Nuclei5x/Testing/'
            out_dir = './master/MDCK_5x/'
            input_mean = 36419.63082
            input_stdev = 7872.98629
            if(output_var == 'DAPI'):
                output_var = 'DAPI/'
                output_mean = 7642.33462
                output_stdev = 6937.91811

        # 256x256
        if(magnification == '20x'):
            out_dir = './master/MDCK_20x/'
            input_mean = 8180.65217
            input_stdev = 565.37316
            TRAIN_PATH = '/tigress/jl40/Cadherin_20x_5_14_2019/Training/'
            TEST_PATH = '/tigress/jl40/Cadherin_20x_5_14_2019/Testing/'
            input_var = 'DIC/'
            if(output_var == 'RFP'):
                output_var = 'RFP/'
                output_mean = 627.32657
                output_stdev = 151.28858
            if(output_var == 'DAPI'):
                output_var = 'DAPI/'
                output_mean = 1789.79255
                output_stdev = 2215.36882

    if(cell_type == 'KC'):
        # 256x256
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 1
        TRAIN_PATH = '/tigress/jl40/KC_2/Training/'
        TEST_PATH = '/tigress/jl40/KC_2/Testing/'
        if(magnification == '10x'):
            out_dir = './master/KC_10x/'
            input_var = 'PHASE/'
            input_mean = 6201.78972
            input_stdev = 1944.04150
            if(output_var == 'DAPI'):
                output_var = 'DAPI/'
                output_mean = 2787.14538
                output_stdev = 3441.03697

    if(cell_type == 'HUVECS_256'):
        # 256x256
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 1
        if(magnification == '20x'):
            input_mean = 10920.42865
            input_stdev = 492.43607
            TRAIN_PATH = '/scratch/gpfs/jl40/HUVECS_Small/Training/'
            TEST_PATH = '/scratch/gpfs/jl40/HUVECS_Small/Testing/'
            input_var = 'DIC/'
            out_dir = './master/HUVECS_256/'
            if(output_var == 'CY5'):
                output_var = 'CY5/'
                output_mean = 898.55859
                output_stdev = 394.94965
            if(output_var == 'YFP'):
                output_var = 'YFP/'
                output_mean = 510.58722
                output_stdev = 160.74766
            if(output_var == 'DAPI'):
                output_var = 'DAPI/'
                output_mean = 482.32611
                output_stdev = 255.59408

    return IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS, out_dir, TRAIN_PATH, TEST_PATH, input_var, output_var, input_mean, input_stdev, output_mean, output_stdev


def get_weights_file_standard_nonorm(cell_type, magnification, output_var, use_loss):
    # Gets the weight files for the 1-stack U-Net experiments. 
    if(cell_type == 'MDCK'):
        if(magnification == '5x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'nonorm_mdck_5x_dapi_1stack_unet_p.h5'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_mdck_5x_dapi_1stack_unet_mse.h5'

        if(magnification == '20x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'nonorm_mdck_20x_dapi_1stack_unet_pear.h5'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_mdck_20x_dapi_1stack_unet_mse.h5'

            if(output_var == 'RFP'):
                if(use_loss == 'p'):
                    weights_file = 'nonorm_mdck_20x_rfp_1stack_unet_pear.h5'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_mdck_20x_rfp_1stack_unet_mse.h5'

    if(cell_type == 'KC'):
        if(magnification == '10x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_kc_10x_dapi_1stack_unet_mse.h5'

    if(cell_type == 'HUVECS_256'):
        if(magnification == '20x'):
            if(output_var == 'CY5'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'None'

            if(output_var == 'YFP'):
                if(use_loss == 'p'):
                    weights_file = 'nonorm_huvecs_20x_yfp_1stack_unet_pear.h5'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_huvecs_20x_yfp_1stack_unet_mse.h5'

            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_huvecs_20x_dapi_1stack_unet_mse.h5'

            if(output_var == 'CY5'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_huvecs_20x_cy5_1stack_unet_mse.h5'

    return weights_file

def get_weights_file_2stacknet_nonorm(cell_type, magnification, output_var, use_loss):
    # Gets the weight files for the 2-stack U-Net experiments (when applicable). 
    if(cell_type == 'MDCK'):
        if(magnification == '5x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_2STACK_mdck_5x_dapi_2stackunet_mse.h5'

        if(magnification == '20x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_2STACKunet_mdck_20x_dapi_mse.h5'

            if(output_var == 'RFP'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'nonorm_2STACKunet_mdck_20x_rfp_mse.h5'

    if(cell_type == 'KC'):
        if(magnification == '10x'):
            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'None'

    if(cell_type == 'HUVECS_256'):
        if(magnification == '20x'):
            if(output_var == 'CY5'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'None'

            if(output_var == 'YFP'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'None'

            if(output_var == 'DAPI'):
                if(use_loss == 'p'):
                    weights_file = 'None'
                if(use_loss == 'mse'):
                    weights_file = 'None'
    return weights_file

def get_signal_thres(cell_type, magnification, output_var):
    # Gets the cutoff intensity thresholds which indicate presence of features (nuclei, junctions, etc.) in image. 
    if(cell_type == 'MDCK'):
        if(magnification == '5x'):
            signal_thres = 14000    
        if(magnification == '20x'):
            if(output_var == 'DAPI'):
                signal_thres = 6000
            if(output_var == 'RFP'):
                signal_thres = 1000
    if(cell_type == 'KC'):
        signal_thres = 6000
    if(cell_type == 'HUVECS_256'):
        if(output_var == 'DAPI'):
            signal_thres = 1000
        if(output_var == 'YFP'):
            signal_thres = 800
        if(output_var == 'CY5'):
            signal_thres = 1000
    return signal_thres

if __name__ == '__main__':
    pear_corr(y_true, y_pred)
