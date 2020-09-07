
# Splits the raw input/output images and saves them as 256x256 pixel^2 images in folders called Split_Input and Split_Output within the working directory. 

import os
from PIL import Image
import numpy as np
import time

from source import frm_utils

def crop_and_save(input_dir, working_dir, input_image):
    print('Cropping image: ', input_image)
    image = Image.open(input_dir + input_image)

    im_width, im_height = image.size
    print('Width = ' + str(im_width) +', Height = ' + str(im_height))
    crop_width = int(np.floor(im_width / 256) * 256)
    crop_height = int(np.floor(im_width / 256) * 256)

    cropped_image = image.crop((0,0,256,256))
    M=256
    N=256
    tiles = [image.crop((x, y, x+M,y+N)) for x in range(0,crop_width,M) for y in range(0,crop_height,N)]
    print('Number of tiles: ' + str(len(tiles)))
    print('-------------------------------------------')
    
    for i in range(len(tiles)):
        save_name = working_dir + 'crop_' + str(i).zfill(5) + '_' + input_image
        tiles[i].save(save_name)

def do_split(input_dir, output_dir, working_dir):
    t = time.time()
    print('-------------------------------------------')
    print('Splitting raw input images....')
    print('-------------------------------------------')
    
    in_working_dir = working_dir + 'Split_Input/'
    if not os.path.exists(in_working_dir):
        os.makedirs(in_working_dir)
    print('Saving split input data to: ' + in_working_dir)
    print('-------------------------------------------')
    
    input_image_list = sorted(frm_utils.get_file_list_from_dir(input_dir))
    for i in range(len(input_image_list)):
        crop_and_save(input_dir, in_working_dir, input_image_list[i])

    print('-------------------------------------------')
    print('Splitting raw output images....')
    print('-------------------------------------------')
    
    out_working_dir = working_dir + 'Split_Output/'
    if not os.path.exists(out_working_dir):
        os.makedirs(out_working_dir)
    print('Saving split input data to: ' + out_working_dir)
    print('-------------------------------------------')
    
    output_image_list = sorted(frm_utils.get_file_list_from_dir(output_dir))
    for i in range(len(output_image_list)):
        crop_and_save(output_dir, out_working_dir, output_image_list[i])

    if(len(input_image_list) != len(output_image_list)):
        print('Raw input and output images not the same size. Stop and check your data.')

    elapsed = time.time() - t
    print('Data Split: Elapsed time in seconds: %d' % elapsed)
    print('-------------------------------------------')


    return
