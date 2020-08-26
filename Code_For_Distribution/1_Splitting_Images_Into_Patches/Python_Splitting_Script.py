import os
from PIL import Image
import numpy as np

# Set the image path and output directory
input_dir = './Sample_Images/Phase_Image/'
output_dir = './Sample_Images/PHASE/'

def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda file: (file.endswith('.tif') or file.endswith('.tiff')), all_files))
    return data_files

input_image_list = sorted(get_file_list_from_dir(input_dir))

def crop_and_save(input_dir, input_image):
    print('Cropping image: ', input_image)
    image = Image.open(input_dir + input_image_list[0])
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
        save_name = output_dir + 'crop_' + str(i).zfill(5) + '_' + input_image
        tiles[i].save(save_name, compression='tiff_raw_16')
        
for i in range(len(input_image_list)):
    crop_and_save(input_dir, input_image_list[i])

