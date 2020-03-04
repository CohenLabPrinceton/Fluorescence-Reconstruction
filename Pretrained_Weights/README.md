This folder contains the pre-trained weights for our standard U-Net implementation, trained with a mean-squared error (MSE) loss function, for the suite of datasets described in our paper. 

The weight files have names such as: "mdck_5x_dapi_1stack_unet_mse.h5" 
They are named using the format:
"{cell type}_{magnification}_{fluorescent filter set}_1stack_unet_mse.h5"

The complete list is as follows:

MDCK cells, 5x magnification, Hoechst 33342-stained nuclei:
mdck_5x_dapi_1stack_unet_mse.h5

KC cells, 10x magnification, Hoechst 33342-stained nuclei:
kc_10x_dapi_1stack_unet_mse.h5

MDCK cells, 20x magnification, Hoechst 33342-stained nuclei:
mdck_20x_dapi_1stack_unet_mse.h5

MDCK cells, 20x magnification, E-cadherin junctions:
mdck_20x_rfp_1stack_unet_mse.h5

HUVEC cells, 20x magnification, Hoechst 33342-stained nuclei:
huvecs_20x_dapi_1stack_unet_mse.h5

HUVEC cells, 20x magnification, VE-Cadherin:
huvecs_20x_yfp_1stack_unet_mse.h5

HUVEC cells, 20x magnification, F-actin:
huvecs_20x_cy5_1stack_unet_mse.h5
