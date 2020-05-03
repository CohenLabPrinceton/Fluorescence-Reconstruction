# Fluorescence Reconstruction Resources: 

All source code for this project can be found in the folder entitled "Code_For_Distribution", along with the User Manual. The User Manual contains instructions for training and testing a U-Net and processing new data. 

Our complete testing dataset, along with corresponding reconstructed images, can be found at:

http://doi.org/10.5281/zenodo.3783678


If you prefer a smaller dataset for rapid testing, we provide subsets of some of our data at the sources below.

A sample dataset for testing our code may be found at:

http://arks.princeton.edu/ark:/88435/dsp019w032593v

This DataSpace location contains a set of 1,000 matched image pairs from our keratinocyte dataset, imaged at 10x magnification. Phase contrast images (input) are paired with fluorescent images (output) of Hoechst 33342-stained nuclei, as imaged in DAPI.

Users may alternatively examine a sample dataset of MDCK cells, imaged at 20x:

 http://arks.princeton.edu/ark:/88435/dsp019880vt87x
 
Here, DIC images (input) and paired with fluorescent images (output) corresponding to either nuclei or E-cadherin cell-cell junctions. 


Users may find pre-trained weights for our standard U-Net implementation in the folder entitled "Pretrained_Weights". The README in that folder indicates which weights correspond to which experimental condition.
