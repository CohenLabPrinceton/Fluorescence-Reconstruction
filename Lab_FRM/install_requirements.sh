#!/bin/bash

apt-get update && apt-get install -y \
    libopencv-dev \
    python3-pip \
    python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Install Python library requirements: 
pip3 install tensorflow && \
    pip3 install numpy sklearn matplotlib jupyter pyyaml h5py tqdm && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils && \
    pip3 install libtiff && \
    pip3 install Pillow
