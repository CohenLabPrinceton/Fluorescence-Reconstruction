FROM ubuntu:20.04
MAINTAINER "Julienne LaChance"

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

# Install Python library requirements: 
RUN pip3 install tensorflow && \
    pip3 install numpy sklearn matplotlib jupyter pyyaml h5py tqdm && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils && \
    pip3 install libtiff && \
    pip3 install Pillow

# Add the file you wish to run: 
ADD ./hello.py /

# Set up Jupyter notebook: 
COPY ./jupyter_notebook_config.py /root/.jupyter
COPY run_jupyter.sh /

# Expose Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Run the example Python script: 
CMD [ "python3", "./hello.py" ]
