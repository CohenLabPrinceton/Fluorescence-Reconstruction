For ease of Python library use, we provide an example Dockerfile. Users may pull library installation commands directly from the Dockerfile (e.g. "pip3 install tensorflow"), or modify the Dockerfile to create their own custom image for running Python scripts. 

To build with the Dockerfile, we run the command:
docker build -f Dockerfile -t frm_container .

To run the container, we use the command:
docker run frm_container

The provided example runs a simple script which will print "Hello World!" when run from a terminal. Users may change this to run alternative Python scripts within the image. We also provide helper files if users wish to initiate Jupyter notebook sessions from the docker. 
