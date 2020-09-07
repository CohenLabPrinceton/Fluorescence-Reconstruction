
This folder contains cleaned FRM code for anyone who just wants to capture some data and get a model trained quickly. 

To train a model, download the contents of this folder, set up the parameters in "main.py" and run it as a Python script. 
Once you trained the model, you can process new images using "process_new_images.py" in the same way. 

For example: "python3 main.py"

If you're in the CohenLab, we provide the .cmd script for running this code in the Princeton Tigergpu cluster. Check out "run_main.cmd" and change it to use your email address, and to output the log as you like. 

Make sure to capture matched image pairs for training which represent real data (e.g. no artificial black boundaries, only real background) and that the images really represent the same spatial region of the tissue/dish. Good luck! 
