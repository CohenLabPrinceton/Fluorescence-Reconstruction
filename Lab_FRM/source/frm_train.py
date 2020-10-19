
import time
import numpy as np

# ML dependencies
import keras
from keras import optimizers
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras import backend as K

from source import frm_models

def do_train(working_dir, weights_file, X_train, Y_train):
    t = time.time()
    print('Training the model....')
    print('-------------------------------------------')

    # Get normalization parameters:
    stats_data = np.load(working_dir + 'stats_data.npy')
    in_mean = stats_data[0]
    in_stdev = stats_data[1]
    out_mean = stats_data[2]
    out_stdev = stats_data[3]

    # Normalize data: 
    X_train = (X_train - in_mean) / in_stdev
    Y_train = (Y_train - out_mean) / out_stdev

    # Get model:
    model = frm_models.get_unet(256, 256, 1)

    # Set up model parameters and optimization settings:
    max_epochs_no = 10000
    adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    loss_func = 'mean_squared_error'
    weights_file = working_dir + weights_file

    # Compile model: 
    model.compile(loss=loss_func,
              optimizer=adad,
              metrics=['mse'])

    # Stop the model if validation loss has not decreased in the last 100 epochs:
    earlystopper = EarlyStopping(patience=100, verbose=1)
    # Save Checkpoints: 
    checkpointer = ModelCheckpoint(weights_file, verbose=1, save_best_only=True)
    # Log the losses to a .csv file: 
    cLog = CSVLogger(working_dir + 'training_logger.csv')

    # Train the model:
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=max_epochs_no, 
                    callbacks=[earlystopper, checkpointer, cLog])

    elapsed = time.time() - t
    print('-------------------------------------------')
    print('Training: Elapsed time in seconds: %d' % elapsed)
    print('-------------------------------------------')

    return
