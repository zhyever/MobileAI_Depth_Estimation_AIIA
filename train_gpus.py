
import tensorflow as tf
import numpy as np
import imageio
import os
import keras
from model import create_model
from model import *
from losses_tf import *
from data import get_train_test_data
from callbacks import *
import time
import pathlib
from keras.utils import multi_gpu_model


os.environ['CUDA_VISIBLE_DEVICES']="0, 1"
BATCH_SIZE = 12
EPOCH = 100
NAME = "depthbaseline"
LR = 1e-4
GPUS = 2


with tf.device('/cpu:0'):
    model = My_model()
    # model = create_model()

multi_model = multi_gpu_model(model, gpus=GPUS)


train_generator, test_generator = get_train_test_data(BATCH_SIZE)
optimizer = keras.optimizers.Adam(lr=LR, amsgrad=True)


# Training session details
runID = str(int(time.time())) + '-n' + \
        str(len(train_generator)) + '-e' + \
        str(EPOCH) + '-bs' + str(BATCH_SIZE) + '-lr' + \
        str(LR) + '-' + NAME
outputPath = './models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

# model.compile(loss=[loss_function, loss_function, loss_function, loss_function,
#                     None, None, None, None], optimizer=optimizer,
#               loss_weights=[1, 1, 1, 1, 0, 0, 0, 0])

multi_model.compile(loss=[loss_function, None], optimizer=optimizer,
              loss_weights=[1, 0])

callbacks = get_callbacks(multi_model, multi_model, train_generator, test_generator, False , runPath)
multi_model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=EPOCH, shuffle=True)
multi_model.save(runPath + '/model.h5')

