
import tensorflow as tf
import numpy as np
import imageio
import os
import keras
# from model import create_model, create_mobilev2_model
from model_unet_attention import create_model
from losses_tf import *
from data import get_train_test_data
from callbacks import *
import time
import pathlib
# from keras.utils import multi_gpu_model

import os, sys, glob, time, pathlib, argparse
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--w1', type=float, default=0.1, help='w1')
parser.add_argument('--w2', type=float, default=0.1, help='w2')
parser.add_argument('--w3', type=float, default=0.1, help='w3')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--gpu', type=str, default="0", help='device')
parser.add_argument('--name', type=str, default="xxx", help='name')
parser.add_argument('--log_val', type=str, default="./log/val_res.txt", help='name')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


f2 = open(args.log_val, 'a')
f2.write("Var starts \n")
f2.close()


# os.environ['CUDA_VISIBLE_DEVICES']="0"

BATCH_SIZE = 3
EPOCH = args.epoch
NAME = args.name
LR = 1e-4

model = create_model()

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

loss = Loss(args.w1, args.w2, args.w3)
model.compile(loss=[loss.my_loss_function, None, None], optimizer=optimizer,
              loss_weights=[1, 0, 0])

callbacks = get_callbacks(model, model, train_generator, test_generator, False , runPath, args.log_val)
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=EPOCH, shuffle=True)
model.save(runPath + '/model.h5')

