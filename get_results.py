import os
import glob
import time
import argparse
import numpy as np
from PIL import Image
import cv2
from model_unet_attention import create_model

# Kerasa / TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.models import load_model
from layers import BilinearUpSampling2D
import tensorflow as tf


def DepthNorm(x, maxDepth):
    return maxDepth / x


import os, sys, glob, time, pathlib, argparse
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--gpu', type=str, default="0", help='device')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



save_path = 'res/'
# Load model into GPU / CPU
print('Loading model...')
# model = load_model("/home/lzyever/python_workspace/DenseDepth-master/models/1612158189-n1287-e100-bs4-lr0.0001-densedepth_nyu/weights.100-1.03.hdf5", custom_objects=custom_objects, compile=False)
model = create_model()
model.load_weights('models/1615770381-n2462-e100-bs3-lr0.0001-xxx/weights.40-3.33.hdf5')

# images = np.asarray(Image.open('/home/lzyever/datas/MobileAI/train/rgb/5.png')).reshape(480,640,3)/255
with open('val.txt', 'r') as f:
    lines = f.read().splitlines()
for i in lines:
    img_path = os.path.join(os.path.expanduser('~'), i)
    image = np.asarray(Image.open(img_path)).reshape(480, 640, 3) / 255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    prediction = model.predict(image, batch_size=2)[1]
    # pred = tf.cast(tf.clip_by_value(prediction, 0.0, 65535.0), tf.uint16)
    pred = np.array(prediction)

    pred = pred.reshape(int(image.shape[1]), int(image.shape[2]), 1)
    # print(pred.dtype)
    # pred = pred.astype(int)
    pred = pred.astype(np.uint16)
    # print(model.predict(image, batch_size=2)[0])

    # print(i[23:])
    cv2.imwrite(save_path + i[23:], pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])

