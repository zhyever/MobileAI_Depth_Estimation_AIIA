import os
import glob
import time
import argparse
import numpy as np
from PIL import Image
import cv2

# Kerasa / TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from keras.models import load_model
from layers import BilinearUpSampling2D
from model_unet_attention import create_model
import tensorflow as tf

import os, sys, glob, time, pathlib, argparse
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--gpu', type=str, default="0", help='device')
parser.add_argument('--non_local', type=bool, default=False, help='if use nonlocal')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def DepthNorm(x, maxDepth):
    return maxDepth / x


save_path = 'res/'
# Load model into GPU / CPU
print('Loading model...')
# model = load_model("/home/lzyever/python_workspace/DenseDepth-master/models/1612158189-n1287-e100-bs4-lr0.0001-densedepth_nyu/weights.100-1.03.hdf5", custom_objects=custom_objects, compile=False)
model = create_model()
model.load_weights('models/1615770381-n2462-e100-bs3-lr0.0001-xxx/weights.40-3.33.hdf5')

# images = np.asarray(Image.open('/home/lzyever/datas/MobileAI/train/rgb/5.png')).reshape(480,640,3)/255
with open('val.txt', 'r') as f:
    lines = f.read().splitlines()

ref_time = 0
nums = len(lines)

for i in lines:
    img_path = os.path.join(os.path.expanduser('~'), i)
    image = np.asarray(Image.open(img_path)).reshape(480, 640, 3) / 255
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    time_start = time.time()
    prediction = model.predict(image, batch_size=2)[1]
    time_end = time.time()

    per_time = time_end - time_start
    ref_time += per_time

print(ref_time/nums)

