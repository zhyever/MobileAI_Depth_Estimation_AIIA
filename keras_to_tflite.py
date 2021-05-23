# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to convert a simple Keras U-Net based model to TFLite format

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model
from keras.optimizers import Adam
from model import create_model, create_mobilev2_model

import os, sys, glob, time, pathlib, argparse
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--gpu', type=str, default="0", help='device')
parser.add_argument('--non_local', type=bool, default=False, help='if use nonlocal')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



def convert_model():

    # Defining the model
    model = create_mobilev2_model(args)

    # Load your pre-trained model
    model.load_weights('models/1614868247-n858-e100-bs6-lr0.0001-robust_new/weights.10-1.36.hdf5')

    # Export your model to the TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Be very careful here:
    # "experimental_new_converter" is enabled by default in TensorFlow 2.2+. However, using the new MLIR TFLite
    # converter might result in corrupted / incorrect TFLite models for some particular architectures. Therefore, the
    # best option is to perform the conversion using both the new and old converter and check the results in each case:
    converter.experimental_new_converter = False

    tflite_model = converter.convert()
    open("model.tflite", "wb").write(tflite_model)

    # -----------------------------------------------------------------------------
    # That's it! Your model is now saved as model.tflite file
    # You can now try to run it using the PRO mode of the AI Benchmark application:
    # https://play.google.com/store/apps/details?id=org.benchmark.demo
    # More details can be found here (RUNTIME VALIDATION):
    # https://ai-benchmark.com/workshops/mai/2021/#runtime
    # -----------------------------------------------------------------------------


convert_model()
