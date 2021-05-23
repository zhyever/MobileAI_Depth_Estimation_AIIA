import numpy as np
from PIL import Image
import tensorflow as tf

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, batch_size=1):
    # Support multiple RGBs, one RGB image, even grayscale 
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    # predictions = model.predict(images, batch_size=batch_size)
    output = model.predict(images, batch_size=batch_size)

    # final output
    predictions = output[0]
    return predictions

def predict_foreval(model, images, batch_size=1):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    # predictions = model.predict(images, batch_size=batch_size)
    output = model.predict(images, batch_size=batch_size)

    # final output
    predictions = output[2]
    return predictions

def predict_forout(model, images, batch_size=1):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    # predictions = model.predict(images, batch_size=batch_size)
    output = model.predict(images, batch_size=batch_size)

    # final output
    predictions = output[1]
    return predictions