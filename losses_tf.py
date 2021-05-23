# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

import tensorflow as tf
import numpy as np

import keras.backend as K

def rmse(img, target, mask, num_pixels):

    diff = tf.math.multiply(img - target, mask) / 1000.0    # mapping the distance from millimeters to meters

    loss_mse = tf.reduce_sum(tf.pow(diff, 2)) / num_pixels
    loss_rmse = tf.sqrt(loss_mse)

    return loss_rmse


def si_rmse(img, target, mask, num_pixels):

    log_diff = tf.math.multiply((tf.math.log(img) - tf.math.log(target)), mask)

    loss_si_rmse = tf.sqrt(tf.reduce_sum(tf.square(log_diff)) / num_pixels -
                           tf.square(tf.reduce_sum(log_diff)) / tf.square(num_pixels))

    return loss_si_rmse


def avg_log10(img, target, mask, num_pixels):

    log_diff_10 = tf.math.multiply(((tf.math.log(img) - tf.math.log(target)) / tf.math.log(tf.constant(10.0))), mask)

    loss_log10 = tf.reduce_sum(tf.abs(log_diff_10)) / num_pixels

    return loss_log10


def rel(img, target, mask, num_pixels):

    diff = tf.math.multiply((img - target), mask)

    loss_rel = tf.reduce_sum(tf.math.divide(tf.abs(diff), target)) / num_pixels

    return loss_rel

class Loss():
    def __init__(self, w1=1, w2=1, theta=0.1):
        self.w1 = w1
        self.w2 = w2
        self.theta = theta


    def loss_function(self, y_true, y_pred):
        BATCH_SIZE = tf.shape(y_true)[0]

        target_mask = tf.cast(tf.math.greater(y_true, 1), tf.float32)
        target_mask = tf.reshape(target_mask, [BATCH_SIZE, 480, 640, 1])
        num_pixels = tf.reduce_sum(target_mask)

        y_pred = tf.image.resize(y_pred, size=[480, 640])

        y_pred = y_pred / 1000
        y_true = y_true / 1000

        # Point-wise depth
        l_depth = tf.reduce_sum(tf.math.multiply(K.abs(y_pred - y_true), target_mask)) / num_pixels

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true * target_mask, y_pred * target_mask, max_val=65.335)) * 0.5, 0, 1)

        # Weights
        # w1 = 1.0
        # w2 = 1.0
        # w3 = theta

        return (self.w1 * l_ssim) + (self.w2 * K.mean(l_edges)) + (self.theta * K.mean(l_depth))

    def naive_loss_function(self, target, img):
        img = tf.image.resize(img, size=[480, 640])

        BATCH_SIZE = tf.shape(img)[0]

        target_mask = tf.cast(tf.math.greater(target, 1), tf.float32)
        target_mask = tf.reshape(target_mask, [BATCH_SIZE, 480, 640, 1])
        num_pixels = tf.reduce_sum(target_mask)

        diff = tf.math.multiply(img - target, target_mask) / 1000.0  # mapping the distance from millimeters to meters

        loss_mse = tf.reduce_sum(tf.pow(diff, 2)) / num_pixels
        loss_rmse = tf.sqrt(loss_mse)

        return loss_rmse

    def siremse_function(self, target, img):
        img = tf.image.resize(img, size=[480, 640])

        BATCH_SIZE = tf.shape(img)[0]

        target_mask = tf.cast(tf.math.greater(target, 1), tf.float32)
        target_mask = tf.reshape(target_mask, [BATCH_SIZE, 480, 640, 1])
        num_pixels = tf.reduce_sum(target_mask)

        diff = tf.math.multiply(tf.math.log(img) - tf.math.log(target), target_mask)

        loss_si_rmse = tf.sqrt(tf.reduce_sum(tf.square(diff)) / num_pixels -
                               tf.square(tf.reduce_sum(diff)) / tf.square(num_pixels))

        return loss_si_rmse




