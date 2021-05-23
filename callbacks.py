
from losses_tf import *
import io
import random
import numpy as np
from PIL import Image

import keras
from keras import backend as K
from utils import DepthNorm, predict_foreval, predict

import tensorflow as tf

def call_rmse(img, target, mask, num_pixels):
    diff = tf.math.multiply(tf.cast(img, tf.double) -
                            tf.cast(target, tf.double), mask) / 1000.0    # mapping the distance from millimeters to meters

    loss_mse = tf.reduce_sum(tf.pow(diff, 2)) / num_pixels
    loss_rmse = tf.sqrt(loss_mse)

    return loss_rmse


def call_si_rmse(img, target, mask, num_pixels):
    log_diff = tf.math.multiply((tf.cast(tf.math.log(img), tf.double) -
                                 tf.cast(tf.math.log(target), tf.double)), mask)

    loss_si_rmse = tf.sqrt(tf.reduce_sum(tf.square(log_diff)) / num_pixels -
                           tf.square(tf.reduce_sum(log_diff)) / tf.square(num_pixels))

    return loss_si_rmse


def call_avg_log10(img, target, mask, num_pixels):
    log_diff_10 = tf.math.multiply(((tf.cast(tf.math.log(img), tf.double) -
                                     tf.cast(tf.math.log(target), tf.double))
                                    / tf.cast(tf.math.log(tf.constant(10.0)),tf.double)), mask)

    loss_log10 = tf.reduce_sum(tf.abs(log_diff_10)) / num_pixels

    return loss_log10


def call_rel(img, target, mask, num_pixels):
    diff = tf.math.multiply((tf.cast(img, tf.double) -
                             tf.cast(target, tf.double)), mask)

    loss_rel = tf.reduce_sum(tf.math.divide(tf.abs(diff), target)) / num_pixels

    return loss_rel

def make_image(tensor):
    image = Image.fromarray(tensor.astype('uint8'))
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=90)
    output.close()

def get_callbacks(model, basemodel, train_generator, test_generator, test_set, runPath, log_val):
    callbacks = []

    # Callback: Tensorboard
    class Eval(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()

            self.num_samples = 6
            self.train_idx = np.random.randint(low=0, high=len(train_generator), size=10)
            self.test_idx = np.random.randint(low=0, high=len(test_generator), size=10)

        def on_epoch_begin(self, epoch, logs=None):

            from skimage.transform import resize

            minDepth, maxDepth = 10, 1000

            train_samples = []
            test_samples = []

            for i in range(self.num_samples):
                x_train, y_train = train_generator.__getitem__(self.train_idx[i], False)
                x_test, y_test = test_generator[self.test_idx[i]]

                # x_train, y_train = x_train[0], np.clip(DepthNorm(y_train[0], maxDepth=1000), minDepth, maxDepth) / maxDepth
                # x_test, y_test = x_test[0], np.clip(DepthNorm(y_test[0], maxDepth=1000), minDepth, maxDepth) / maxDepth

                x_train, y_train = x_train[0], y_train[0]
                x_test, y_test = x_test[0], y_test[0]

                h, w = y_train.shape[0], y_train.shape[1]

                rgb_train = resize(x_train, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
                rgb_test = resize(x_test, (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)

                gt_train = y_train[:,:,0]
                gt_test = y_test[:,:,0]

                predict_train = predict(model, x_train)[0,:,:,0]
                predict_test = predict(model, x_test)[0,:,:,0]

                train_samples.append([rgb_train, gt_train, predict_train])
                test_samples.append([rgb_test, gt_test, predict_test])

            train_rmse = 0
            train_si_rmse = 0
            train_avg_log10 = 0
            train_rel = 0

            test_rmse = 0
            test_si_rmse = 0
            test_avg_log10 = 0
            test_rel = 0

            for i in range(self.num_samples):

                target = train_samples[i][1]
                train_target_mask = tf.cast(tf.math.greater(target, 1), tf.double)
                train_target_mask = tf.reshape(train_target_mask, [480, 640])
                train_num_pixels = tf.reduce_sum(train_target_mask)

                target = test_samples[i][1]
                test_target_mask = tf.cast(tf.math.greater(target, 1), tf.double)
                test_target_mask = tf.reshape(test_target_mask, [480, 640])
                test_num_pixels = tf.reduce_sum(test_target_mask)

                # call_loss(train_samples[i][2], train_samples[i][1], train_target_mask, train_num_pixels)

                train_rmse += call_rmse(train_samples[i][2], train_samples[i][1], train_target_mask, train_num_pixels)
                train_si_rmse += call_si_rmse(train_samples[i][2], train_samples[i][1], train_target_mask, train_num_pixels)
                train_avg_log10 += call_avg_log10(train_samples[i][2], train_samples[i][1], train_target_mask, train_num_pixels)
                train_rel += call_rel(train_samples[i][2], train_samples[i][1], train_target_mask, train_num_pixels)

                test_rmse += call_rmse(test_samples[i][2], test_samples[i][1], test_target_mask, test_num_pixels)
                test_si_rmse += call_si_rmse(test_samples[i][2], test_samples[i][1], test_target_mask, test_num_pixels)
                test_avg_log10 += call_avg_log10(test_samples[i][2], test_samples[i][1], test_target_mask, test_num_pixels)
                test_rel += call_rel(test_samples[i][2], test_samples[i][1], test_target_mask, test_num_pixels)

            train_rmse /= self.num_samples
            train_si_rmse /= self.num_samples
            train_avg_log10 /= self.num_samples
            train_rel /= self.num_samples

            test_rmse /= self.num_samples
            test_si_rmse /= self.num_samples
            test_avg_log10 /= self.num_samples
            test_rel /= self.num_samples

            logs_losses1 = "Train | RMSE: %.4g, SI_RMSE: %.4g, LOG_10: %.4g, REL: %.4g \n" % \
                          (train_rmse, train_si_rmse, train_avg_log10, train_rel)

            str1 = "%.4g %.4g %.4g %.4g \n" % \
                           (train_rmse, train_si_rmse, train_avg_log10, train_rel)

            logs_losses2 = "Test | RMSE: %.4g, SI_RMSE: %.4g, LOG_10: %.4g, REL: %.4g \n" % \
                          (test_rmse, test_si_rmse, test_avg_log10, test_rel)

            str2 = "%.4g %.4g %.4g %.4g \n" % \
                   (test_rmse, test_si_rmse, test_avg_log10, test_rel)

            # f1 = open('./train_res.txt', 'a')
            # f1.write(str1)
            # f1.close()

            f2 = open(log_val, 'a')
            f2.write(str2)
            f2.close()

            print(logs_losses1)
            print(logs_losses2)

    eval = Eval()
    callbacks.append(eval)

    # Callback: Learning Rate Scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=0.00009, min_delta=1e-2)
    callbacks.append(lr_schedule) # reduce learning rate when stuck

    # Callback: save checkpoints
    callbacks.append(keras.callbacks.ModelCheckpoint(runPath + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
        verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=5))

    return callbacks
