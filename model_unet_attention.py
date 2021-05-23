import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate,Add,Reshape
from layers import BilinearUpSampling2D
import tensorflow as tf
import keras
from keras import models, layers, regularizers
from keras import backend as K


def create_model(is_twohundred=False, is_halffeatures=True):
    print('Loading base model (DenseNet)..')

    # Encoder Layers
    if is_twohundred:
        base_model = applications.DenseNet201(input_shape=(480, 640, 3), include_top=False)
    else:
        base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)

    print('Base model loaded.')

    # Starting point for decoder
    base_model_output_shape = base_model.layers[-1].output.shape

    # Layer freezing?
    for layer in base_model.layers: layer.trainable = True

    # Starting number of decoder filters
    if is_halffeatures:
        decode_filters = int(int(base_model_output_shape[-1]) / 2)
    else:
        decode_filters = int(base_model_output_shape[-1])

    def expend_as(tensor, rep):
        return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                             arguments={'repnum': rep})(tensor)

    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
        up_i = Concatenate(name=name + '_concat')(
            [up_i, concat_with])  # Skip connection
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convA')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name + '_convB')(up_i)
        up_i = LeakyReLU(alpha=0.2)(up_i)
        return up_i

    def global_non_local(X,cc):
        h, w , c = list(X.shape)[1], list(X.shape)[2], list(X.shape)[3]
        c=cc
        theta = Conv2D(c, kernel_size=(1,1), padding='same')(X)
        theta_rsh = Reshape((h*w, c))(theta)

        phi = Conv2D(c, kernel_size=(1,1), padding='same')(X)
        phi_rsh = Reshape((c, h*w))(phi)

        g = Conv2D(c, kernel_size=(1,1), padding='same')(X)
        g_rsh = Reshape((h*w, c))(g)

        theta_phi = tf.matmul(theta_rsh, phi_rsh)
        theta_phi = tf.keras.layers.Softmax()(theta_phi)

        theta_phi_g = tf.matmul(theta_phi, g_rsh)
        theta_phi_g = Reshape((h, w, c))(theta_phi_g)

        theta_phi_g = Conv2D(c*2, kernel_size=(1,1), padding='same')(theta_phi_g)

        out = Add()([theta_phi_g, X])

        return out

    def gating_signal(input, out_size, batch_norm=False):
        """
        resize the down layer feature map into the same dimension as the up layer feature map
        using 1x1 conv
        :param input:   down-dim feature map
        :param out_size:output channel number
        :return: the gating feature map with the same dimension of the up layer feature map
        """
        x = keras.layers.Conv2D(out_size, (1, 1), padding='same')(input)
        if batch_norm:
            x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        return x

    def attention_block(x, gating, inter_shape):
        shape_x = K.int_shape(x)
        shape_g = K.int_shape(gating)

        theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
        shape_theta_x = K.int_shape(theta_x)

        phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
        upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                            strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                            padding='same')(phi_g)  # 16

        concat_xg = layers.add([upsample_g, theta_x])
        act_xg = layers.Activation('relu')(concat_xg)
        psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
        sigmoid_xg = layers.Activation('sigmoid')(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(
            sigmoid_xg)  # 32

        upsample_psi = expend_as(upsample_psi, shape_x[3])

        y = layers.multiply([upsample_psi, x])

        result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
        result_bn = layers.BatchNormalization()(result)
        return result_bn

    print('decode_filters=',decode_filters)
    non_local=global_non_local(base_model.output,decode_filters)

    # Decoder Layers
    decoder_1 = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                     name='conv2')(non_local)

    gating_2=gating_signal(decoder_1,int(decode_filters/2),False)
    att2=attention_block(base_model.get_layer('pool3_pool').output,gating_2,int(decode_filters/2))
    decoder_2 = upproject(decoder_1, int(decode_filters / 2), 'up1', concat_with=att2)

    gating_3 = gating_signal(decoder_2, int(decode_filters / 4), False)
    att3 = attention_block(base_model.get_layer('pool2_pool').output, gating_3, int(decode_filters / 4))
    decoder_3 = upproject(decoder_2, int(decode_filters / 4), 'up2', concat_with=att3)

    # conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3_')(decoder)

    gating_4 = gating_signal(decoder_3,int(decode_filters/8),False)
    att4 = attention_block(base_model.get_layer('pool1').output,gating_4,int(decode_filters/8))
    decoder_4 = upproject(decoder_3, int(decode_filters / 8), 'up3', concat_with=att4)

    # conv2 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv2_')(decoder)

    gating_5 = gating_signal(decoder_4,int(decode_filters/16),False)
    att5 = attention_block(base_model.get_layer('conv1/relu').output,gating_5,int(decode_filters/16))
    decoder_5 = upproject(decoder_4, int(decode_filters / 16), 'up4', concat_with=att5)
    # conv1 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv1_')(decoder)

    gating_6 = gating_signal(decoder_5,int(decode_filters/32),False)
    att6=attention_block(base_model.get_layer('input_1').output,gating_6,int(decode_filters/32))
    decoder_6 = upproject(decoder_5, int(decode_filters / 32), 'up5', concat_with=att6)

    conv0 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv0_')(decoder_6)

    predictions_raw_0 = conv0 * 1000
    # predictions_raw_1 = conv1 * 1000
    # predictions_raw_2 = conv2 * 1000
    # predictions_raw_3 = conv3 * 1000

    predictions_0 = tf.clip_by_value(predictions_raw_0, 1.0, 65535.0)
    # predictions_1 = tf.clip_by_value(predictions_raw_1, 1.0, 65535.0)
    # predictions_2 = tf.clip_by_value(predictions_raw_2, 1.0, 65535.0)
    # predictions_3 = tf.clip_by_value(predictions_raw_3, 1.0, 65535.0)

    final_outputs_0 = tf.cast(tf.clip_by_value(predictions_raw_0, 0, 65535.0), tf.int16)
    eval_outputs__0 = tf.clip_by_value(predictions_raw_0, 0.1, 65535.0)
    # final_outputs_1 = tf.cast(tf.clip_by_value(predictions_raw_1, 0, 65535.0), tf.int16)
    # final_outputs_2 = tf.cast(tf.clip_by_value(predictions_raw_2, 0, 65535.0), tf.int16)
    # final_outputs_3 = tf.cast(tf.clip_by_value(predictions_raw_3, 0, 65535.0), tf.int16)

    # Create the model
    model = Model(inputs=base_model.input, outputs=
                  [predictions_0, final_outputs_0, eval_outputs__0])
    model.summary()
    print('Model created.')

    return model
