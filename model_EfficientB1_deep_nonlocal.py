import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate,Add,Reshape
from layers import BilinearUpSampling2D
import tensorflow as tf
import keras
#from efficientnet import EfficientNetB0 as Net

import efficientnet.tfkeras as efn

def create_model(is_twohundred=False, is_halffeatures=True):
    print('Loading base model (DenseNet)..')

    # Encoder Layers

    if is_twohundred:
        base_model = applications.DenseNet201(input_shape=(480, 640, 3), include_top=False)
    else:
        #base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)
        base_model = efn.EfficientNetB1(weights='imagenet',include_top=False,input_shape=((480, 640, 3)))

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
    base_model.summary()
    # Define upsampling layer
    def upproject(tensor, filters, name, concat_with):
        up_i = BilinearUpSampling2D((2, 2), name=name + '_upsampling2d')(tensor)
        up_i = Concatenate(name=name + '_concat')(
            [up_i, base_model.get_layer(concat_with).output])  # Skip connection
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

    non_local=global_non_local(base_model.output,decode_filters)

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                     name='conv2')(non_local)

    decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='block5d_add')
    decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='block3c_add')

    # conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='block2c_add')
    # conv2 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv2_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='block1b_add')
    # conv1 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv1_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')
    conv0 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv0_')(decoder)

    '''
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                     name='conv2')(base_model.output)
    decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='pool2_pool')
    decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='pool1')
    decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1/relu')
    decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')
    conv0 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv0_')(decoder)
    '''

    predictions_raw_0 = conv0 * 1000
    # predictions_raw_1 = conv1 * 1000
    # predictions_raw_2 = conv2 * 1000
    # predictions_raw_3 = conv3 * 1000

    predictions_0 = tf.clip_by_value(predictions_raw_0, 1.0, 65535.0)
    # predictions_1 = tf.clip_by_value(predictions_raw_1, 1.0, 65535.0)
    # predictions_2 = tf.clip_by_value(predictions_raw_2, 1.0, 65535.0)
    # predictions_3     = tf.clip_by_value(predictions_raw_3, 1.0, 65535.0)

    final_outputs_0 = tf.cast(tf.clip_by_value(predictions_raw_0, 0, 65535.0), tf.int16)
    eval_outputs__0 = tf.clip_by_value(predictions_raw_0, 0.1, 65535.0)
    # final_outputs_1 = tf.cast(tf.clip_by_value(predictions_raw_1, 0, 65535.0), tf.int16)
    # final_outputs_2 = tf.cast(tf.clip_by_value(predictions_raw_2, 0, 65535.0), tf.int16)
    # final_outputs_3 = tf.cast(tf.clip_by_value(predictions_raw_3, 0, 65535.0), tf.int16)

    # Create the model
    model = Model(inputs=base_model.input, outputs=
    [predictions_0, final_outputs_0, eval_outputs__0])



    model.summary()
    #print(base_model.output)
    #print(base_model.layers[-1].output)
    print('Model created.')

    return model
#a=create_model()