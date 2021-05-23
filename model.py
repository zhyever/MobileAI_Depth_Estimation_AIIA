import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate
from layers import BilinearUpSampling2D
import tensorflow as tf
import keras


from keras.layers import Activation, Reshape, Lambda, dot, add
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K


def non_local_block(ip, intermediate_dim=None, compression=2,
                    mode='embedded', add_residual=True):
    """
    Adds a Non-Local block for self attention to the input tensor.
    Input tensor can be or rank 3 (temporal), 4 (spatial) or 5 (spatio-temporal).
    Arguments:
        ip: input tensor
        intermediate_dim: The dimension of the intermediate representation. Can be
            `None` or a positive integer greater than 0. If `None`, computes the
            intermediate dimension as half of the input channel dimension.
        compression: None or positive integer. Compresses the intermediate
            representation during the dot products to reduce memory consumption.
            Default is set to 2, which states halve the time/space/spatio-time
            dimension for the intermediate step. Set to 1 to prevent computation
            compression. None or 1 causes no reduction.
        mode: Mode of operation. Can be one of `embedded`, `gaussian`, `dot` or
            `concatenate`.
        add_residual: Boolean value to decide if the residual connection should be
            added or not. Default is True for ResNets, and False for Self Attention.
    Returns:
        a tensor of same shape as input
    """
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    ip_shape = K.int_shape(ip)

    if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
        raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

    if compression is None:
        compression = 1

    dim1, dim2, dim3 = None, None, None

    # check rank and calculate the input shape
    if len(ip_shape) == 3:  # temporal / time series data
        rank = 3
        batchsize, dim1, channels = ip_shape

    elif len(ip_shape) == 4:  # spatial / image data
        rank = 4

        if channel_dim == 1:
            batchsize, channels, dim1, dim2 = ip_shape
        else:
            batchsize, dim1, dim2, channels = ip_shape

    elif len(ip_shape) == 5:  # spatio-temporal / Video or Voxel data
        rank = 5

        if channel_dim == 1:
            batchsize, channels, dim1, dim2, dim3 = ip_shape
        else:
            batchsize, dim1, dim2, dim3, channels = ip_shape

    else:
        raise ValueError('Input dimension has to be either 3 (temporal), 4 (spatial) or 5 (spatio-temporal)')

    # verify correct intermediate dimension specified
    if intermediate_dim is None:
        intermediate_dim = channels // 2

        if intermediate_dim < 1:
            intermediate_dim = 1

    else:
        intermediate_dim = int(intermediate_dim)

        if intermediate_dim < 1:
            raise ValueError('`intermediate_dim` must be either `None` or positive integer greater than 1.')

    if mode == 'gaussian':  # Gaussian instantiation
        x1 = Reshape((-1, channels))(ip)  # xi
        x2 = Reshape((-1, channels))(ip)  # xj
        f = dot([x1, x2], axes=2)
        f = Activation('softmax')(f)

    elif mode == 'dot':  # Dot instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        f = dot([theta, phi], axes=2)

        size = K.int_shape(f)

        # scale the values to make it size invariant
        f = Lambda(lambda z: (1. / float(size[-1])) * z)(f)

    elif mode == 'concatenate':  # Concatenation instantiation
        raise NotImplementedError('Concatenate model has not been implemented yet')

    else:  # Embedded Gaussian instantiation
        # theta path
        theta = _convND(ip, rank, intermediate_dim)
        theta = Reshape((-1, intermediate_dim))(theta)

        # phi path
        phi = _convND(ip, rank, intermediate_dim)
        phi = Reshape((-1, intermediate_dim))(phi)

        if compression > 1:
            # shielded computation
            phi = MaxPool1D(compression)(phi)

        f = dot([theta, phi], axes=2)
        f = Activation('softmax')(f)

    # g path
    g = _convND(ip, rank, intermediate_dim)
    g = Reshape((-1, intermediate_dim))(g)

    if compression > 1 and mode == 'embedded':
        # shielded computation
        g = MaxPool1D(compression)(g)

    # compute output path
    y = dot([f, g], axes=[2, 1])

    # reshape to input tensor format
    if rank == 3:
        y = Reshape((dim1, intermediate_dim))(y)
    elif rank == 4:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2))(y)
    else:
        if channel_dim == -1:
            y = Reshape((dim1, dim2, dim3, intermediate_dim))(y)
        else:
            y = Reshape((intermediate_dim, dim1, dim2, dim3))(y)

    # project filters
    y = _convND(y, rank, channels)

    # residual connection
    if add_residual:
        y = add([ip, y])

    return y


def _convND(ip, rank, channels):
    assert rank in [3, 4, 5], "Rank of input must be 3, 4 or 5"

    if rank == 3:
        x = Conv1D(channels, 1, padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    elif rank == 4:
        x = Conv2D(channels, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    else:
        x = Conv3D(channels, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(ip)
    return x


def create_model(args, is_twohundred=False, is_halffeatures=True):
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

    # Decoder Layers
    decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape,
                     name='conv2')(base_model.output)

    decoder = upproject(decoder, int(decode_filters / 2), 'up1', concat_with='pool3_pool')
    decoder = upproject(decoder, int(decode_filters / 4), 'up2', concat_with='pool2_pool')

    # conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 8), 'up3', concat_with='pool1')
    # conv2 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv2_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 16), 'up4', concat_with='conv1/relu')
    # conv1 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv1_')(decoder)

    decoder = upproject(decoder, int(decode_filters / 32), 'up5', concat_with='input_1')
    conv0 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv0_')(decoder)

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


def create_mobilev2_model():
    print("gg")