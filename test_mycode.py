import tensorflow as tf

features = tf.zeros((5, 50, 50, 1))
decoder_mu = tf.zeros((5, 50, 50, 1))

residual = tf.reshape(features - decoder_mu, [50] * 2)

print(residual.shape)