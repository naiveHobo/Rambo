import tensorflow as tf


def weight_variable(shape, init, stddev=0.1, mean=0.0, const_init=0.0):
    if init == 'xavier':
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        initial = initializer(shape=shape)
    elif init == 'constant':
        initial = tf.constant(const_init, shape=shape)
    else:
        initial = tf.random_normal(shape=shape, stddev=stddev, mean=mean)
    return tf.Variable(initial_value=initial, name="weights")


def bias_variable(shape, init):
    initial = tf.constant(init, shape=shape)
    return tf.Variable(initial, name="bias")


def input_layer(name, shape, dtype=tf.float32):
    return tf.placeholder(dtype=dtype, shape=shape, name=name)


def conv2d(name, input, filters, filter_size=(2, 2), stride=(2, 2), padding='SAME', kernel_init='random_normal',
           mean=0.0, stddev=0.1, const_init=0.0, bias_init=0.0):
    with tf.variable_scope(name):
        shape = [filter_size[0], filter_size[1], input.get_shape()[-1].value, filters]
        W = weight_variable(shape, kernel_init, mean=mean, stddev=stddev, const_init=const_init)
        b = bias_variable([filters], bias_init)
        xW = tf.nn.conv2d(input, W, strides=[1, stride[0], stride[1], 1], padding=padding)
    return tf.nn.bias_add(xW, b, name=name)


def deconv2d(name, input, out_shape, filter_size=(2, 2), stride=(2, 2), padding='SAME', kernel_init='random_normal',
             mean=0.0, stddev=0.1, const_init=0.0):
    with tf.variable_scope(name):
        shape = [filter_size[0], filter_size[1], out_shape[-1], input.get_shape()[-1].value]
        W = weight_variable(shape, kernel_init, mean=mean, stddev=stddev, const_init=const_init)
    return tf.nn.conv2d_transpose(input, W, output_shape=out_shape, strides=[1, stride[0], stride[1], 1],
                                  padding=padding, name=name)


def fully_connected(name, input, num_output, weight_init='random_normal', bias_init=0.0, mean=0.0, stddev=0.1):
    with tf.variable_scope(name):
        W = weight_variable([input.get_shape()[-1].value, num_output], weight_init, mean=mean, stddev=stddev)
        b = bias_variable([num_output], bias_init)
        xW = tf.matmul(input, W)
    return tf.nn.bias_add(xW, b, name=name)


def activation(name, input, function='relu'):
    if function == 'relu':
        return tf.nn.relu(input, name=name)


def flatten(name, input):
    num = 1
    shape = input.get_shape().as_list()
    for i in shape[1:]:
        num *= i
    return tf.reshape(input, [-1, num], name=name)


def dropout(name, input, keep_prob=1.0):
    return tf.nn.dropout(input, keep_prob=keep_prob, name=name)


def batch_norm(name, input, is_training):
    return tf.contrib.layers.batch_norm(input, is_training=is_training, trainable=True, name=name)
