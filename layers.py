# -*- coding: utf-8 -*-

'''
This is a modification of the SeparableConv3D code in Keras,
to perform just the Depthwise Convolution (1st step) of the
Depthwise Separable Convolution layer.
'''
from __future__ import absolute_import

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras import layers
from tensorflow.keras.layers import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Conv1D, Conv3D

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_models as tfm


def _preprocess_conv3d_input(x, data_format):
    if K.dtype(x) == 'float64':
        x = tf.cast(x, 'float32')
    if data_format == 'channels_first':
        x = tf.transpose(x, (0, 4, 1, 2, 3))
    return x
     
def _preprocess_padding(padding):
    if padding == 'same':
        padding = 'SAME'
    elif padding == 'valid':
        padding = 'VALID'
    else:
        raise ValueError('Invalid padding:', padding)
    return padding

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = tf.stack((tf.sin(sin_inp), tf.cos(sin_inp)), -1)
    emb = tf.reshape(emb, (*emb.shape[:-2], -1))
    return emb

@tf.keras.utils.register_keras_serializable(package="Addons")
class CausalConv1D(Conv1D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name, #**kwargs
        )
       
    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)
    
class TemporalBlock(layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv2")
        self.down_sample = None
    
    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.layer_norma1 = tfa.layers.WeightNormalization(self.conv1)
        self.layer_norma2 = tfa.layers.WeightNormalization(self.conv2)
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1, 
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = layers.Dense(self.n_outputs, activation=None)
    
    def call(self, inputs, training=True):
        #x = self.conv1(inputs)
        x = self.layer_norma1(inputs)
        x = self.dropout1(x, training=training)
        #x = self.conv2(x)
        x = self.layer_norma2(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return tf.nn.relu(x + inputs)

class TemporalConvNet(layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2,
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=1, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs

@tf.keras.utils.register_keras_serializable(package="Addons")
class CausalConv3D(Conv3D):
    def __init__(self, filters,
               kernel_size,
               strides=1,
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(CausalConv3D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,  # Adjust dilation rate for 3D convolution
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            name=name,# **kwargs
        )
       
    def call(self, inputs):
        padding_dim = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (padding_dim, 0), (padding_dim, 0), (padding_dim, 0), (0, 0)]))
        return super(CausalConv3D, self).call(inputs)

class SpatialBlock3D(tf.keras.layers.Layer):
    def __init__(self, n_outputs, kernel_size, pool_size, strides, dilation_rate, dropout=0.2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(SpatialBlock3D, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )        
        self.dropout = dropout
        self.pool_size = pool_size
        self.n_outputs = n_outputs
        self.conv1 = CausalConv3D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv1")
        self.conv2 = CausalConv3D(
            n_outputs, kernel_size, strides=strides, 
            dilation_rate=dilation_rate, activation=tf.nn.relu, 
            name="conv2")
        self.down_sample = None
    
    def build(self, input_shape):
        channel_dim = 1
        self.dropout1 = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.layer_norma1 = tfa.layers.WeightNormalization(self.conv1)
        self.layer_norma2 = tfa.layers.WeightNormalization(self.conv2)
        self.layer_pooling = tf.keras.layers.MaxPooling3D(self.pool_size)
        if input_shape[channel_dim] != self.n_outputs:
            self.down_sample = Conv3D(
                self.n_outputs, kernel_size=(1, 1, 1), 
                activation=None, data_format="channels_last", padding="valid")
    
    def call(self, inputs, training=True):
        #x = self.conv1(inputs)
        x = self.layer_norma1(inputs)
        x = self.dropout1(x, training=training)
        #x = self.conv2(x)
        x = self.layer_norma2(x)
        x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
        return self.layer_pooling(tf.nn.relu(x + inputs))


class SCN3D(layers.Layer):
    def __init__(self, num_channels, kernel_size=2, dropout=0.2, pool_size=2, 
                 trainable=True, name=None, dtype=None, 
                 activity_regularizer=None, **kwargs):
        super(SCN3D, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                SpatialBlock3D(out_channels, kernel_size, strides=1, pool_size=2, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )
    
    def call(self, inputs, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)
        return outputs

class DepthwiseConv3D(Conv3D):
    #@legacy_depthwise_conv3d_support
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 groups=None,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate = (1, 1, 1),
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.groups = groups
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"
        self.input_dim = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[channel_axis])

        if self.groups is None:
            self.groups = self.input_dim

        if self.groups > self.input_dim:
            raise ValueError('The number of groups cannot exceed the number of channels')

        if self.input_dim % self.groups != 0:
            raise ValueError('Warning! The channels dimension is not divisible by the group size chosen')

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  self.input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs = _preprocess_conv3d_input(inputs, self.data_format)

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        if self._data_format == 'NCDHW':
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, i:i+self.input_dim//self.groups, :, :, :], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=1)

        else:
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, :, :, :, i:i+self.input_dim//self.groups], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=-1)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
            out_filters = self.groups * self.depth_multiplier
        elif self.data_format == 'channels_last':
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = self.groups * self.depth_multiplier

        depth = conv_utils.conv_output_length(depth, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])

        rows = conv_utils.conv_output_length(rows, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        cols = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, depth, rows, cols)

        elif self.data_format == 'channels_last':
            return (input_shape[0], depth, rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

DepthwiseConvolution3D = DepthwiseConv3D

class SpatialPyramidPooling3D(layers.Layer):
    def __init__(
            self,
            output_channels,
            dilation_rates,
            pool_kernel_size=None,
            use_sync_bn=False,
            batchnorm_momentum=0.99,
            batchnorm_epsilon=0.001,
            activation='relu',
            dropout=0.5,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=None,
            interpolation='trilinear',  # Change to trilinear for 3D
            use_depthwise_convolution=False,
            **kwargs):
        super(SpatialPyramidPooling3D, self).__init__(**kwargs)

        self.output_channels = output_channels
        self.dilation_rates = dilation_rates
        self.use_sync_bn = use_sync_bn
        self.batchnorm_momentum = batchnorm_momentum
        self.batchnorm_epsilon = batchnorm_epsilon
        self.activation = activation
        self.dropout = dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=5)  # Change to 5D
        self.pool_kernel_size = pool_kernel_size
        self.use_depthwise_convolution = use_depthwise_convolution

    def build(self, input_shape):
        channels = input_shape[4]  # Change to 4 for 3D

        self.aspp_layers = []
        bn_op = tf.keras.layers.BatchNormalization

        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        conv_sequential = tf.keras.Sequential([
            Conv3D(  # Change to Conv3D
                filters=self.output_channels,
                kernel_size=(1, 1, 1),  # Change to 3D kernel size
                kernel_initializer=tfm.utils.clone_initializer(
                    self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False),
            bn_op(
                axis=bn_axis,
                momentum=self.batchnorm_momentum,
                epsilon=self.batchnorm_epsilon,
                synchronized=self.use_sync_bn),
            layers.Activation(self.activation)
        ])
        self.aspp_layers.append(conv_sequential)

        for dilation_rate in self.dilation_rates:
            leading_layers = []
            kernel_size = (3, 3, 3)  # Change to 3D kernel size
            if self.use_depthwise_convolution:
                leading_layers += [
                    DepthwiseConv3D(  # Change to DepthwiseConv3D
                        depth_multiplier=1,
                        kernel_size=kernel_size,
                        padding='same',
                        dilation_rate=dilation_rate,
                        use_bias=False)
                ]
                kernel_size = (1, 1, 1)  # Change to 3D kernel size
            conv_sequential = tf.keras.Sequential(leading_layers + [
                Conv3D(  # Change to Conv3D
                    filters=self.output_channels,
                    kernel_size=kernel_size,
                    padding='same',
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_initializer=tfm.utils.clone_initializer(
                        self.kernel_initializer),
                    dilation_rate=dilation_rate,
                    use_bias=False),
                bn_op(
                    axis=bn_axis,
                    momentum=self.batchnorm_momentum,
                    epsilon=self.batchnorm_epsilon,
                    synchronized=self.use_sync_bn),
                layers.Activation(self.activation)
            ])
            self.aspp_layers.append(conv_sequential)

        if self.pool_kernel_size is None:
            pool_sequential = tf.keras.Sequential([
                layers.GlobalAveragePooling3D(),  # Change to GlobalAveragePooling3D
                layers.Reshape((1, 1, 1, channels))])  # Change to 1D for 3D
        else:
            pool_sequential = tf.keras.Sequential([
                layers.AveragePooling3D(self.pool_kernel_size)])  # Change to AveragePooling3D

        pool_sequential.add(
            tf.keras.Sequential([
                Conv3D(  # Change to Conv3D
                    filters=self.output_channels,
                    kernel_size=(1, 1, 1),  # Change to 3D kernel size
                    kernel_initializer=tfm.utils.clone_initializer(
                        self.kernel_initializer),
                    kernel_regularizer=self.kernel_regularizer,
                    use_bias=False),
                bn_op(
                    axis=bn_axis,
                    momentum=self.batchnorm_momentum,
                    epsilon=self.batchnorm_epsilon,
                    synchronized=self.use_sync_bn),
                layers.Activation(self.activation)
            ]))

        self.aspp_layers.append(pool_sequential)

        self.projection = tf.keras.Sequential([
            Conv3D(  # Change to Conv3D
                filters=self.output_channels,
                kernel_size=(1, 1, 1),  # Change to 3D kernel size
                kernel_initializer=tfm.utils.clone_initializer(
                    self.kernel_initializer),
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False),
            bn_op(
                axis=bn_axis,
                momentum=self.batchnorm_momentum,
                epsilon=self.batchnorm_epsilon,
                synchronized=self.use_sync_bn),
            layers.Activation(self.activation),
            layers.Dropout(rate=self.dropout)
        ])

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        result = []
        for i, layer in enumerate(self.aspp_layers):
            x = layer(inputs, training=training)
            # Apply resize layer to the end of the last set of layers.
            if i == len(self.aspp_layers) - 1:
                x = self.resize_3d(tf.cast(x, tf.float32), tf.shape(inputs)[1:4])
            result.append(tf.cast(x, inputs.dtype))
        result = tf.concat(result, axis=-1)
        result = self.projection(result, training=training)
        return result
    
    def resize_3d(self, x, size):
        input_shape = tf.shape(x)
        x = tf.transpose(x, perm=[0, 3, 1, 2, 4])
        new_shape = [-1, input_shape[1], input_shape[2], input_shape[4]]
        x = tf.reshape(x, new_shape)
        new_size = size[0:2]
        x = tf.image.resize(x, new_size)
        new_shape = [-1, input_shape[3], size[0], size[1], 1, input_shape[4]]
        x = tf.reshape(x, new_shape)
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4, 5])
        new_shape = [-1, input_shape[3], 1, input_shape[4]]
        x = tf.reshape(x, new_shape)
        new_size = (size[2], 1)
        x = tf.image.resize(x, new_size)
        new_shape = [-1, size[0], size[1], size[2], input_shape[4]]
        x = tf.reshape(x, new_shape)
        return x

    def get_config(self):
        config = {
            'output_channels': self.output_channels,
            'dilation_rates': self.dilation_rates,
            'pool_kernel_size': self.pool_kernel_size,
            'use_sync_bn': self.use_sync_bn,
            'batchnorm_momentum': self.batchnorm_momentum,
            'batchnorm_epsilon': self.batchnorm_epsilon,
            'activation': self.activation,
            'dropout': self.dropout,
            'kernel_initializer': tf.keras.initializers.serialize(
                self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(
                self.kernel_regularizer),
            'interpolation': self.interpolation,
        }
        base_config = super(SpatialPyramidPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BilinearAttentionPooling(layers.Layer):
    def __init__(self, filters):
        super(BilinearAttentionPooling, self).__init__()
        self.filters = filters

    def build(self, input_shape):
        _, _, _, _, channels = input_shape
        self.convolution = Conv3D(self.filters, (1, 1, 1), activation='sigmoid', padding='same')

    def call(self, inputs):
        # Generate Attention map
        attention_map = self.convolution(inputs)

        attended_feature_maps = []
        for k in range(self.filters):
            # Element-wise multiplication
            attended_feature_map = tf.math.multiply(inputs, attention_map[...,k:k+1])

            # Global Average Pooling
            g_k = tf.reduce_mean(attended_feature_map, axis=[1, 2, 3])

            attended_feature_maps.append(g_k)

        # Concatenate along axis 1
        concatenated_features = tf.concat(attended_feature_maps, axis=1)

        return concatenated_features, attention_map
    
    def get_config(self):
        config = {
            'filters': self.filters
        }
        base_config = super(BilinearAttentionPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class TFPositionalEncoding3D(layers.Layer):
    def __init__(self, channels: int, dtype=tf.float32):
        super(TFPositionalEncoding3D, self).__init__()

        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = np.float32(
            1
            / np.power(
                10000, np.arange(0, self.channels, 2) / np.float32(self.channels)
            )
        )
        self.cached_penc = None

    def call(self, inputs):
        if len(inputs.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == inputs.shape:
            return self.cached_penc

        self.cached_penc = None
        _, x, y, z, org_channels = inputs.shape

        dtype = self.inv_freq.dtype

        pos_x = tf.range(x, dtype=dtype)
        pos_y = tf.range(y, dtype=dtype)
        pos_z = tf.range(z, dtype=dtype)

        sin_inp_x = tf.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = tf.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = tf.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_x), 1), 1)
        emb_y = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_y), 1), 0)
        emb_z = tf.expand_dims(tf.expand_dims(get_emb(sin_inp_z), 0), 0)

        emb_x = tf.tile(emb_x, (1, y, z, 1))
        emb_y = tf.tile(emb_y, (x, 1, z, 1))
        emb_z = tf.tile(emb_z, (x, y, 1, 1))

        emb = tf.concat((emb_x, emb_y, emb_z), -1)
        self.cached_penc = tf.repeat(
            emb[None, :, :, :, :org_channels], tf.shape(inputs)[0], axis=0
        )
        return self.cached_penc
    
    def get_config(self):
        config = {
            'channels': self.channels,
        }
        base_config = super(TFPositionalEncoding3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
###############################################################################

class UnimodelS1(tf.keras.Model):
    def __init__(
            self,
            input_name,
            num_classes,
            timesteps,
            scn_channels,
            aspp_channels,
            tcn_channels,
            fc_units,
            dilation_rates=[6,12,18,24],
            kernel_size=3,
            pool_size=2,
            dropout=0.2,
            **kwargs):
        super(UnimodelS1, self).__init__(**kwargs)
        self.input_name    = input_name 
        self.num_classes   = num_classes
        self.timesteps     = timesteps
        self.scn_channels  = scn_channels
        self.aspp_channels = aspp_channels
        self.tcn_channels  = tcn_channels
        self.dilation_rates = dilation_rates
        self.fc_units    = fc_units
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.dropout     = dropout
        
        self.scn_layer = SCN3D(num_channels=self.scn_channels,
                               kernel_size=self.kernel_size,
                               pool_size=self.pool_size,
                               dropout=self.dropout)
        #self.pe_layer = tf.keras.layers.Embedding(input_dim=timesteps, output_dim=128)
        self.aspp_layer = SpatialPyramidPooling3D(self.aspp_channels,
                                                  self.dilation_rates,
                                                  dropout=self.dropout)
        self.tcn_layer = TemporalConvNet(num_channels=self.tcn_channels,
                                         kernel_size=self.kernel_size-1,
                                         dropout=self.dropout)
        self.flatten_layer = layers.Flatten()
        self.fc1 = layers.Dense(self.fc_units[0],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout1 = layers.Dropout(self.dropout)
        self.fc2 = layers.Dense(self.fc_units[1],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout2 = layers.Dropout(self.dropout)
        if self.num_classes==2:
            self.output_layer = layers.Dense(self.num_classes-1, activation='sigmoid')
        elif self.num_classes>2:
            self.output_layer = layers.Dense(self.num_classes, activation='softmax')        
    
    def call(self, inputs, training=False):
        x = inputs[self.input_name]
        #d = inputs['deltas']
        #pe = self.pe_layer(d)
        
        x = self.scn_layer(x)
        x = self.aspp_layer(x)
        
        x = tf.reshape(x, [-1, tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3], tf.shape(x)[4]])
        x = tf.transpose(x, perm=[0, 2, 1])
        
        #x = tf.concat([x, pe], axis=-1)
        
        x = self.tcn_layer(x)
        x = self.flatten_layer(x)
        x = self.fc1(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout1(x)
        x = self.fc2(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout2(x)
        out = self.output_layer(x)
        
        return out

class UnimodelS1_CNN(tf.keras.Model):
    def __init__(
            self,
            input_name,
            num_classes,
            num_filters,
            fc_units,
            kernel_size=3,
            pool_size=2,
            dropout=0.2,
            **kwargs):
        super(UnimodelS1_CNN, self).__init__(**kwargs)
        self.input_name  = input_name
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.fc_units    = fc_units
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.dropout     = dropout
        
        self.cnn_layers = []
        num_levels = len(num_filters)
        for i in range(num_levels):
            filters = num_filters[i]
            self.cnn_layers.append(Conv3D(filters,
                                          kernel_size=self.kernel_size,
                                          padding='same'))
            self.cnn_layers.append(layers.BatchNormalization())
            self.cnn_layers.append(layers.Activation('relu'))
            self.cnn_layers.append(layers.MaxPooling3D(pool_size=self.pool_size))
        self.global_pool = layers.GlobalAveragePooling3D()
        self.fc1 = layers.Dense(self.fc_units[0],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout1 = layers.Dropout(self.dropout)
        self.fc2 = layers.Dense(self.fc_units[1],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout2 = layers.Dropout(self.dropout)
        if int(self.num_classes) == 2:
            self.output_layer = layers.Dense(self.num_classes-1, activation='sigmoid')
        elif int(self.num_classes) > 2:
            self.output_layer = layers.Dense(self.num_classes, activation='softmax')  
    
    def call(self, inputs, training=False):
        x = inputs[self.input_name]
        for layer in self.cnn_layers:
            x = layer(x)
        x = self.global_pool(x)
        x = self.fc1(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout1(x)
        x = self.fc2(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout2(x)
        out = self.output_layer(x)
        
        return out

class UnimodelS1_CNN_Attention(tf.keras.Model):
    def __init__(
            self,
            input_name,
            num_classes,
            num_filters,
            bap_filters,
            fc_units,
            kernel_size=3,
            pool_size=2,
            dropout=0.2,
            **kwargs):
        super(UnimodelS1_CNN_Attention, self).__init__(**kwargs)
        self.input_name  = input_name
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.bap_filters = bap_filters
        self.fc_units    = fc_units
        self.kernel_size = kernel_size
        self.pool_size   = pool_size
        self.dropout     = dropout
        
        self.cnn_layers = []
        num_levels = len(num_filters)
        for i in range(num_levels-1):
            filters = num_filters[i]
            self.cnn_layers.append(Conv3D(filters,
                                          kernel_size=self.kernel_size,
                                          padding='same'))
            self.cnn_layers.append(layers.BatchNormalization())
            self.cnn_layers.append(layers.Activation('relu'))
            self.cnn_layers.append(layers.MaxPooling3D(pool_size=self.pool_size))
        self.last_cnn_layer = Conv3D(num_filters[-1],
                                     kernel_size=self.kernel_size,
                                     padding='same')
        self.bap_pool = BilinearAttentionPooling(self.bap_filters)
        self.fc1 = layers.Dense(self.fc_units[0],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout1 = layers.Dropout(self.dropout)
        self.fc2 = layers.Dense(self.fc_units[1],
                                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-4, l2=1e-4),
                                activation=tf.nn.gelu)
        self.dropout2 = layers.Dropout(self.dropout)
        if int(self.num_classes) == 2:
            self.output_layer = layers.Dense(self.num_classes-1, activation='sigmoid')
        elif int(self.num_classes) > 2:
            self.output_layer = layers.Dense(self.num_classes, activation='softmax')  
    
    def call(self, inputs, training=False):
        x = inputs#[self.input_name]
        for layer in self.cnn_layers:
            x = layer(x)
        x = self.last_cnn_layer(x)
        x, att_map = self.bap_pool(x)
        x = self.fc1(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout1(x)
        x = self.fc2(x)
        if (self.dropout > 0.) and (training==True):
            x = self.dropout2(x)
        out = self.output_layer(x)
        
        return out, att_map
    
    