# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
various cnn blocks
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from ncgenes7.utils.keras_utils import KerasCompositionalLayer


class DenseLayer(KerasCompositionalLayer):
    """
    DenseNet layer consisting of multiple dense blocks

    Parameters
    ----------
    number_of_blocks
        number of dense blocks in layer
    growth_rate
        see paper
    bottleneck_factor
        bottleneck factor; if its is <= 0, then absolute number will be used as
        number of filters in bottleneck layer, otherwise this number will be
        multiplied with growth_rate to have the number of feature maps in
        bottleneck layer
    inception_layer_fn
        function to generate the layer; it should have the same signature as
        :obj:`tf.keras.layers.Conv2D`
    kernel_size
        kernel size for single convolution or list of sizes if
        use_inception_module == True
    dilation_rate
        if is int, then this value will be used in whole inception module;
        else single values will be used with corresponding kernel_size values;
        if it is a list and use_inception_module == True, then
        len(dilation_rate) == len(kernel_size)
    batch_normalization_params
        parameters for batch normalization to apply on the subblock
        inputs and on the transition block inputs;
        see :obj:`tf.keras.layers.BatchNormalization` for description
    dropout_fn
        dropout function, which takes only 2 arguments: input tensor and
        training flag; dropout will be added before every 2nd block in layer

    References
    ----------
    densenet paper
        https://arxiv.org/abs/1608.06993
    """

    def __init__(
            self,
            number_of_blocks: int,
            growth_rate: int = 6,
            kernel_size: Union[int, list] = 3,
            inception_layer_fn: Callable = tf.keras.layers.Conv2D,
            bottleneck_factor: Optional[int] = None,
            dropout_fn: Optional[Callable[[tf.Tensor, bool], tf.Tensor]] = None,
            dilation_rate: Union[list, int] = 1,
            batch_normalization_params: Optional[dict] = None,
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = tf.nn.relu,
            data_format=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs):
        # pylint: disable=too-many-arguments,too-many-locals
        super(DenseLayer, self).__init__(name=name, **kwargs)
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.number_of_blocks = number_of_blocks
        self._dense_blocks = []
        for i_block in range(number_of_blocks):
            block_dropout_fn = dropout_fn if i_block % 2 else None
            dense_block = self.add_keras_layer(
                DenseBlock(
                    inception_layer_fn=inception_layer_fn,
                    growth_rate=self.growth_rate,
                    kernel_size=kernel_size,
                    bottleneck_factor=self.bottleneck_factor,
                    dropout_fn=block_dropout_fn,
                    dilation_rate=dilation_rate,
                    batch_normalization_params=batch_normalization_params,
                    activation=activation,
                    data_format=data_format,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    **kwargs,
                ))
            self._dense_blocks.append(dense_block)

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        all_outputs = [inputs]

        for block_layer in self._dense_blocks:
            if len(all_outputs) == 1:
                block_inputs = all_outputs[-1]
            else:
                block_inputs = tf.keras.layers.Concatenate(-1)(
                    all_outputs)
            block_output = block_layer(block_inputs, training=training)
            all_outputs.append(block_output)
        out = tf.keras.layers.Concatenate(-1)(all_outputs)
        return out


class DenseBlock(KerasCompositionalLayer):
    """
    Single DenseNet block consisting of batch normalization, dropout function
    following inception module.

    Parameters
    ----------
    growth_rate
        see paper
    bottleneck_factor
        bottleneck factor; if its is <= 0, then absolute number will be used as
        number of filters in bottleneck layer, otherwise this number will be
        multiplied with growth_rate to have the number of feature maps in
        bottleneck layer
    inception_layer_fn
        function to generate the layer; it should have the same signature as
        :obj:`tf.keras.layers.Conv2D`
    kernel_size
        kernel size for single convolution or list of sizes if
        use_inception_module == True
    dilation_rate
        if is int, then this value will be used in whole inception module;
        else single values will be used with corresponding kernel_size values;
        if it is a list and use_inception_module == True, then
        len(dilation_rate) == len(kernel_size)
    batch_normalization_params
        parameters for batch normalization to apply on the subblock
        inputs and on the transition block inputs;
        see :obj:`tf.keras.layers.BatchNormalization` for description
    dropout_fn
        dropout function, which takes only 2 arguments: input tensor and
        training flag

    References
    ----------
    densenet paper
        https://arxiv.org/abs/1608.06993
    """

    def __init__(
            self,
            growth_rate: int,
            kernel_size: Union[int, list],
            inception_layer_fn: Callable = tf.keras.layers.Conv2D,
            bottleneck_factor: Optional[int] = None,
            dropout_fn: Optional[Callable[[tf.Tensor, bool], tf.Tensor]] = None,
            dilation_rate: Union[list, int] = 1,
            batch_normalization_params: Optional[dict] = None,
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            data_format=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs):
        # pylint: disable=too-many-arguments,too-many-locals
        super().__init__(name=name, **kwargs)
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.dropout_fn = dropout_fn

        if self.bottleneck_factor is not None:
            if self.bottleneck_factor < 0:
                # pylint: disable=invalid-unary-operand-type
                bottleneck_filters = -self.bottleneck_factor
            else:
                bottleneck_filters = self.growth_rate * self.bottleneck_factor
            self._bottleneck_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    bottleneck_filters, kernel_size=1, padding="same",
                    activation=activation,
                    data_format=data_format,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    **kwargs,
                ))
        else:
            self._bottleneck_layer = None
        if batch_normalization_params is not None:
            self._batch_normalization_layer = self.add_keras_layer(
                tf.keras.layers.BatchNormalization(
                    **batch_normalization_params))
        else:
            self._batch_normalization_layer = None
        self._inception_layer = self.add_keras_layer(
            inception_layer_fn(
                filters=growth_rate, kernel_size=kernel_size, padding="same",
                dilation_rate=dilation_rate,
                activation=activation,
                data_format=data_format,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
            ))

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        x = inputs
        if self._batch_normalization_layer:
            x = self._batch_normalization_layer(x)
        if self.dropout_fn:
            x = self.dropout_fn(x, training=training)
        x = self._inception_layer(x)
        return x


class DUC(KerasCompositionalLayer):
    """
    DUC module

    Parameters
    ----------
    kernel_size
        kernel size of DUC convolution
    num_classes
        number of classes
    stride_upsample
        see paper
    dilation_rates
        see paper

    References
    ----------
    DUC module
        https://arxiv.org/abs/1702.08502
    """

    def __init__(
            self,
            kernel_size: Union[int, list],
            num_classes: int,
            stride_upsample: int,
            dilation_rates: Union[Tuple[int], List[int]] = (2, 4, 8, 16),
            data_format=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs):
        # pylint: disable=too-many-arguments,too-many-locals
        super().__init__(name=name, **kwargs)
        self.stride_upsample = stride_upsample
        self.num_classes = num_classes
        self.dilation_rates = dilation_rates

        filters = self.num_classes * self.stride_upsample * self.stride_upsample
        self._duc_layers = [
            self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters, kernel_size=kernel_size,
                    dilation_rate=each_dilation_rate, padding="same",
                    activation=None, data_format=data_format, use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                ))
            for each_dilation_rate in self.dilation_rates]

    def call(self, inputs):  # pylint: disable=arguments-differ
        outputs_with_dilations = []
        for each_duc_layer in self._duc_layers:
            out = each_duc_layer(inputs)
            outputs_with_dilations.append(out)
        x = tf.add_n(outputs_with_dilations)
        x_shape = tf.shape(x)
        new_shape = tf.concat(
            [[-1], x_shape[1:3],
             [self.stride_upsample, self.stride_upsample, self.num_classes]],
            -1)
        x = tf.reshape(x, new_shape)
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        out_shape = tf.concat(
            [[-1], x_shape[1:3] * self.stride_upsample, [self.num_classes]],
            -1)
        x = tf.reshape(x, out_shape)
        return x


# pylint: disable=too-many-instance-attributes
class PSPNetLayer(KerasCompositionalLayer):
    """
    Pyramid pooling module

    Parameters
    ----------
    filters
        number of feature maps in each convolution after pooling and after
        concatenation of psp features
    output_height
        height of the upsampled output; if negative, then it is a multiplier to
        input size; otherwise it is a size itself
    output_width
        width of the upsampled output; if negative, then it is a multiplier to
        input size; otherwise it is a size itself
    pool_sizes
        pool sizes to use; if it is negative, then it is the same as in the
        paper - bin sizes; otherwise it is a pool sizes; in case of bin sizes,
        the input spatial dimensions must be defined and so cannot be None

    References
    ----------
    Pyramid Scene Parsing Network
        https:arxiv.org/abs/1612.01105
    """

    def __init__(
            self,
            filters,
            output_height: Union[float, int],
            output_width: Union[float, int],
            pool_sizes: Union[Tuple[int], List[int]] = (-1, -2, -3, -6),
            kernel_size: int = 3,
            num_classes: Optional[int] = None,
            activation: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
            data_format=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs):
        # pylint: disable=too-many-arguments,too-many-locals
        super().__init__(name=name, **kwargs)
        assert len(set(pool > 0 for pool in pool_sizes)) == 1, (
            "pool_sizes should have the same type: all >0 if you want to "
            "specify the absolute pool sizes or <0 if it is bin sizes!"
        )
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_height = output_height
        self.output_width = output_width
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes

        self._pyramid_conv_layers = []
        for _ in self.pool_sizes:
            conv_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters=self.filters, kernel_size=1,
                    data_format=data_format, activation=activation,
                    use_bias=use_bias, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                ))
            self._pyramid_conv_layers.append(conv_layer)
        self._last_conv_layer = self.add_keras_layer(
            tf.keras.layers.Conv2D(
                filters=self.filters, kernel_size=kernel_size,
                padding="same",
                data_format=data_format, activation=activation,
                use_bias=use_bias, kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
            ))
        if self.num_classes:
            self._logits_conv_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters=self.num_classes, kernel_size=1,
                    data_format=data_format,
                    use_bias=use_bias, kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                ))
        else:
            self._logits_conv_layer = None

    def call(self, inputs):  # pylint: disable=arguments-differ
        pool_sizes = self._get_pool_sizes(inputs)
        psp_outputs = []
        inputs_shape = tf.shape(inputs)
        for i, each_pool_size in enumerate(pool_sizes):
            pool_layer = self.add_keras_layer(
                tf.keras.layers.MaxPool2D(each_pool_size, name="pool_%s" % i))
            x = pool_layer(inputs)
            x = self._pyramid_conv_layers[i](x)
            x = tf.image.resize_bilinear(x, inputs_shape[1:3],
                                         align_corners=True)
            psp_outputs.append(x)

        x = tf.keras.layers.Concatenate(-1)(psp_outputs)
        x = self._last_conv_layer(x)
        if self._logits_conv_layer:
            x = self._logits_conv_layer(x)
        output_sizes = self._get_output_sizes(inputs)
        x = tf.image.resize_bilinear(x, output_sizes, align_corners=True)
        return x

    def _get_pool_sizes(self, inputs: tf.Tensor):
        if self.pool_sizes[0] > 0:
            return self.pool_sizes
        inputs_spatial_shape = inputs.shape.as_list()[1:3]
        pool_sizes = [[-spatial_shape // each_bin_size
                       for spatial_shape in inputs_spatial_shape]
                      for each_bin_size in self.pool_sizes]
        return pool_sizes

    def _get_output_sizes(self, inputs):
        input_shapes_static = inputs.shape.as_list()
        input_shapes_dynamic = tf.shape(inputs)

        if self.output_height < 0:
            input_height = input_shapes_static[1]
            if input_height is None:
                input_height = input_shapes_dynamic[1]
            output_height = input_height * (-self.output_height)
        else:
            output_height = self.output_height
        if self.output_width < 0:
            input_width = input_shapes_static[2]
            if input_width is None:
                input_width = input_shapes_dynamic[2]
            output_width = input_width * (-self.output_width)
        else:
            output_width = self.output_width
        return output_height, output_width


# pylint: enable=too-many-instance-attributes


class InceptionConv2D(KerasCompositionalLayer):
    """
    Inception which applies the convolution on inputs with different kernel
    sizes and same padding and then applies the convolution with kernel size
    of 1 on the concatenated filters from all other convolutions (if
    add_1x1_conv == True)

    Parameters
    ----------
    filters
        number of filters for each convolution
    kernel_size
        list of kernel sizes for different convolutions
    activation
        activation to use
    dilation_rates
        list of dilation rates for each of first convolutions
    add_1x1_conv
        if the 1x1 convolution should be added after concatenation
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 add_1x1_conv: bool = True,
                 name=None,
                 **kwargs):
        # pylint: disable=too-many-arguments,too-many-locals
        if "padding" in kwargs:
            kwargs.pop("padding")
        super().__init__(name=name, **kwargs)
        assert isinstance(kernel_size, (list, tuple)), (
            "kernel_size must be a list with values for each subblock!")
        if isinstance(dilation_rate, (list, tuple)):
            assert len(dilation_rate) == len(kernel_size), (
                "if you want to provide different dilation rates, the length"
                "of dilation_rate must be the same as number of kernel_size! "
                "received: kernel_size: {}, dilation_rates: {}".format(
                    kernel_size, dilation_rate))
        self.add_1x1_conv = add_1x1_conv

        self._inception_layers = []
        for kernel_i, each_kernel_size in enumerate(kernel_size):
            if not dilation_rate:
                each_dilation_rate = 1
            else:
                each_dilation_rate = (
                    dilation_rate if isinstance(dilation_rate, int)
                    else dilation_rate[kernel_i])
            inception_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters, kernel_size=each_kernel_size,
                    dilation_rate=each_dilation_rate,
                    padding="same",
                    activation=activation,
                    data_format=data_format,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    **kwargs,
                ))
            self._inception_layers.append(inception_layer)

        if self.add_1x1_conv:
            self._last_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters, kernel_size=1, padding="same",
                    activation=activation,
                    data_format=data_format,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint,
                    **kwargs,
                ))
        else:
            self._last_layer = None

    def call(self, inputs, **kwargs):  # pylint: disable=arguments-differ
        inception_outputs = []
        for each_inception_layer in self._inception_layers:
            inception_outputs.append(each_inception_layer(inputs))
        if len(inception_outputs) == 1:
            out = inception_outputs[0]
        else:
            out = tf.concat(inception_outputs, -1)
        if self._last_layer:
            out = self._last_layer(out)
        return out
