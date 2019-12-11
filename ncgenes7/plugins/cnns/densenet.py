# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""Versions of CNN encoder and decoder using densenet architecture

References
----------
densenet paper
    https://arxiv.org/abs/1608.06993
"""
from typing import List
from typing import Optional
from typing import Union
import warnings

from nucleus7.utils.tf_varscopes_utils import with_name_scope
import tensorflow as tf

from ncgenes7.plugins.cnns import cnn_blocks
from ncgenes7.plugins.cnns.base import BaseCNNPlugin


class DensenetPlugin(BaseCNNPlugin):
    """
    Densenet architecture with different inception modules

    Parameters
    ----------
    num_dense_blocks_per_layer
        number of dense blocks in each building block; you do not need to
        provide num_blocks and cnn_blocks_parameters
    growth_rate
        see paper
    bottleneck_factor
        bottleneck factor; if its is <= 0, then absolute number will be used as
        number of filters in bottleneck layer, otherwise this number will be
        multiplied with growth_rate to have the number of feature maps in
        bottleneck layer
    transition_downsample_factor
        see paper
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
    use_inception_module
        if the inception module with multiple convolutions on same input
        following concatenating and 1x1 convolution should be used like
        x1=conv(x, kern1); x2=conv(x, kern2); x3=concat(x1, x2); in that case
        kernel_sizel should be a list of kernel sizes for different convolutions
        in inception block; default single convolution is used

    Attributes
    ----------
    incoming_keys
        * feature_maps : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32
        * auxiliary_feature_maps
          [batch_size, w_i, h_i, num_channels_i] to be used for skip connections

    generated_keys
        * feature_maps : feature maps after processing
        * auxiliary_feature_maps : (optional)  auxiliary feature maps for
          each block

    References
    ----------
    densenet paper
        https://arxiv.org/abs/1608.06993
    """
    incoming_keys = [
        "feature_maps",
        "_auxiliary_feature_maps",
    ]

    generated_keys = [
        "feature_maps",
        "_auxiliary_feature_maps",
    ]

    def __init__(self, *,
                 num_dense_blocks_per_layer: List[int],
                 growth_rate: int = 16,
                 bottleneck_factor: int = 4,
                 transition_downsample_factor: float = 0.5,
                 kernel_size: Union[List[int], int] = 3,
                 dilation_rate: Union[List[int], int] = 1,
                 batch_normalization_params: Optional[dict] = None,
                 use_inception_module: bool = False,
                 **plugin_kwargs):
        assert isinstance(num_dense_blocks_per_layer, (list, tuple)), (
            "num_dense_blocks_per_layer should be a list of int!!!")
        num_blocks = len(num_dense_blocks_per_layer)
        if "num_blocks" in plugin_kwargs:
            warnings.warn(
                "num_blocks for {} is inferred from "
                "num_dense_blocks_per_layer and is {}".format(
                    self.__class__.__name__, num_blocks))
        if "cnn_blocks_parameters" in plugin_kwargs:
            raise ValueError("cnn_blocks_parameters must be not se for "
                             "{}".format(self.__class__.__name__))
        cnn_blocks_parameters = [{'number_of_blocks': k}
                                 for k in num_dense_blocks_per_layer]
        super().__init__(num_blocks=num_blocks,
                         cnn_blocks_parameters=cnn_blocks_parameters,
                         **plugin_kwargs)
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        self.transition_downsample_factor = transition_downsample_factor
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.batch_normalization_params = batch_normalization_params
        self.use_inception_module = use_inception_module

    @with_name_scope("densenet_block")
    def build_cnn_block(self, inputs, number_of_blocks):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        layer_name = self.get_current_layer_full_name("cnn")
        inception_layer_fn = (
            cnn_blocks.InceptionConv2D if self.use_inception_module
            else tf.keras.layers.Conv2D)
        densenet_layer = self.add_keras_layer(
            cnn_blocks.DenseLayer(
                inception_layer_fn=inception_layer_fn,
                number_of_blocks=number_of_blocks,
                growth_rate=self.growth_rate,
                kernel_size=self.kernel_size,
                activation=self.activation,
                dropout_fn=self.dropout,
                bottleneck_factor=self.bottleneck_factor,
                dilation_rate=self.dilation_rate,
                batch_normalization_params=self.batch_normalization_params,
                kernel_initializer=self.initializer),
            name=layer_name)
        x = densenet_layer(inputs, training=self.is_training)
        # sampling is performed inside of the plugin.sample operation so here
        # is only a conv part of transitional layer is needed

        if self.batch_normalization_params:
            bn_layer = self.add_keras_layer(
                tf.keras.layers.BatchNormalization(
                    **self.batch_normalization_params),
                name=layer_name + "_transition_bn_layer"
            )
            x = bn_layer(x)

        x_number_of_filters = x.get_shape().as_list()[-1]
        number_of_filters_transition = int(
            x_number_of_filters * self.transition_downsample_factor)
        conv_1x1_layer = self.add_keras_layer(
            tf.keras.layers.Conv2D(number_of_filters_transition, 1,
                                   use_bias=False, name='transition'),
            name=layer_name + "_transition_conv_1x1")
        out = conv_1x1_layer(x)
        return out
