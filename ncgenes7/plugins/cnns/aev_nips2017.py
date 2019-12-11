# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
NIPS2017 AEV-21 plugin for cnn encoder and decoder

@authors: Oleksandr Vorobiov, Rupesh Durgesh

The AEV-21 in-house designed architecture for Nips-2017 demo.
"""

from nucleus7.utils.tf_varscopes_utils import with_name_scope
import tensorflow as tf

from ncgenes7.plugins.cnns import cnn_blocks
from ncgenes7.plugins.cnns.base import BaseCNNPlugin


class AEVNips2017Plugin(BaseCNNPlugin):
    """
    NIPS2017 AEV-21 architecture

    The architecture uses inception, dilation, residual blocks.

    Parameters
    ----------
    filters
        list of filters number for each block
    inception_kernels
        kernel size for inception blocks
    dilation_kernel
        dilation kernel size
    dilation_rates
        dilation rates for dilation convolution block

    Attributes
    ----------
    incoming_keys
        * feature_maps : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32
        * auxiliary_feature_maps : (optional), list of tensors of shape
          [batch_size, w_i, h_i, num_channels_i] to be used for skip
          connections

    generated_keys
        * feature_maps : feature maps after processing
        * auxiliary_feature_maps : (optional)  auxiliary feature maps for
          each block


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
                 filters,
                 inception_kernel_size=None,
                 dilated_inception_kernel_size=None,
                 dilated_inception_dilation_rate=None,
                 **plugin_kwargs):
        num_blocks = len(filters)

        self.filters = filters
        if not isinstance(inception_kernel_size, (tuple, list)):
            inception_kernel_size = [inception_kernel_size]
        if not isinstance(dilated_inception_kernel_size, (tuple, list)):
            dilated_inception_kernel_size = [dilated_inception_kernel_size]
        self.inception_kernel_sizes = [inception_kernel_size] * num_blocks
        self.dilated_inception_kernel_size = (
            [dilated_inception_kernel_size] * num_blocks)
        self.dilated_inception_dilation_rate = (
            [dilated_inception_dilation_rate] * num_blocks)
        cnn_blocks_parameters = [
            {"filters": block_f,
             "kernel_size": block_ks or [3],
             "dilated_kernel_size": block_dks or [1],
             "dilation_rate": block_dr or 1}
            for (block_f, block_ks, block_dks, block_dr) in zip(
                self.filters, self.inception_kernel_sizes,
                self.dilated_inception_kernel_size,
                self.dilated_inception_dilation_rate)]

        super(AEVNips2017Plugin, self).__init__(
            num_blocks=num_blocks, cnn_blocks_parameters=cnn_blocks_parameters,
            **plugin_kwargs)

    @with_name_scope("cnn_block")
    def build_cnn_block(self, inputs: tf.Tensor,
                        filters, kernel_size, dilated_kernel_size,
                        dilation_rate) -> tf.Tensor:
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        layer_name = self.get_current_layer_full_name("cnn")
        inception1 = self.add_keras_layer(
            cnn_blocks.InceptionConv2D(filters, kernel_size,
                                       activation=self.activation,
                                       kernel_initializer=self.initializer,
                                       name=layer_name + "_inc01"))
        dilated_inception = self.add_keras_layer(
            cnn_blocks.InceptionConv2D(filters, dilated_kernel_size,
                                       dilation_rate=dilation_rate,
                                       activation=self.activation,
                                       kernel_initializer=self.initializer,
                                       name=layer_name + "dilated_inc01"))
        x = inputs
        x = inception1(x)
        if self.sampling_type == "encoder":
            if self.dropout is not None:
                x = self.dropout(x, training=self.is_training)
        x = dilated_inception(x)
        if self.sampling_type == "encoder":
            inception2 = self.add_keras_layer(
                cnn_blocks.InceptionConv2D(filters, kernel_size,
                                           activation=self.activation,
                                           kernel_initializer=self.initializer,
                                           name=layer_name + "_inc02"))
            x = inception2(x)
        return x
