# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Base plugin for cnn encoder and decoder
"""

import functools
import logging
from typing import List
from typing import Optional
from typing import Union
import warnings

import nucleus7 as nc7
from nucleus7.utils.tf_varscopes_utils import with_name_scope
import tensorflow as tf


# pylint: disable=too-many-instance-attributes
# attributes cannot be combined or extracted further
class BaseCNNPlugin(nc7.model.ModelPlugin):
    """Base class for cnn plugin encoder and decoder implementations

    Base architecture can be written as follows:

    `[first_conv]->[first_max_pool]->[block1]->[sample]->[block2]->
    [sample]->...->[last_conv]`

    Type of sampling is defined by `sampling_type`. In case of
    sampling_type=="encoder", it is downsampling, in case of "decoder" -
    upsampling. In this implementation, it is light functions like
    :obj:`tf.keras.layers.MaxPool2D` and :obj:`tf.keras.layers.Conv2DTranspose`.
    To change this behaviour, override method :func:`BaseCNNPlugin.sample`.

    `auxiliary_feature_maps` are iterated in reverse order. Also they should
    be produced using same padding

    Parameters
    ----------
    sampling_type
        type of the sampling
    first_conv_params
        parameters for first convolution; see :obj:`tf.keras.layers.Conv2D`;
        if not specified, no convolution will be performed
    first_sampling_params
        parameters for sampling after first convolution as defined in
        sample method; see corresponding functions;
        if not specified, no sampling will be performed
    last_conv_params
        parameters for convolution after blocks; :obj:`tf.keras.layers.Conv2D`;
        if not specified, no convolution will be performed
    last_conv_without_activation
        if last convolution should not have an activation
    sampling_params
        parameters for sampling like kernel_size and strides
    num_blocks
        number of blocks to create
    cnn_blocks_parameters
        list of dicts to pass to building block method; if defined, and not of
        length of num_blocks, last kwargs will be used for last remaining
        blocks; if defined as one dict, it will be used for all blocks
    add_skip_connections
        if the skip connections should be generated
        (only if auxiliary_blocks is inside of incoming_keys)
    skip_connection_type
        how to combine feature map with auxiliary one to build the skip
        connection; one of ['sum', 'concat']
    skip_connection_resize_type
        type of resizing performed on auxiliary feature map before connecting
        it to original maps; if None, then original maps will be sliced to
        auxiliary feature map size; it allows to preserve the original sizes
        (from encoder); one of [None, 'pad', 'imresize']
    auxiliary_feature_maps_start_ind
        index of first auxiliary_feature_map to use in skip connections;
        indexes are in normal order, e.g. 0 is first item from
        auxiliary_feature_maps list in order it was passed
    last_block_with_sampling
        if sampling operation should be applied after last block and before
        last convolution
    add_block_residual_connections
        flag to add residual connections in format
        `x -> block -> sample -> x1 -> res(x, x1)`;
        residual connection is the conv or deconv layer with same parameters
        as the sampling layer and with same number of feature maps as block
        output
    block_residual_connection_type
        type of residual connection if `add_block_residual_connections=True`;
        one of ['sum', 'concat']

    Attributes
    ----------
    block_var_scope_name
        suffix of block variable scope; complete name will be
        block_var_scope_name_{block_number}
    incoming_keys
        * feature_maps : tensor of shape  [batch_size, w, h, num_channels],
          tf.float32
        * auxiliary_feature_maps : (optional), list of tensors of shape
          [batch_size, w_i, h_i, num_channels_i] to be used for skip
          connections

    generated_keys
        * feature_maps : feature maps after processing
        * auxiliary_feature_maps : auxiliary feature maps for each block
    """

    incoming_keys = [
        "feature_maps",
        "auxiliary_feature_maps",
    ]
    generated_keys = [
        "feature_maps",
        "auxiliary_feature_maps",
    ]

    def __init__(self, *,
                 sampling_type: str = 'encoder',
                 num_blocks: int = 2,
                 first_conv_params: Optional[dict] = None,
                 first_sampling_params: Optional[dict] = None,
                 last_conv_params: Optional[dict] = None,
                 last_conv_without_activation: bool = False,
                 last_block_with_sampling: bool = True,
                 cnn_blocks_parameters: Union[List[dict], dict],
                 sampling_params: Optional[dict] = None,
                 add_skip_connections: bool = False,
                 skip_connection_type: str = 'concat',
                 skip_connection_resize_type: Optional[str] = None,
                 auxiliary_feature_maps_start_ind: int = 0,
                 add_block_residual_connections: bool = False,
                 block_residual_connection_type: str = 'sum',
                 **plugin_kwargs):
        # pylint: disable=too-many-locals
        # all of them are constructor parameters
        super().__init__(**plugin_kwargs)
        self.sampling_type = sampling_type
        self.first_conv_params = first_conv_params
        self.first_sampling_params = first_sampling_params
        self.last_conv_params = last_conv_params
        self.num_blocks = num_blocks
        self.cnn_blocks_parameters = cnn_blocks_parameters
        self.sampling_params = sampling_params
        self.add_skip_connections = add_skip_connections
        self.skip_connection_type = skip_connection_type
        self.skip_connection_resize_type = skip_connection_resize_type
        self.auxiliary_feature_maps_start_ind = (
            auxiliary_feature_maps_start_ind)
        self.last_conv_without_activation = last_conv_without_activation
        self.last_block_with_sampling = last_block_with_sampling
        self.add_block_residual_connections = add_block_residual_connections
        self.block_residual_connection_type = block_residual_connection_type
        assert self.skip_connection_type in ['sum', 'concat'], (
            "skip_connection_type should be in ['concat', 'sum']")
        assert self.skip_connection_resize_type in [None, 'pad', 'imresize'], (
            "skip_connection_type should be in [None, 'pad', 'imresize']")
        assert self.block_residual_connection_type in ['sum', 'concat'], (
            "block_residual_connection_type should be in ['sum', 'concat']")
        len_cnn_blocks_parameters = (
            len(self.cnn_blocks_parameters)
            if isinstance(self.cnn_blocks_parameters, list) else 1)
        if len_cnn_blocks_parameters > self.num_blocks:
            warnings.warn(
                "Last {} cnn_blocks_parameters will not be used!".format(
                    len_cnn_blocks_parameters - self.num_blocks))
        self._current_block_index = -1

    @property
    def defaults(self):
        defaults = super().defaults
        defaults["sampling_params"] = {
            "kernel_size": 2, "strides": 2, "padding": "same"}
        return defaults

    def get_current_layer_full_name(self, layer_name):
        """
        Get current full layer name with respect to current block_index
        like block_{block_index+1}_{layer_name}_layer

        Parameters
        ----------
        layer_name
            layer name without index

        Returns
        -------
        layer_full_name
            full name of layer
        """
        layer_full_name = "_".join(
            ["block", str(self._current_block_index + 1), layer_name, "layer"])
        return layer_full_name

    @with_name_scope("cnn_block")
    def build_cnn_block(self, inputs: tf.Tensor,
                        **block_parameters) -> tf.Tensor:
        """
        Build one cnn block

        Parameters
        ----------
        inputs
            block inputs
        block_parameters
            further parameters to build the block

        Returns
        -------
        result
            result
        """
        layer_name = self.get_current_layer_full_name("cnn")
        layer = self.add_keras_layer(
            tf.keras.layers.Conv2D(activation=self.activation,
                                   bias_initializer=self.initializer,
                                   kernel_initializer=self.initializer,
                                   name=layer_name,
                                   **block_parameters))
        out = layer(inputs)
        return out

    @with_name_scope("downsample")
    def downsample(self, inputs: tf.Tensor, *,
                   kernel_size: Union[list, int], strides: Union[list, int],
                   name: Optional[str] = None,
                   **additional_params) -> tf.Tensor:
        """
        Downsample inputs in case of encoder

        Parameters
        ----------
        inputs
            input tensor
        kernel_size
            kernel size for downsampling
        strides
            strides for downsampling
        name
            name of the operation

        Returns
        -------
        downsampled_tensor
            downsampled result
        """
        layer_name = name or self.get_current_layer_full_name("downsample")
        downsampled_layer = self.add_keras_layer(
            tf.keras.layers.MaxPool2D(pool_size=kernel_size, strides=strides,
                                      name=layer_name, **additional_params))

        out = downsampled_layer(inputs)
        return out

    @with_name_scope("upsample")
    def upsample(self, inputs: tf.Tensor, *,
                 kernel_size: Union[list, int], strides: Union[list, int],
                 name: Optional[str] = None, **additional_params) -> tf.Tensor:
        """
        Upsample inputs in case of decoder

        Parameters
        ----------
        inputs
            input tensor
        kernel_size
            kernel size for upsampling
        strides
            strides for upsampling
        name
            name of the operation

        Returns
        -------
        upsampled_tensor
            upsampled result
        """
        layer_name = name or self.get_current_layer_full_name("upsample")
        num_filters = inputs.get_shape().as_list()[-1]
        upsample_layer = self.add_keras_layer(
            tf.keras.layers.Conv2DTranspose(
                num_filters, kernel_size=kernel_size,
                strides=strides,
                bias_initializer=self.initializer,
                kernel_initializer=self.initializer,
                activation=self.activation, name=layer_name,
                **additional_params))
        out = upsample_layer(inputs)
        return out

    def sample(self, inputs: tf.Tensor, sampling_params: dict,
               name: Optional[str] = None) -> tf.Tensor:
        """
        Sample the inputs

        Parameters
        ----------
        inputs
            inputs to sample
        sampling_params
            parameters to pass to corresponding sample method (e.g. upsample or
            downsample)
        name
            name of the sample operation

        Returns
        -------
        sampled_result
            sampled result
        """
        sample_fun = {'encoder': self.downsample,
                      'decoder': self.upsample}[self.sampling_type]
        return sample_fun(inputs, **sampling_params, name=name)

    def create_skip_connection_maps(self, auxiliary_feature_map: tf.Tensor,
                                    input_feature_map: tf.Tensor) -> tf.Tensor:
        """
        Apply convolutions to auxiliary_feature_map before combining it with
        inputs

        Parameters
        ----------
        auxiliary_feature_map
            auxiliary feature maps for skip connections
        input_feature_map
            input feature maps for skip connections
        Returns
        -------
        skip_connection_maps
            skip connection maps

        """
        layer_name = self.get_current_layer_full_name("skip_connection")
        num_channels_inputs = input_feature_map.get_shape().as_list()[-1]
        skip_connection_layer = self.add_keras_layer(
            tf.keras.layers.Conv2D(num_channels_inputs, kernel_size=3,
                                   padding='same',
                                   activation=self.activation,
                                   bias_initializer=self.initializer,
                                   kernel_initializer=self.initializer,
                                   name=layer_name))

        skip_connection_maps = skip_connection_layer(auxiliary_feature_map)
        return skip_connection_maps

    @with_name_scope("skip_connection")
    def build_skip_connection(self, inputs: tf.Tensor,
                              auxiliary_feature_map: tf.Tensor) -> tf.Tensor:
        """
        First fit input size to to auxiliary_feature_map size and then
        apply skip function on them, e.g. concatenate or sum

        Parameters
        ----------
        auxiliary_feature_map
            auxiliary feature maps for skip connections
        inputs
            inputs
        layer_name_suffix
            suffix to layer name

        Returns
        -------
        feature_maps_after_skip_connection
            feature maps after skip connection
        """
        x = inputs
        if self.skip_connection_resize_type == "pad":
            auxiliary_feature_map = _pad_image_to_other(
                auxiliary_feature_map, x)
        elif self.skip_connection_resize_type == "imresize":
            new_shape = tf.shape(x)[1:3]
            auxiliary_feature_map = tf.image.resize_bilinear(
                auxiliary_feature_map, new_shape)
        else:
            x = _slice_image_to_other(auxiliary_feature_map, x)
        skip_connection_maps = self.create_skip_connection_maps(
            auxiliary_feature_map, x)
        skip_fun = {'concat': functools.partial(tf.concat, axis=-1),
                    'sum': tf.add_n}[self.skip_connection_type]
        x = skip_fun([x, skip_connection_maps])
        return x

    @with_name_scope("residual_connection")
    def build_residual_block_connection(self, block_input: tf.Tensor,
                                        block_output: tf.Tensor) -> tf.Tensor:
        """
        Create residual connection by applying convolution on block_input
        and add or concatenate it with block_output

        Applies convolution with kernel and stride from sampling_params

        Parameters
        ----------
        block_input
            feature maps before convolution block
        block_output
            feature maps after convolution block

        Returns
        -------
        block_res
            feature maps after residual connection
        """
        filters = block_output.get_shape().as_list()[-1]
        layer_name = self.get_current_layer_full_name("residual")
        if self.sampling_type == 'encoder':
            residual_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(
                    filters=filters, activation=self.activation,
                    bias_initializer=self.initializer,
                    kernel_initializer=self.initializer,
                    name=layer_name, **self.sampling_params))
        else:
            residual_layer = self.add_keras_layer(
                tf.keras.layers.Conv2DTranspose(
                    filters=filters, activation=self.activation,
                    bias_initializer=self.initializer,
                    kernel_initializer=self.initializer,
                    name=layer_name, **self.sampling_params))

        res_connection = residual_layer(block_input)
        if self.block_residual_connection_type == 'sum':
            out = tf.add_n([block_output, res_connection])
        else:
            out = tf.concat([block_output, res_connection], -1)
        return out

    @with_name_scope("first_layer")
    def build_first_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build first convolution layer following max pooling before blocks

        Parameters
        ----------
        inputs
            inputs

        Returns
        -------
        output
            result of the first convolution
        """
        out = inputs
        if self.first_conv_params is not None:
            conv_layer = self.add_keras_layer(
                tf.keras.layers.Conv2D(activation=self.activation,
                                       **self.first_conv_params,
                                       bias_initializer=self.initializer,
                                       kernel_initializer=self.initializer,
                                       name="first_layer_conv"))
            out = conv_layer(out)
        if self.first_sampling_params is not None:
            out = self.sample(out, self.first_sampling_params,
                              name="first_layer_sample")
        return out

    @with_name_scope("last_layer")
    def build_last_layer(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Build last convolution layer after all blocks

        Parameters
        ----------
        inputs
            inputs

        Returns
        -------
        output
            result of the last convolution
        """
        activation = (None if self.last_conv_without_activation else
                      self.activation)
        conv_layer = self.add_keras_layer(
            tf.keras.layers.Conv2D(activation=activation,
                                   **self.last_conv_params,
                                   bias_initializer=self.initializer,
                                   kernel_initializer=self.initializer,
                                   name="last_layer_conv"))
        out = conv_layer(inputs)
        return out

    def predict(self, feature_maps: tf.Tensor,
                auxiliary_feature_maps: List[tf.Tensor] = None):
        # pylint: disable=arguments-differ
        # parent save method has more generic signature
        logger = logging.getLogger(__name__)
        if not self.add_skip_connections:
            auxiliary_feature_maps = None
        auxiliary_feature_maps_reversed = None
        if auxiliary_feature_maps is not None:
            start_ind = self.auxiliary_feature_maps_start_ind
            auxiliary_feature_maps_reversed = (
                auxiliary_feature_maps[start_ind:][::-1])
        auxiliary_feature_maps_out = []
        nets = feature_maps

        # add first layer
        nets = self.build_first_layer(nets)

        for block_index in range(self.num_blocks):
            self._current_block_index = block_index
            block_input = nets
            if (auxiliary_feature_maps_reversed is None or
                    block_index >= len(auxiliary_feature_maps_reversed)):
                aux_map = None
            else:
                aux_map = auxiliary_feature_maps_reversed[block_index]
            cnn_blocks_parameters = self._get_cnn_block_parameters(block_index)
            # build cnn block
            nets = self.build_cnn_block(inputs=nets, **cnn_blocks_parameters)

            # stop if it was last layer and not sample if not wanted
            if (block_index == self.num_blocks - 1 and
                    not self.last_block_with_sampling):
                break

            auxiliary_feature_maps_out.append(nets)
            # sample block output
            nets = self.sample(nets, self.sampling_params)
            # add residual connection for the whole block
            if self.add_block_residual_connections:
                nets = self.build_residual_block_connection(block_input, nets)
            # add skip connections
            if aux_map is not None:
                logger.info("Create skip connection for block "
                            "%s from auxiliary map %s and input %s",
                            block_index + 1, aux_map, nets)
                nets = self.build_skip_connection(nets, aux_map)

        if self.last_conv_params is not None:
            nets = self.build_last_layer(nets)
        result = {'feature_maps': nets,
                  'auxiliary_feature_maps': auxiliary_feature_maps_out}
        return result

    def _get_cnn_block_parameters(self, block_index) -> dict:
        if isinstance(self.cnn_blocks_parameters, dict):
            return self.cnn_blocks_parameters
        if len(self.cnn_blocks_parameters) < block_index + 1:
            return self.cnn_blocks_parameters[-1]
        return self.cnn_blocks_parameters[block_index]


def _pad_image_to_other(image_to_pad: tf.Tensor, image_large: tf.Tensor
                        ) -> tf.Tensor:
    """
    Pad image_small with zeroes to size of image_large
    (only width and height)

    image_large has larger spatial dimensions as image_small

    Parameters
    ----------
    image_to_pad
        image to pad
    image_large
        image with size to pad to

    Returns
    -------
    image_small_padded : tensor
        image_small padded with zeros to image_large shape
    """
    image_small_shape = tf.shape(image_to_pad)[1:3]
    image_large_shape = tf.shape(image_large)[1:3]
    shape_diffs = image_large_shape - image_small_shape
    paddings = [[0, 0], [0, shape_diffs[0]], [0, shape_diffs[1]], [0, 0]]
    image_small_padded = tf.pad(image_to_pad, paddings)
    image_shall_shape_list = image_to_pad.get_shape().as_list()
    image_large_shape_list = image_large.get_shape().as_list()
    image_small_padded_new_shape = (image_shall_shape_list[:1] +
                                    image_large_shape_list[1:3] +
                                    image_shall_shape_list[-1:])
    image_small_padded.set_shape(image_small_padded_new_shape)
    return image_small_padded


def _slice_image_to_other(image_small: tf.Tensor, image_to_slice: tf.Tensor
                          ) -> tf.Tensor:
    """
    Slice image1 to fit to dimension of image2.

    image_large has larger spatial dimensions as image_small

    Parameters
    ----------
    image_small
        image to slice to
    image_to_slice
        image to slice

    Returns
    -------
    image_large_sliced
        image_large sliced to to image_small shape
    """
    image_small_shape = tf.shape(image_small)[1:3]
    image_large_shape = tf.shape(image_to_slice)[1:3]
    shape_diffs = image_large_shape - image_small_shape
    const_ = tf.constant(0, dtype=tf.int32, shape=(1,))
    slice_starts = tf.concat([const_, shape_diffs, const_], axis=0)
    image_large_sliced = tf.slice(image_to_slice, slice_starts, [-1] * 4)
    return image_large_sliced
