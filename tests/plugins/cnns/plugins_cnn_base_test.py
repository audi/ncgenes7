# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import math

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.plugins.cnns.base import BaseCNNPlugin


class TestBaseCNNPlugin(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        tf.reset_default_graph()

    def test_empty_graph_after_build(self):
        tf.reset_default_graph()
        plugin = BaseCNNPlugin(cnn_blocks_parameters={}).build()
        self.assertListEqual([],
                             tf.get_default_graph().get_operations())

    @parameterized.parameters(
        {"sampling_type": "encoder"},
        {"sampling_type": "encoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5,
                               "padding": "same"},
         "first_sampling_params": {"kernel_size": 2, "strides": 2}},
        {"multi_processes": True,
         "sampling_type": "encoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2,
                              "padding": "same"},
         "add_block_residual_connections": True},
        {"multi_processes": True,
         "sampling_type": "decoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5,
                               "padding": "same"},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2,
                              "padding": "same"},
         "add_block_residual_connections": True,
         "add_skip_connections": True},
        {"sampling_type": "decoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5,
                               "padding": "same"},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2,
                              "padding": "same"},
         "add_block_residual_connections": True,
         "add_skip_connections": True},
        {"inputs_with_nan_dimensions": True,
         "sampling_type": "encoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2},
         "add_block_residual_connections": True,
         "block_residual_connection_type": "concat"},
        {"sampling_type": "decoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5,
                               "padding": "same"},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2,
                              "padding": "same"},
         "add_block_residual_connections": True,
         "add_skip_connections": True,
         "skip_connection_type": "sum",
         "skip_connection_resize_type": "pad",
         "block_residual_connection_type": "concat"},
        {"sampling_type": "decoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5,
                               "padding": "same"},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2,
                              "padding": "same"},
         "add_block_residual_connections": True,
         "add_skip_connections": True,
         "skip_connection_type": "sum",
         "skip_connection_resize_type": "imresize",
         "block_residual_connection_type": "concat"},
        {"inputs_with_nan_dimensions": True,
         "sampling_type": "decoder",
         "first_conv_params": {"filters": 10, "kernel_size": 5},
         "first_sampling_params": {"kernel_size": 2, "strides": 2},
         "last_conv_params": {"filters": 30, "kernel_size": 2, "strides": 2},
         "add_block_residual_connections": True,
         "add_skip_connections": True},
    )
    def test_predict_tf(self,
                        sampling_type,
                        first_conv_params=None,
                        first_sampling_params=None,
                        last_conv_params=None,
                        num_blocks=3,
                        add_skip_connections=False,
                        skip_connection_type="concat",
                        skip_connection_resize_type=None,
                        last_conv_without_activation=False,
                        last_block_with_sampling=True,
                        add_block_residual_connections=False,
                        block_residual_connection_type="sum",
                        multi_processes=False,
                        cnn_blocks_parameters=None,
                        inputs_with_nan_dimensions=False,
                        ):
        if cnn_blocks_parameters is None:
            cnn_blocks_parameters = [
                {"filters": 5 + i, "kernel_size": 3, "padding": "same"}
                for i in range(num_blocks)]
        plugin = BaseCNNPlugin(
            sampling_type=sampling_type,
            num_blocks=num_blocks,
            first_conv_params=first_conv_params,
            first_sampling_params=first_sampling_params,
            last_conv_params=last_conv_params,
            last_conv_without_activation=last_conv_without_activation,
            last_block_with_sampling=last_block_with_sampling,
            cnn_blocks_parameters=cnn_blocks_parameters,
            add_skip_connections=add_skip_connections,
            skip_connection_type=skip_connection_type,
            skip_connection_resize_type=skip_connection_resize_type,
            add_block_residual_connections=add_block_residual_connections,
            block_residual_connection_type=block_residual_connection_type,
        ).build()

        if inputs_with_nan_dimensions:
            (feature_maps, auxiliary_feature_maps
             ) = self._get_inputs_nan_dimensions(num_blocks)
        else:
            if sampling_type == "encoder":
                (feature_maps, auxiliary_feature_maps
                 ) = self._get_inputs_encoder(
                    num_blocks=num_blocks,
                    add_first_convolution=first_conv_params is not None)
            else:
                (feature_maps, auxiliary_feature_maps
                 ) = self._get_inputs_decoder(num_blocks=num_blocks)

        inputs = {"feature_maps": feature_maps}
        if add_skip_connections:
            inputs["auxiliary_feature_maps"] = auxiliary_feature_maps
        result = plugin.predict(**inputs)

        self.assertSetEqual(set(plugin.generated_keys_all),
                            set(result.keys()))

        if multi_processes:
            trainable_vars = tf.trainable_variables()
            result = plugin.predict(**inputs)
            self.assertSetEqual(set(trainable_vars),
                                set(tf.trainable_variables()))
            plugin.reset_keras_layers()
            result = plugin.predict(**inputs)
            self.assertEqual(len(trainable_vars) * 2,
                             len(tf.trainable_variables()))

        result_shapes = {
            "feature_maps": result["feature_maps"].shape.as_list(),
            "auxiliary_feature_maps": tuple([
                each_map.shape.as_list()
                for each_map in result["auxiliary_feature_maps"]])
        }

    @parameterized.parameters({"inputs_with_nan_dimensions": True},
                              {"inputs_with_nan_dimensions": False})
    def test_encoder_decoder(self, inputs_with_nan_dimensions):
        cnn_blocks_parameters_encoder = [
            {"filters": 20, "kernel_size": 5, "padding": "same"},
            {"filters": 30, "kernel_size": 3, "padding": "same"},
            {"filters": 40, "kernel_size": 3, "padding": "same"},
            {"filters": 50, "kernel_size": 3, "padding": "same"},
            {"filters": 60, "kernel_size": 1, "padding": "same"},
        ]
        cnn_blocks_parameters_decoder = [
            {"filters": 60, "kernel_size": 3, "padding": "same"},
            {"filters": 20, "kernel_size": 3, "padding": "same"},
            {"filters": 5, "kernel_size": 3, "padding": "same"},
        ]
        encoder = BaseCNNPlugin(
            sampling_type="encoder",
            num_blocks=len(cnn_blocks_parameters_encoder),
            first_conv_params={"filters": 10, "kernel_size": 5,
                               "strides": 2, "padding": "same"},
            first_sampling_params={"kernel_size": 2, "strides": 2},
            last_conv_params={"filters": 20, "kernel_size": 1,
                              "padding": "same"},
            last_conv_without_activation=False,
            last_block_with_sampling=True,
            cnn_blocks_parameters=cnn_blocks_parameters_encoder,
            add_block_residual_connections=True,
        ).build()

        decoder = BaseCNNPlugin(
            sampling_type="decoder",
            num_blocks=len(cnn_blocks_parameters_decoder),
            first_conv_params={"filters": 128, "kernel_size": 1,
                               "padding": "same"},
            last_conv_params={"filters": 20, "kernel_size": 1,
                              "padding": "same"},
            last_conv_without_activation=True,
            last_block_with_sampling=False,
            cnn_blocks_parameters=cnn_blocks_parameters_decoder,
            add_block_residual_connections=True,
            add_skip_connections=True,
            skip_connection_resize_type="pad",
            skip_connection_type="concat",
        ).build()

        if inputs_with_nan_dimensions:
            feature_maps = tf.placeholder(tf.float32, [None, None, None, 3])
        else:
            feature_maps = tf.placeholder(tf.float32, [None, 206, 289, 3])

        encoder_result = encoder.predict(feature_maps=feature_maps)
        decoder_result = decoder.predict(**encoder_result)

    def _get_inputs_nan_dimensions(self, num_blocks):
        feature_maps = tf.placeholder(tf.float32, [None, None, None, 3])
        auxiliary_feature_maps = [
            tf.placeholder(tf.float32, [None, None, None, 10])
            for _ in range(num_blocks)]
        return feature_maps, auxiliary_feature_maps

    def _get_inputs_encoder(self, num_blocks, add_first_convolution):
        feature_maps = tf.placeholder(tf.float32, [None, 100, 120, 3])
        ind_inc = int(add_first_convolution)
        auxiliary_feature_maps = [
            tf.placeholder(tf.float32,
                           [None, math.ceil(100 / 2 ** (i + ind_inc)),
                            math.ceil(120 / 2 ** (i + ind_inc)), 3])
            for i in range(num_blocks)]
        return feature_maps, auxiliary_feature_maps

    def _get_inputs_decoder(self, num_blocks):
        feature_maps = tf.placeholder(tf.float32, [None, 13, 15, 10])
        auxiliary_feature_maps = [
            tf.placeholder(tf.float32,
                           [None, math.ceil(100 / 2 ** i),
                            math.ceil(120 / 2 ** i), 3])
            for i in range(num_blocks)]
        return feature_maps, auxiliary_feature_maps
