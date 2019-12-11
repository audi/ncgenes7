# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import pytest
import tensorflow as tf

from ncgenes7.plugins.keras_applications import TFKerasApplicationsPlugin


@pytest.mark.not_available_tf1_11
class TestTFKerasApplicationsPlugin(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        tf.reset_default_graph()
        self.model_name = "MobileNetV2"
        self.model_plugin_params = {"include_top": False}
        self.input_shape = [1, 244, 244, 3]
        self.shape_must = [1, 7, 7, 1280]
        self.input_sample = np.random.random(self.input_shape)

    @parameterized.parameters({"weights": "imagenet"},
                              {"weights": None})
    def test_build(self, weights):
        _ = self._get_model_plugin(weights=weights)
        self.assertSetEqual(set(),
                            set(tf.trainable_variables()))

    def test_predict_shapes(self):
        input_tensor = tf.constant(self.input_sample, tf.float32)
        model_plugin = self._get_model_plugin()
        model_plugin_features = model_plugin.predict(input_tensor)
        self.assertListEqual(
            self.shape_must,
            model_plugin_features["features"].shape.as_list())

    def test_predict_shapes_with_feed_dict(self):
        model_plugin = self._get_model_plugin()
        input_tensor = tf.placeholder(tf.float32, shape=self.input_shape)
        result_features = model_plugin.predict(features=input_tensor)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(result_features,
                              feed_dict={input_tensor: self.input_sample})
            result_shape = list(result['features'].shape)
            self.assertListEqual(self.shape_must,
                                 result_shape)

    @pytest.mark.slow
    def test_load_weights(self):
        model_keras_params = {"include_top": False}
        input_tensor = tf.constant(self.input_sample, tf.float32)
        with self.test_session():
            model_plugin = self._get_model_plugin(weights="imagenet")
            model_plugin.mode = "train"
            _ = model_plugin.predict(input_tensor)
            model_from_plugin = model_plugin.keras_layers_with_names.get(
                model_plugin._model_name)
            model_plugin_weights = model_from_plugin.get_weights()

        tf.reset_default_graph()
        model_keras_class_name = getattr(tf.keras.applications,
                                         self.model_name)
        model_keras = model_keras_class_name(weights="imagenet",
                                             **model_keras_params)
        model_keras_weights = model_keras.get_weights()
        self.assertAllClose(model_plugin_weights,
                            model_keras_weights)

    def test_multi_process(self):
        input_tensor = tf.constant(self.input_sample, tf.float32)
        model_plugin = self._get_model_plugin(weights=None)
        result = model_plugin.predict(features=input_tensor)
        trainable_vars = tf.trainable_variables()
        self.assertSetEqual(set(trainable_vars),
                            set(tf.trainable_variables()))
        model_plugin.reset_keras_layers()
        result = model_plugin.predict(features=input_tensor)
        self.assertEqual(len(trainable_vars) * 2,
                         len(tf.trainable_variables()))

    def _get_model_plugin(self, weights=None):
        model_plugin = TFKerasApplicationsPlugin(
            keras_application_model_name=self.model_name,
            model_params={"weights": weights, **self.model_plugin_params},
        ).build()
        return model_plugin
