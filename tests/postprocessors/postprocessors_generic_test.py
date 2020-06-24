# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Tests for generic post-processors
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.postprocessors.generic import ArgmaxPostProcessor
from ncgenes7.postprocessors.generic import IdentityPostProcessor
from ncgenes7.postprocessors.generic import SoftmaxPostProcessor
from ncgenes7.postprocessors.generic import NonlinearityPostProcessor


class TestIdentityPostProcessor(tf.test.TestCase):

    def setUp(self):
        np.random.seed(4564)
        tf.reset_default_graph()
        self.tensor_np = np.random.rand(10, 20, 30)
        self.list_of_tensors_np = [np.random.rand(2, 3) for _ in range(3)]
        self.dict_of_tensors_np = {
            "key1": np.random.randint(0, 5),
            "key2": np.random.randn(10),
        }
        self.inputs = {
            "scalar": 10,
            "tensor": tf.constant(self.tensor_np),
            "list_of_tensors": [
                tf.constant(each_t) for each_t in self.list_of_tensors_np],
            "dict_of_tensors": {
                k: tf.constant(v) for k, v in self.dict_of_tensors_np.items()
            }
        }
        self.outputs_must = {
            "scalar": 10,
            "tensor": self.tensor_np,
            "list_of_tensors": self.list_of_tensors_np,
            "dict_of_tensors": self.dict_of_tensors_np,
        }

    def test_call(self):
        processor = IdentityPostProcessor(
            inbound_nodes=[]).build()
        processor.mode = "train"
        outputs = processor(**self.inputs)

        outputs_without_scalar = {k: v for k, v in outputs.items()
                                  if k != "scalar"}

        outputs_eval = self.evaluate(outputs_without_scalar)
        outputs_eval["scalar"] = outputs["scalar"]
        self.assertAllClose(self.outputs_must,
                            outputs_eval)

        tensor_names = {
            "tensor": outputs["tensor"].name,
            "list_of_tensors": [t.name for t in outputs["list_of_tensors"]],
            "dict_of_tensors": {
                k: t.name for k, t in outputs["dict_of_tensors"].items()
            }
        }
        tensor_names_must = {
            "tensor": "IdentityPostProcessor/tensor:0",
            "list_of_tensors": [
                "IdentityPostProcessor/list_of_tensors//%d:0" % i
                for i in range(3)],
            "dict_of_tensors": {
                k: "IdentityPostProcessor/dict_of_tensors//%s:0" % k
                for k in self.dict_of_tensors_np}
        }
        self.assertDictEqual(tensor_names_must,
                             tensor_names)


class TestArgMaxPostProcessor(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {"keepdims": True, "axis": -1},
        {"keepdims": False, "axis": -1},
        {"keepdims": True, "axis": 1},
        {"keepdims": False, "axis": 0},
        {"keepdims": True, "axis": 1, "create_scores": True},
        {"keepdims": False, "axis": 1, "create_scores": True})
    def test_process(self, keepdims, axis, create_scores=False):
        np.random.seed(4564)
        tf.reset_default_graph()

        data_in = np.random.uniform(size=[2, 3, 4])
        output_must = {"argmax": np.argmax(data_in, axis=axis)}
        if create_scores:
            output_must["scores"] = np.max(data_in, axis=axis)
        if keepdims:
            output_must = {k: np.expand_dims(v, axis)
                           for k, v in output_must.items()}

        processor = ArgmaxPostProcessor(axis=axis,
                                        keepdims=keepdims,
                                        create_scores=create_scores,
                                        inbound_nodes=[]).build()

        output = processor.process(tf.constant(data_in))
        output_eval = self.evaluate(output)
        self.assertAllClose(output_must,
                            output_eval)


class TestSoftmaxPostProcessor(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters({"input": [[0.5, 0.2, 1.0]]},
                              {"input": [[0.1, 1.2, 1.0]]},
                              {"input": [[5.0, 1.2, 1.0], [0.1, 1.2, 1.0]]})
    def test_process(self, input):
        tf.reset_default_graph()

        input = np.asarray(input)
        exp_term = np.exp(input)
        result_must = exp_term / np.sum(exp_term, axis=-1, keepdims=True)
        processor = SoftmaxPostProcessor(axis=-1, inbound_nodes=[]).build()

        result = processor.process(tf.constant(input))
        output_eval = self.evaluate(result)
        self.assertAllClose({"softmax": result_must},
                            output_eval)


class TestNonlinearityPostProcessor(parameterized.TestCase, tf.test.TestCase):

    @parameterized.parameters(
        {"activation_name": "relu", "features": [[-0.5, 0.2, 1.0, -10]]},
        {"activation_name": "sigmoid", "features": [[-0.5, 0.2, 1.0, -10]]}
    )
    def test_process(self, activation_name, features):
        tf.reset_default_graph()
        features = np.asarray(features)
        if activation_name == 'relu':
            result_must = np.array(features)
            result_must[result_must < 0] = 0
        elif activation_name == 'sigmoid':
            result_must = 1/(1 + np.exp(-features))
        processor = NonlinearityPostProcessor(activation_name=activation_name)
        result = processor.process(tf.constant(features))
        output_eval = self.evaluate(result)
        self.assertAllClose({'features': result_must},
                            output_eval)

    def test_invalid_activation_name(self):
        tf.reset_default_graph()
        with self.assertRaises(AttributeError):
            _ = NonlinearityPostProcessor(activation_name='abc')
