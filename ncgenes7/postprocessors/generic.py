# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Generic postprocessors
"""
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import nucleus7 as nc7
import tensorflow as tf


class IdentityPostProcessor(nc7.model.ModelPostProcessor):
    """
    Identity post-processor

    Will add the tf.identity on all the flatten inputs and unflatten them back
    to outputs. Generated keys are the same as incoming keys

    """
    dynamic_incoming_keys = True
    dynamic_generated_keys = True

    def process(self, **inputs) -> Dict[str, tf.Tensor]:
        return inputs


class ArgmaxPostProcessor(nc7.model.ModelPostProcessor):
    """
    Argmax post-processor to extract the class of the maximum logits

    Parameters
    ----------
    axis
        axis parameter passed to tf.argmax
    create_scores
        if the scores should be calculated; if score_conversion_fn_name
        is not None, then create_scores is set to True
    score_conversion_fn_name
        name of the function from 'tf.nn' namescope to use for scores
        conversion
    keepdims
        if the dimensionality should be preserved, e.g. singleton shape will
        be at the axis

    Attributes
    ----------
    incoming_keys
        * features : input features
    generated_keys
        * argmax : position of maximum along specified axis, tf.int64
        * scores : (optional) max scores along axis after applying
          score_conversion_fn on input features
    """
    incoming_keys = [
        'features',
    ]
    generated_keys = [
        'argmax',
        '_scores',
    ]

    def __init__(self,
                 *,
                 axis: Union[int, List[int]] = -1,
                 score_conversion_fn_name: Optional[str] = None,
                 create_scores: bool = False,
                 keepdims: bool = False,
                 **postprocessor_kwargs: dict):
        super().__init__(**postprocessor_kwargs)
        self.axis = axis
        self.score_conversion_fn_name = score_conversion_fn_name
        self.create_scores = (create_scores
                              or self.score_conversion_fn_name is not None)
        self.keepdims = keepdims
        self._score_conversion_fn = None

    def build(self):
        super().build()
        if self.score_conversion_fn_name:
            self._score_conversion_fn = getattr(
                tf.nn, self.score_conversion_fn_name)
        else:
            self._score_conversion_fn = tf.identity
        return self

    def process(self,
                features: tf.Tensor
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        argmax_indices = tf.argmax(features, axis=self.axis)
        if self.keepdims:
            argmax_indices = tf.expand_dims(argmax_indices, self.axis)
        result = {"argmax": argmax_indices}
        if self.create_scores:
            scores = self._get_max_scores(features)
            result["scores"] = scores
        return result

    def _get_max_scores(self, features: tf.Tensor):
        try:
            scores = self._score_conversion_fn(features, axis=self.axis)
        except TypeError:
            scores = self._score_conversion_fn(features)

        max_scores = tf.reduce_max(
            scores, axis=self.axis, keepdims=self.keepdims)
        return max_scores


class SoftmaxPostProcessor(nc7.model.ModelPostProcessor):
    """
    Softmax post-processor to get the probabilities from logits

    Parameters
    ----------
    axis
        axis parameter passed to tf.nn.softmax

    Attributes
    ----------
    incoming_keys
        * logits : input logits in shape of [bs, ...], tf.float32
    generated_keys
        * softmax : applied softmax on logits in shape of
          [bs, ...], tf.float32
    """
    incoming_keys = ['logits']
    generated_keys = ['softmax']

    def __init__(self, *,
                 axis: Union[int, List[int]] = -1,
                 **postprocessor_kwargs: dict):
        super().__init__(**postprocessor_kwargs)
        self.axis = axis

    def process(self, logits: Union[tf.Tensor, List[tf.Tensor]]
                ) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        softmax = tf.nn.softmax(logits, axis=self.axis)
        return {"softmax": softmax}


class NonlinearityPostProcessor(nc7.model.ModelPostProcessor):
    """
    Generic post-processor for non-linear activations

    Parameters
    ----------
    activation
        Name of the Keras activation, e.g. 'relu', 'selu'

    Attributes
    ----------
    incoming_keys
        * features : original tensor, tf.Tensor
    generated_keys
        * features : tensor after applying the activation, tf.Tensor
    """
    incoming_keys = ["features"]
    generated_keys = ["features"]

    def __init__(self, *,
                 activation_name: str,
                 **postprocessor_kwargs: dict):
        super().__init__(**postprocessor_kwargs)
        if not hasattr(tf.keras.activations, activation_name):
            raise AttributeError(
                '{} activation not found in tf.keras.activations'
                'namespace'.format(activation_name))
        self.activation = getattr(tf.keras.activations, activation_name)

    def process(self, features: tf.Tensor) -> Dict[str, tf.Tensor]:
        # pylint: disable=arguments-differ
        # base class has more generic signature
        return {'features': self.activation(features)}
