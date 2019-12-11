# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Provide the tensorflow decays but using also warmup
"""
from nucleus7.optimization import LearningRateManipulator
from nucleus7.optimization import TFLearningRateDecay
import tensorflow as tf


class TFLearningRateDecayWithWarmup(LearningRateManipulator):
    """
    Wrap the default tensorflow learning rate manipulation with a warmup phase

    Parameters
    ----------
    number_warmup_step
        number of steps to reach the initial learning rate
    decay_type_name
        type of the decay
    **decay_params
        parameters of the tensorflow decay
    """

    def __init__(
            self,
            number_warmup_steps: int,
            decay_type_name: str,
            **decay_params):
        super(TFLearningRateDecayWithWarmup, self).__init__()
        self.number_warmup_steps = number_warmup_steps
        self.tf_learning_rate_decay = TFLearningRateDecay(
            decay_type_name, **decay_params)

    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor):
        step_as_int = tf.to_int64(global_step)

        def _build_warmup_lr():
            slope = float(
                initial_learning_rate / float(self.number_warmup_steps))
            step_to_use = tf.to_double(
                tf.minimum(step_as_int, self.number_warmup_steps * 2))
            _warmup_lr = step_to_use * slope
            return tf.to_float(_warmup_lr)

        tf_lr = self.tf_learning_rate_decay.get_current_learning_rate(
            initial_learning_rate, global_step)
        warmup_lr = _build_warmup_lr()

        current_lr = tf.minimum(tf_lr, warmup_lr)

        return current_lr
