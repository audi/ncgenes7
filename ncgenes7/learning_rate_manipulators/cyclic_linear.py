# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Cyclic learning rate with linear transition and warmup from 0
"""
from typing import Dict

from nucleus7.optimization import LearningRateManipulator
import tensorflow as tf


class CyclicLinearLearningRate(LearningRateManipulator):
    """
    Provide a cyclic learning rate with linear transition between maximum
    and minimum learning rate. In the first half cycle, the learning rate will
    be raised from 0 on

    Parameters
    ----------
    number_cycle_steps
        The full cycle length
    minimum_learning_rate_factor
        The relative minimum learning rate. E.g. if you want to be your minimum
        learning rate to be 1/10th of the specified learning rate set it to 0.1
    """

    def __init__(self,
                 number_cycle_steps: int,
                 minimum_learning_rate_factor: float = 0.1):
        self.minimum_learning_rate_factor = minimum_learning_rate_factor
        self.number_cycle_steps = number_cycle_steps
        super().__init__()

    def get_current_learning_rate(
            self,
            initial_learning_rate: float,
            global_step: tf.Tensor) -> tf.Tensor:
        slopes_and_offset = self._calculate_slopes_and_offset(
            initial_learning_rate)
        current_learning_rate = self._calculate_learning_rate(
            global_step, slopes_and_offset)
        return current_learning_rate

    def _calculate_slopes_and_offset(
            self, initial_learning_rate: float) -> Dict[str, float]:
        """
        Calculates the slopes for both warmup and later cycles as well as the
        offset (=minimum learning rate)

        Parameters
        ----------
        initial_learning_rate
            the initial learning rate

        Returns
        -------
        slopes_and_offset_dict
            dict containing the slopes for both 'warmup' and 'normal' phase as
            well as the offset (=minimum_learning_rate)
        """
        half_steps = float(self.number_cycle_steps / 2.0)
        slope_warmup = initial_learning_rate / half_steps
        slope_normal = (initial_learning_rate
                        * (1.0 - self.minimum_learning_rate_factor)
                        / half_steps)
        offset = initial_learning_rate * self.minimum_learning_rate_factor
        return {'normal': slope_normal,
                'warmup': slope_warmup,
                'offset': offset}

    def _calculate_learning_rate(
            self,
            global_step: tf.Tensor,
            slopes_and_offset: Dict[str, float]) -> tf.Tensor:
        """
        Calculate the learning rate
        """
        step_as_int = tf.to_int64(global_step)

        def _build_cylic_lr():
            relative_step = tf.to_double(
                tf.mod(step_as_int, self.number_cycle_steps))
            raising_lr = (relative_step
                          * slopes_and_offset['normal']
                          + slopes_and_offset['offset'])
            decreasing_lr = ((slopes_and_offset['normal']
                              * self.number_cycle_steps
                              + slopes_and_offset['offset'])
                             - (relative_step
                                * slopes_and_offset['normal']))
            _cyclic_lr = tf.minimum(raising_lr, decreasing_lr)
            return _cyclic_lr

        def _build_warmup_lr():
            step_to_use = tf.to_double(
                tf.minimum(step_as_int, self.number_cycle_steps))
            _warmup_lr = step_to_use * slopes_and_offset['warmup']
            return _warmup_lr

        cyclic_lr = _build_cylic_lr()
        warmup_lr = _build_warmup_lr()

        current_lr = tf.to_float(
            tf.minimum(cyclic_lr, warmup_lr))

        return current_lr
