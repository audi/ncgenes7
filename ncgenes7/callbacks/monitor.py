# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
Callbacks to monitor training process
"""

import json
import logging
import os

import nucleus7 as nc7
from nucleus7.utils import model_utils
import numpy as np
import tensorflow as tf


class EarlyStoppingCallback(nc7.coordinator.BufferCallback):
    """
    Callback for early stopping of the training according to some metric.

    Can be used from eval mode (preferred way) to monitor some metric and then
    stop training when it is not improving.

    It uses run_context.request_stop() when there was no improvement in last
    patience number of epochs. It will save the best iteration information, e.g.
    epoch number and iteration number to {name}-best_iteration_info.json.

    It uses a AverageBuffer to save all the metrics. So the condition is
    checked when the buffer is evaluated and accumulated.

    Parameters
    ----------
    monitor_mode
        if is max, then negative change in metric is treated as degradation and
        will cause the stop
    patience
        number of epochs to monitor the metric
    min_delta
        absolute of difference that is treated as an improvement; in case of
        max mode this value will be added to current best and in case of min
        mode it will be subtracted.

    Attributes
    ----------
    incoming_keys
        * monitor : metric to monitor, []

    """
    incoming_keys = [
        "monitor",
    ]

    def __init__(self, *,
                 monitor_mode: str = 'max',
                 patience: int = 10,
                 min_delta: float = 0,
                 **callback_kwargs):
        assert monitor_mode in ["min", "max"], (
            "monitor_mode must be in ['min', 'max']")

        buffer_processor = nc7.core.BufferProcessor(
            buffer=nc7.core.AverageBuffer())
        super().__init__(buffer_processor=buffer_processor,
                         **callback_kwargs)
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.min_delta = min_delta

        self._steps_without_improvement = 0
        self._monitored_values = []
        self._current_best = None
        self._current_best_iteration_stat = {
            "epoch": None,
            "iteration": None
        }
        self._stop_var = None
        self._stop_op = None

    def begin(self):
        self._stop_var = model_utils.get_or_create_early_stop_var()
        self._stop_op = tf.assign(self._stop_var, True)

    def on_iteration_end(self, evaluate=None, *, monitor):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        if not np.shape(monitor):
            monitor = np.reshape(monitor, [1])
        return super().on_iteration_end(evaluate, monitor=monitor)

    def process_buffer(self, *, monitor):
        # pylint: disable=arguments-differ
        # parent method has more generic signature
        monitor = monitor[0]
        if self._current_best is None:
            self._set_best(monitor)
            return

        if (self.monitor_mode == "max" and
                monitor >= self._current_best + self.min_delta):
            self._set_best(monitor)
            return

        if (self.monitor_mode == "min" and
                monitor <= self._current_best - self.min_delta):
            self._set_best(monitor)
            return
        self._steps_without_improvement += 1
        self._maybe_stop()

    def _set_best(self, monitor):
        logger = logging.getLogger(__name__)
        logger.info("%s: got best monitored value", self.name)
        self._current_best = monitor
        self._steps_without_improvement = 0
        self._current_best_iteration_stat["epoch"] = (
            self.iteration_info.epoch_number)
        self._current_best_iteration_stat["iteration"] = (
            self.iteration_info.iteration_number)

    def _maybe_stop(self):
        logger = logging.getLogger(__name__)
        logger.info("%s has no improvement from for %s epochs since last "
                    "best score %s on epoch %s.",
                    self.name, self._steps_without_improvement,
                    self._current_best, self.iteration_info.epoch_number)
        if self._steps_without_improvement >= self.patience:
            logger.info("Request STOP")
            self._write_best_stats()
            self.iteration_info.session_run_context.session.run(self._stop_op)
            self.iteration_info.session_run_context.request_stop()
        else:
            logger.info("%s: continue to monitor")

    def _write_best_stats(self):
        logger = logging.getLogger(__name__)
        path = os.path.join(self.log_dir, "{}-best_iteration_info.json".format(
            self.name))
        logger.info("%s: Write best iteration info to %s", self.name, path)
        with open(path, "w") as file:
            json.dump(self._current_best_iteration_stat, file, indent=4,
                      sort_keys=True)
