# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import os
from unittest.mock import MagicMock

from absl.testing import parameterized
from nucleus7.coordinator.configs import RunIterationInfo
from nucleus7.utils.io_utils import load_json
import tensorflow as tf

from ncgenes7.callbacks.monitor import EarlyStoppingCallback


class TestEarlyStoppingCallback(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.log_dir = self.get_temp_dir()
        self.min_delta = 1.0
        self.patience = 5
        self.max_epochs = 15
        self.last_epoch = 11
        self.best_epoch = 6
        self.monitor_data_max = [0, 1, 2, 1, 4, 7, 5, 5, 7, 5, 1, 4, 7, 7, 5]
        self.monitor_data_min = [i * -1 for i in self.monitor_data_max]
        self.final_stat_fname = os.path.join(
            self.log_dir, "early_stopping-best_iteration_info.json")

    @parameterized.parameters({"monitor_mode": "max"},
                              {"monitor_mode": "min"})
    def test_on_iteration_end(self, monitor_mode):
        callback = EarlyStoppingCallback(inbound_nodes=[],
                                         monitor_mode=monitor_mode,
                                         min_delta=self.min_delta,
                                         patience=self.patience,
                                         name="early_stopping").build()
        callback.log_dir = self.log_dir

        with self.test_session() as sess:
            callback.begin()
            sess.run(tf.global_variables_initializer())
            for epoch_number in range(1, self.max_epochs + 1):
                inputs, iteration_info, should_stop = self._get_inputs(
                    epoch_number, monitor_mode, sess)
                callback.iteration_info = iteration_info
                callback.on_iteration_end(**inputs)
                run_context = callback.iteration_info.session_run_context
                result_stop_var = sess.run(callback._stop_var)
                if should_stop:
                    run_context.request_stop.assert_called_once_with()
                    self.assertTrue(run_context.stop_requested)
                    self.assertTrue(result_stop_var)
                else:
                    run_context.request_stop.assert_not_called()
                    self.assertFalse(run_context.stop_requested)
                    self.assertFalse(result_stop_var)
                if run_context.stop_requested:
                    break
        self.assertTrue(os.path.isfile(self.final_stat_fname))

        best_iter_info = load_json(self.final_stat_fname)
        best_iter_info_must = {"epoch": self.best_epoch,
                               "iteration": self.best_epoch * 100}
        self.assertDictEqual(best_iter_info_must,
                             best_iter_info)

    def _get_inputs(self, epoch_number, monitor_mode, sess):
        run_context = tf.train.SessionRunContext([], session=sess)
        run_context.request_stop = MagicMock(wraps=run_context.request_stop)
        iteration_info = RunIterationInfo(epoch_number, epoch_number * 100,
                                          0, True, run_context)
        monitor_data = (self.monitor_data_max if monitor_mode == "max"
                        else self.monitor_data_min)
        inputs = {"monitor": monitor_data[epoch_number - 1]}
        should_stop = False
        if epoch_number == self.last_epoch:
            should_stop = True
        return inputs, iteration_info, should_stop
