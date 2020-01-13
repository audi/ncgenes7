# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from io import StringIO
import unittest
from unittest.mock import patch

from nucleus7.coordinator.configs import RunIterationInfo
import numpy as np

from ncgenes7.callbacks.generic import BaseLogger


class TestBaseLogger(unittest.TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_on_iteration_end(self, mock_stdout):
        callback = BaseLogger(inbound_nodes=['node1'])
        mode = 'train'
        epoch = 10
        iter_n = 112
        time_exec = 10.1
        iter_per_epoch = 256
        data = {'loss': {'total_loss': 10, 'loss_2': 20},
                'metric': {'metric_1': np.array([10]),
                           'metric_2': None}}

        iteration_info = RunIterationInfo(epoch_number=epoch,
                                          iteration_number=iter_n,
                                          execution_time=time_exec)
        callback.iteration_info = iteration_info
        callback.mode = mode
        callback.number_iterations_per_epoch = iter_per_epoch
        callback.on_iteration_end(**data)
        time_remain = time_exec / iter_n * iter_per_epoch - time_exec
        printed = mock_stdout.getvalue()
        printed_lines = printed.split('\n')[:-1]
        self.assertEqual(len(printed_lines), 2)
        self.assertEqual(len(printed_lines[0]), len(printed_lines[1]))

        printed_names = printed_lines[0].strip().split('|')[1:-1]
        printed_values = printed_lines[1].strip().split('|')[1:-1]
        self.assertListEqual([len(n) for n in printed_names],
                             [len(v) for v in printed_values])
        printed_names_str = list(map(str.strip, printed_names))
        printed_values_str = list(map(str.strip, printed_values))
        names_after_stat = printed_names_str[5:]

        self.assertListEqual(names_after_stat,
                             ['loss_2', 'total_loss', 'metric_1'])
        printed_dict = dict(zip(printed_names_str, printed_values_str))
        printed_dict = {k: float(v) if k not in ['mode', 'iter'] else v
                        for k, v in printed_dict.items()}
        not_printable_key = 'metric_wrong'
        self.assertNotIn(not_printable_key, printed_dict)
        printed_must = {'mode': mode, 'epoch': epoch,
                        'time_exec, [s]': round(time_exec, 2),
                        'time_remain, [s]': round(time_remain, 2),
                        'iter': '{}/{}'.format(iter_n, iter_per_epoch),
                        'total_loss': 10,
                        'loss_2': 20,
                        'metric_1': np.array([10])}
        self.assertDictEqual(printed_dict, printed_must)

        mock_stdout.truncate(0)
        callback.on_iteration_end(**data)
        printed = mock_stdout.getvalue()
        printed_lines = printed.split('\n')[:-1]
        self.assertEqual(len(printed_lines), 1)

    def test_on_iteration_end_invalid_values(self):
        callback = BaseLogger(inbound_nodes=['node1'])
        mode = 'train'
        epoch = 10
        iter_n = 112
        time_exec = 10.1
        iter_per_epoch = 256
        iteration_info = RunIterationInfo(epoch_number=epoch,
                                          iteration_number=iter_n,
                                          execution_time=time_exec)
        callback.iteration_info = iteration_info
        callback.mode = mode
        callback.number_iterations_per_epoch = iter_per_epoch
        with self.assertRaises(ValueError):
            data = {'loss': {'total_loss': 10, 'loss_2': [20, 10]},
                    'metric': {'metric_1': None}}
            callback.on_iteration_end(**data)

        with self.assertRaises(ValueError):
            data = {'loss': 0.1,
                    'metric': 'some_str_value'}
            callback.on_iteration_end(**data)
