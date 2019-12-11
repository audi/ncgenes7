# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

"""
General purpose callback implementations
"""
import numbers

import nucleus7 as nc7
import numpy as np


class BaseLogger(nc7.coordinator.CoordinatorCallback):
    """
    Callback class for base logging

    Parameters
    ----------
    print_header_n_lines
        prints the header each print_header_n_lines lines

    Attributes
    ----------
    incoming_keys
        * losses : (optional) list of losses to print or just one loss
        * metrics : (optional) list of metrics to print or just one loss
    _lines_printed
        number of lines already printed for iteration
    _min_col_width
        minimum column width for printing
    _header_iter
        header string for iteration prints
    _header_widths_iter
        header width for iteration prints
    """
    incoming_keys = ["_loss", "_metric"]

    def __init__(self, *,
                 print_header_n_lines: int = 20,
                 **callback_kwargs):
        super(BaseLogger, self).__init__(**callback_kwargs)
        self.print_header_n_lines = print_header_n_lines
        self._lines_printed = 0
        self._min_col_width = 11
        self._header_iter = None
        self._header_widths_iter = None
        self._time_elapsed = 0

    def on_iteration_end(self, *, loss=None, metric=None):
        # pylint: disable=arguments-differ
        # parent on_iteration_end method has more generic signature
        iteration_number = self.iteration_info.iteration_number
        epoch_number = self.iteration_info.epoch_number
        execution_time = self.iteration_info.execution_time
        number_iterations_per_epoch = self.number_iterations_per_epoch
        if iteration_number <= 1:
            self._lines_printed = 0
            self._time_elapsed = 0
        if iteration_number <= 1:
            print('\nStarting epoch {} in mode {} with {} iterations'.format(
                epoch_number, self.mode, number_iterations_per_epoch))
        self._time_elapsed += execution_time
        time_remain = self._get_remaining_time_to_end(
            iteration_number, number_iterations_per_epoch)

        printable_names, printable_values = (
            _get_printable_names_and_values(loss, metric))
        if self._header_iter is None:
            self._get_print_header(printable_names)
        iter_value = (
            "{}/{}".format(iteration_number, number_iterations_per_epoch)
            if number_iterations_per_epoch > 0 else str(iteration_number))
        print_values = [self.mode, epoch_number, iter_value, execution_time,
                        time_remain]
        print_values.extend(printable_values)

        print_formats = ['{:^{w}s}', '{:^{w}d}', '{:^{w}s}']
        print_formats.extend(['{:^{w}.3e}'
                              for _ in range(len(print_values) - 3)])
        print_str = '|'.join([f.format(v, w=w) for v, f, w in zip(
            print_values, print_formats, self._header_widths_iter)])
        print_str = '|' + print_str + '|'
        if self._lines_printed % self.print_header_n_lines == 0:
            print(self._header_iter)
        self._lines_printed += 1
        print(print_str)

    def _get_print_header(self, printable_names):
        header_str = (['mode', 'epoch', 'iter', 'time_exec, [s]',
                       'time_remain, [s]'] + list(printable_names))
        self._header_widths_iter = [max(self._min_col_width, len(s) + 2)
                                    for s in header_str]
        self._header_iter = '|'.join(
            ['{:^{w}s}'.format(s, w=w) for s, w in
             zip(header_str, self._header_widths_iter)])
        self._header_iter = '|' + self._header_iter + '|'

    def _get_remaining_time_to_end(self, iter_n, iter_per_epoch):
        if iter_per_epoch < 0 or iter_n <= 0 or self._time_elapsed < 0:
            return -1
        time_remaining = (self._time_elapsed / iter_n * iter_per_epoch -
                          self._time_elapsed)
        return time_remaining


def _get_value_printable(value):
    if isinstance(value, numbers.Number):
        return value
    if (isinstance(value, np.ndarray)
            and value.ndim == 1 and value.shape[0] == 1):
        return value[0]
    if (isinstance(value, np.ndarray)
            and value.ndim == 0):
        return value
    return None


def _get_printable_names_and_values(loss, metric):
    loss = loss or {}
    metric = metric or {}
    if not isinstance(loss, dict):
        loss = {'loss': loss}
    if not isinstance(metric, dict):
        metric = {'metric': metric}
    loss_names = sorted(loss)
    metric_names = sorted(metric)
    loss = [loss[name] for name in loss_names]
    metric = [metric[name] for name in metric_names]
    names = loss_names + metric_names
    values = loss + metric
    if not names:
        printable_names, printable_values = [], []
    else:
        printable_names, printable_values = zip(*(
            [(n, _get_value_printable(v)) for (n, v) in zip(names, values)
             if _get_value_printable(v) is not None]))
    return printable_names, printable_values
