# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import tensorflow as tf

from ncgenes7.summaries.generic import BaseSummary


class TestBaseSummary(parameterized.TestCase, tf.test.TestCase):

    def test_base_summary(self):
        data = {k: v for k, v in zip(
            ['images', 'text', 'scalar', 'audio', 'histogram'],
            ['data1', 'data2', 'data3', 'data4', 'data5'])}
        summary_name = 'summary_name'
        summary = BaseSummary(inbound_nodes=[], name=summary_name)
        summary.mode = 'train'
        res = summary.process(**data)
        res_must = {'image_summary_name': 'data1',
                    'text_summary_name': 'data2',
                    'scalar_summary_name': 'data3',
                    'audio_summary_name': 'data4',
                    'histogram_summary_name': 'data5'}
        self.assertDictEqual(res, res_must)

    def test_base_summary_dict(self):
        data = {k: v for k, v in zip(
            ['images', 'scalar', 'audio'],
            [{'im1': 'data1', 'im2': 'data2'},
             {'sc1': 'data3', 'sc2': 'data4'},
             ['data5', 'data6']])}
        summary_name = 'summary_name'
        summary = BaseSummary(inbound_nodes=[], name=summary_name)
        summary.mode = 'train'
        res = summary.process(**data)
        res_must = {'image_summary_name_im1': 'data1',
                    'image_summary_name_im2': 'data2',
                    'scalar_summary_name_sc1': 'data3',
                    'scalar_summary_name_sc2': 'data4',
                    'audio_summary_name_0': 'data5',
                    'audio_summary_name_1': 'data6'}
        self.assertDictEqual(res, res_must)
