# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from unittest.mock import patch

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from ncgenes7.losses.generic import GenericTFLoss


class TestGenericTFLoss(parameterized.TestCase, tf.test.TestCase):

    @patch('tensorflow.losses.softmax_cross_entropy', return_value=156)
    def test_process(self, loss_fun):
        tf_fun_name = 'softmax_cross_entropy'
        loss = GenericTFLoss(tf_fun_name=tf_fun_name,
                             arg_name_labels='onehot_labels',
                             arg_name_predictions='logits')
        loss.mode = 'train'
        predictions = np.zeros([10, 10])
        labels = np.ones([10, 10])
        res = loss.process(predictions=predictions, labels=labels)

        data_used = loss_fun.call_args[1]
        data_must = {'onehot_labels': labels,
                     'logits': predictions}

        self.assertEqual(loss_fun.call_count, 1)
        self.assertSetEqual(set(data_used), set(data_must))
        for k in data_must:
            self.assertAllClose(data_used[k], data_must[k])

        res_must = {'loss': 156}
        self.assertDictEqual(res, res_must)
