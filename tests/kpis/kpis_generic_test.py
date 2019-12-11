# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

from absl.testing import parameterized
import numpy as np
import sklearn
import tensorflow as tf

from ncgenes7.kpis.generic import ConfusionMatrixKPI


class TestConfusionMatrixKPI(parameterized.TestCase, tf.test.TestCase):

    def setUp(self):
        self.num_classes = 5
        self.labels = np.array([[0, 0, 1], [2, 3, 0]])
        self.predictions = np.array([[1, 1, 1], [2, 3, 0]])

    @parameterized.parameters({"multidimensional_data": True},
                              {"multidimensional_data": False})
    def test_process(self, multidimensional_data):
        labels = self.labels if multidimensional_data else self.labels.flatten()
        predictions = (self.predictions if multidimensional_data
                       else self.predictions.flatten())
        kpi_plugin = ConfusionMatrixKPI(num_classes=self.num_classes).build()

        kpi = kpi_plugin.process(predictions=predictions,
                                 labels=labels)
        kpi_must = sklearn.metrics.confusion_matrix(
            y_true=self.labels.flatten(),
            y_pred=self.predictions.flatten(),
            labels=np.arange(self.num_classes))
        self.assertAllClose({"confusion_matrix": kpi_must},
                            kpi)
