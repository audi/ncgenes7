# ==============================================================================
# Copyright (c) 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from ncgenes7.kpis.semantic_segmentation import SemanticSegmentationMeanIOUKPI


class TestSemanticSegmentationMeanIOUKPI(tf.test.TestCase):

    def setUp(self):
        self.num_classes = 3
        self.data = [
            {"confusion_matrix": np.array([[1, 0, 0],
                                           [0, 0, 0],
                                           [1, 0, 1]])},
            {"confusion_matrix": np.array([[0, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 1]])},
            {"confusion_matrix": np.array([[3, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]])},
            {"confusion_matrix": np.array([[0, 0, 5],
                                           [0, 0, 0],
                                           [0, 0, 0]])},
            {"confusion_matrix": np.array([[2, 0, 1],
                                           [0, 0, 0],
                                           [0, 0, 2]])},
        ]
        self.kpi_must = {
            "meanIoU": 0.4125874125874126,
            "iou-classwise-class_0": 0.46153846153846156,
            "iou-classwise-class_1": np.NAN,
            "iou-classwise-class_2": 0.36363636363636365,
        }

    def test_evaluate_on_sample(self):
        kpi_accumulator = SemanticSegmentationMeanIOUKPI(
            num_classes=self.num_classes).build()
        for i_sample, each_sample in enumerate(self.data):
            evaluate_flag = i_sample == len(self.data) - 1
            _ = kpi_accumulator.evaluate_on_sample(evaluate=evaluate_flag,
                                                   **each_sample)
        self.assertAllClose(self.kpi_must,
                            kpi_accumulator.last_kpi)
