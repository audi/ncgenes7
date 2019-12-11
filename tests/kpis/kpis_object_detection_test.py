# ==============================================================================
# Copyright @ 2019 Audi Electronics Venture GmbH. All rights reserved.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License, v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ==============================================================================

import numpy as np
import tensorflow as tf

from ncgenes7.kpis.object_detection import DetectionsMatcherKPIPlugin
from ncgenes7.kpis.object_detection import ObjectDetectionKPIAccumulator


class TestDetectionsMatcherKPIPlugin(tf.test.TestCase):

    def setUp(self):
        self.number_of_samples = 6
        self.num_classes = 2
        self.groundtruth_object_boxes = [
            np.array([[0, 0, 1, 1],
                      [1, 1, 10, 10]]).astype(np.float32),
            np.array([[0, 0, 0, 0]]).astype(np.float32),
            np.array([[0.2, 0.3, 0.4, 0.5]]),
            np.array([[]]).astype(np.float32).reshape([0, 4]),
            np.array([[]]).astype(np.float32).reshape([0, 4]),
            np.array([[0, 0, 1, 1]]).astype(np.float32),
        ]
        self.groundtruth_object_classes = [
            np.array([1, 1]),
            np.array([0]),
            np.array([2]),
            np.array([]).reshape([0]),
            np.array([]).reshape([0]),
            np.array([1]),
        ]
        self.detection_object_boxes = [
            np.array([[0, 0, 1, 1],
                      [1, 1, 5, 5],
                      [2, 2, 10, 10]]).astype(np.float32),
            np.array([[0.1, 0.1, 0.5, 0.5],
                      [0.5, 0.5, 0.6, 0.6]]),
            np.array([[0.2, 0.3, 0.4, 0.5]]),
            np.array([[0.2, 0.3, 0.4, 0.5]]),
            np.array([[]]).astype(np.float32).reshape([0, 4]),
            np.array([[]]).astype(np.float32).reshape([0, 4]),
        ]
        self.detection_object_classes = [
            np.array([1, 2, 1]),
            np.array([2, 1]),
            np.array([2]),
            np.array([2]),
            np.array([]).reshape([0]),
            np.array([]).reshape([0]),
        ]
        self.detection_object_scores = [
            np.array([0.8, 0.9, 0.3]),
            np.array([0.5, 0.4]),
            np.array([0.6]),
            np.array([0.8]),
            np.array([]).reshape([0]),
            np.array([]).reshape([0]),
        ]

        self.result_must = [
            {"matched_scores_per_class": [np.array([0.8, 0.3]),
                                          np.array([0.9])],
             "matched_true_pos_false_pos_per_class": [np.array([1, 1]),
                                                      np.array([0])],
             "correctly_detected_classes": np.array([1, 0]),
             "number_of_groundtruth_objects_per_class": np.array([2, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([1, 0])},
            {"matched_scores_per_class": [np.array([0.4]), np.array([0.5])],
             "matched_true_pos_false_pos_per_class": [np.array([0]),
                                                      np.array([0])],
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": [np.array([]), np.array([0.6])],
             "matched_true_pos_false_pos_per_class": [np.array([]),
                                                      np.array([1])],
             "correctly_detected_classes": np.array([0, 1]),
             "number_of_groundtruth_objects_per_class": np.array([0, 1]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 1])},
            {"matched_scores_per_class": [np.array([]), np.array([0.8])],
             "matched_true_pos_false_pos_per_class": [np.array([]),
                                                      np.array([0])],
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": [np.array([]), np.array([])],
             "matched_true_pos_false_pos_per_class": [np.array([]),
                                                      np.array([])],
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": [np.array([]), np.array([])],
             "matched_true_pos_false_pos_per_class": [np.array([]),
                                                      np.array([])],
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([1, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([1, 0])},
        ]

    def test_process(self):
        kpi_plugin = DetectionsMatcherKPIPlugin(
            num_classes=self.num_classes,
            matching_iou_threshold=0.5).build()
        for i in range(self.number_of_samples):
            result_sample = kpi_plugin.process(
                groundtruth_object_boxes=self.groundtruth_object_boxes[i],
                groundtruth_object_classes=self.groundtruth_object_classes[i],
                detection_object_boxes=self.detection_object_boxes[i],
                detection_object_classes=self.detection_object_classes[i],
                detection_object_scores=self.detection_object_scores[i],
            )
            self.assertAllClose(self.result_must[i],
                                result_sample)


class TestObjectDetectionKPIAccumulator(tf.test.TestCase):

    def setUp(self):
        self.num_classes = 2
        self.number_of_samples = 6

        self.data = [
            {"matched_scores_per_class": (np.array([0.8, 0.3]),
                                          np.array([0.9])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([1, 1]).astype(float), np.array([0]).astype(float)),
             "correctly_detected_classes": np.array([1, 0]),
             "number_of_groundtruth_objects_per_class": np.array([2, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([1, 0])},
            {"matched_scores_per_class": (np.array([0.4]), np.array([0.5])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([0]).astype(float), np.array([0]).astype(float)),
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": (np.array([]), np.array([0.6])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([]).astype(float), np.array([1]).astype(float)),
             "correctly_detected_classes": np.array([0, 1]),
             "number_of_groundtruth_objects_per_class": np.array([0, 1]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 1])},
            {"matched_scores_per_class": (np.array([]), np.array([0.8])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([]).astype(float), np.array([0]).astype(float)),
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": (np.array([]), np.array([])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([]).astype(float), np.array([]).astype(float)),
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([0, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([0, 0])},
            {"matched_scores_per_class": (np.array([]), np.array([])),
             "matched_true_pos_false_pos_per_class": (
                 np.array([]).astype(float), np.array([]).astype(float)),
             "correctly_detected_classes": np.array([0, 0]),
             "number_of_groundtruth_objects_per_class": np.array([1, 0]),
             "number_of_samples_with_groundtruth_per_class": np.array([1, 0])},
        ]

        self.kpi_must = {
            'AP-classwise-class_1': 0.5555555555555556,
            'AP-classwise-class_2': 0.3333333333333333,
            'CorLoc-classwise-class_1': 0.5,
            'CorLoc-classwise-class_2': 1.0,
            'Precision-classwise-class_1': np.array([1, 0.5, 2 / 3]),
            'Precision-classwise-class_2': np.array([0., 0., 1 / 3, 1 / 4]),
            'Recall-classwise-class_1': np.array([1 / 3, 1 / 3, 2 / 3]),
            'Recall-classwise-class_2': np.array([0., 0., 1., 1.]),
            'mAP': 0.4444444444444444,
            'meanCorLoc': 0.75,
            'Best_confidence_for_f1-score/class_id-1-name-class_1': 0.3,
            'Best_f1-score/class_id-1-name-class_1': 2 / 3,
            'Best_confidence_for_f1-score/class_id-2-name-class_2': 0.6,
            'Best_f1-score/class_id-2-name-class_2': 0.5,
        }

    def test_evaluate_on_sample(self):
        kpi_accumulator = ObjectDetectionKPIAccumulator(
            num_classes=self.num_classes,
            evaluate_corlocs=True, evaluate_precision_recall=True).build()
        for i_sample, each_sample in enumerate(self.data):
            evaluate_flag = i_sample == len(self.data) - 1
            _ = kpi_accumulator.evaluate_on_sample(evaluate=evaluate_flag,
                                                   **each_sample)
        self.assertAllClose(self.kpi_must,
                            kpi_accumulator.last_kpi)
